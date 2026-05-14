import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    """RMSNorm: faster than LayerNorm, no mean subtraction, no bias."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE). Applied only in SWA blocks."""

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(
            seq_len,
            device=self.inv_freq.device,  # type: ignore[attr-defined]
            dtype=self.inv_freq.dtype,  # type: ignore[attr-defined]
        )
        freqs = torch.outer(t, self.inv_freq)  # type: ignore[attr-defined]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """x: [B, H, T, head_dim], cos/sin: [T, head_dim]."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


class FfnLayer(nn.Module):
    """
    SwiGLU feed-forward layer (bias-free).
    Returns the delta only — caller is responsible for the residual add.
    ff_dim = ⌊8/3 × dim⌋ compensates for the extra gate projection.
    """

    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        expected = int(8 / 3 * dim)
        assert ff_dim == expected, (
            f"SwiGLU ff_dim should be ⌊8/3 × dim⌋ = {expected}, got {ff_dim}"
        )
        self.norm = RMSNorm(dim)
        self.gate_proj = nn.Linear(dim, ff_dim, bias=False)
        self.up_proj = nn.Linear(dim, ff_dim, bias=False)
        self.down_proj = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class ConvSubModule(nn.Module):
    """
    Conformer conv sub-module (bias-free):
      RMSNorm → pointwise expand + GLU → depthwise Conv1d
      → RMSNorm + SiLU → pointwise contract
    Returns the delta only — caller is responsible for the residual add.
    """

    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.pointwise_expand = nn.Linear(dim, 2 * dim, bias=False)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=dim,
            bias=False,
        )
        self.conv_norm = RMSNorm(dim)
        self.pointwise_contract = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)  # [B, T, dim]
        h = self.pointwise_expand(h)  # [B, T, 2*dim]
        h, gate = h.chunk(2, dim=-1)
        h = h * F.silu(gate)  # SwiGLU gate → [B, T, dim]
        h = h.transpose(1, 2)  # [B, dim, T]
        h = self.depthwise(h)
        h = h.transpose(1, 2)  # [B, T, dim]
        h = F.silu(self.conv_norm(h))
        return self.pointwise_contract(h)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with GQA (bias-free).

    window_size=None  → full bidirectional attention, NoPE.
                        Handles 30k-frame inference without positional extrapolation.
    window_size=int   → sliding-window attention, RoPE + QK-norm via flex_attention
                        on CUDA; falls back to full F.scaled_dot_product_attention
                        (without window constraint) on MPS for smoke-testing.

    Returns the delta only — caller is responsible for the residual add.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        window_size: int | None,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        self.norm = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        if window_size is not None:
            self.rotary = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.norm(x)

        q = self.q_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.window_size is not None:
            cos, sin = self.rotary(T)
            q = _apply_rotary_emb(q, cos, sin)
            k = _apply_rotary_emb(k, cos, sin)

        if x.device.type == "cuda" and self.window_size is not None:
            out = self._swa_flex(q, k, v, padding_mask, B, T)
        else:
            # Full attention: NoPE blocks on all devices, and SWA fallback on MPS.
            attn_mask = (
                self._padding_bias(padding_mask, q.dtype)
                if padding_mask is not None
                else None
            )
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                enable_gqa=True,
            )

        out = (
            out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        )
        return self.o_proj(out)

    def _swa_flex(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        padding_mask: torch.Tensor | None,
        B: int,
        T: int,
    ) -> torch.Tensor:
        ws = self.window_size

        def swa_mask(b, h, q_idx, kv_idx):
            return abs(q_idx - kv_idx) <= ws

        block_mask = create_block_mask(
            swa_mask,
            B,
            self.num_heads,
            T,
            T,
            device=q.device,
        )

        if padding_mask is not None:
            # log: 0.0 for valid frames, -inf for padding frames
            log_pm = padding_mask.float().log()

            def score_mod(score, b, h, q_idx, kv_idx):
                return score + log_pm[b, kv_idx]

            return flex_attention(  # type: ignore[return-value]
                q,
                k,
                v,
                block_mask=block_mask,
                score_mod=score_mod,
                enable_gqa=True,
            )

        return flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)  # type: ignore[return-value]

    @staticmethod
    def _padding_bias(padding_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """[B, T] bool → [B, 1, 1, T] additive float bias (-inf for padding)."""
        B, T = padding_mask.shape
        bias = torch.zeros(B, 1, 1, T, device=padding_mask.device, dtype=dtype)
        return bias.masked_fill(~padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))


class ConformerBlock(nn.Module):
    """
    Macaron-style Conformer block:
      x = x + ½·FFN1(x)
      x = x + Attention(x)
      x = x + Conv(x)
      x = x + ½·FFN2(x)
      x = RMSNorm(x)          ← final norm outside the residual stream
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        ff_dim: int,
        kernel_size: int,
        window_size: int | None,
    ):
        super().__init__()
        self.ff1 = FfnLayer(dim, ff_dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, num_kv_heads, window_size)
        self.conv = ConvSubModule(dim, kernel_size)
        self.ff2 = FfnLayer(dim, ff_dim)
        self.norm = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x, padding_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class ConformerEncoder(nn.Module):
    """
    Input projection + stack of ConformerBlocks.

    Full-attention (NoPE) blocks are placed every 4 blocks
    (indices 3, 7, 11, …) for ~25% global context without positional
    extrapolation issues at inference time.
    All other blocks use sliding-window attention (SWA) with RoPE + QK-norm.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        ff_dim: int,
        kernel_size: int,
        swa_window_size: int,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.input_proj = nn.Linear(input_dim, d_model, bias=False)

        step = 4  # every 4th block is NoPE/full-attention (~25%)
        full_attn_indices = set(range(step - 1, num_layers, step))

        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    dim=d_model,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    ff_dim=ff_dim,
                    kernel_size=kernel_size,
                    window_size=None if i in full_attn_indices else swa_window_size,
                )
                for i in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, padding_mask, use_reentrant=False)  # pyright: ignore[reportAssignmentType]
            else:
                x = block(x, padding_mask)
        return x


class YohaneFA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        output_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        ff_dim: int,
        kernel_size: int,
        swa_window_size: int,
        gradient_checkpointing: bool,
    ):
        super().__init__()
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            ff_dim=ff_dim,
            kernel_size=kernel_size,
            swa_window_size=swa_window_size,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.head = nn.Linear(d_model, output_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.head(self.encoder(x, padding_mask))
