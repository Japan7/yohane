import torch
from torch import nn


class TdnnLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size // 2) * dilation,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.batch_norm(self.relu(self.conv(x))))


class FfnLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, dropout: float):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.relu(self.linear(x))) + x


class TdnnFfn(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, *, dropout: float):
        super().__init__()
        self.tdnn = nn.Sequential(
            TdnnLayer(
                input_dim,
                hidden_dim,
                kernel_size=5,
                stride=1,
                dilation=1,
                dropout=dropout,
            ),
            TdnnLayer(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                dilation=2,
                dropout=dropout,
            ),
            TdnnLayer(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                dilation=4,
                dropout=dropout,
            ),
            TdnnLayer(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                dilation=8,
                dropout=dropout,
            ),
            TdnnLayer(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                dilation=16,
                dropout=dropout,
            ),
            TdnnLayer(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                dilation=32,
                dropout=dropout,
            ),
            TdnnLayer(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                dilation=64,
                dropout=dropout,
            ),
        )
        self.ffn = nn.Sequential(
            FfnLayer(hidden_dim, hidden_dim, dropout=dropout),
            FfnLayer(hidden_dim, hidden_dim, dropout=dropout),
            FfnLayer(hidden_dim, hidden_dim, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.tdnn(x)
        x = x.transpose(1, 2)
        x = self.ffn(x)
        return x


class YohaneFA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        dropout: float,
    ):
        super().__init__()
        self.encoder = TdnnFfn(input_dim, hidden_dim, dropout=dropout)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
