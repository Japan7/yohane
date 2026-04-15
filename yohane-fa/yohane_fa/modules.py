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
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class FfnLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(x))


class TdnnFfn(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.tdnn = nn.Sequential(
            TdnnLayer(input_dim, hidden_dim, kernel_size=5, stride=1),
            TdnnLayer(hidden_dim, hidden_dim, kernel_size=3, stride=1),
            TdnnLayer(hidden_dim, hidden_dim, kernel_size=3, stride=1),
        )
        self.ffn = nn.Sequential(
            FfnLayer(hidden_dim, hidden_dim),
            FfnLayer(hidden_dim, hidden_dim),
            FfnLayer(hidden_dim, hidden_dim),
            FfnLayer(hidden_dim, hidden_dim),
            FfnLayer(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.tdnn(x)
        x = x.transpose(1, 2)
        x = self.ffn(x)
        return x


class YohaneFA(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = TdnnFfn(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
