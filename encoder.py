import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Lightweight residual block (no GroupNorm → much lower VRAM)."""
    def __init__(self, channels: int, residual_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),  # inplace=False required for residual + AMP
            nn.Conv1d(channels, residual_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(residual_channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class AudioEncoder(nn.Module):
    """
    Memory-friendly multi-stage encoder for 8 GB GPUs.

    Total stride remains 32, but the first stage uses stride=4 so that
    sequence length drops quickly and intermediate activations stay smaller.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        residual_channels: int,
        num_residual_layers: int,
        stride: int = 32,
        embedding_dim: int = 256,
    ):
        super().__init__()
        self.stride = stride

        # Prefer a larger first stride to reduce memory pressure early
        self.stage_strides = self._factorize_stride(stride)

        layers = []
        current_channels = in_channels

        for s in self.stage_strides:
            out_ch = hidden_channels
            layers.append(
                nn.Conv1d(
                    current_channels,
                    out_ch,
                    kernel_size=2 * s,
                    stride=s,
                    padding=s // 2,
                )
            )
            layers.append(nn.ReLU())  # inplace=False

            for _ in range(num_residual_layers):
                layers.append(ResidualBlock(out_ch, residual_channels))

            current_channels = out_ch

        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Conv1d(hidden_channels, embedding_dim, kernel_size=1)

    @staticmethod
    def _factorize_stride(total: int):
        """
        Create stages that multiply to `total`.
        Start with a larger factor (4) when possible so the first feature map
        is already much shorter → big VRAM saving.
        """
        if total <= 1:
            return [1]

        factors = []
        # Take one larger stride first if possible
        if total % 4 == 0:
            factors.append(4)
            total //= 4
        elif total % 2 == 0:
            factors.append(2)
            total //= 2

        while total % 2 == 0 and total > 1:
            factors.append(2)
            total //= 2
        if total > 1:
            factors.append(total)
        return factors if factors else [1]

    def forward(self, x):
        x = self.backbone(x)
        x = self.proj(x)
        return x

