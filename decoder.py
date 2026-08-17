import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import ResidualBlock


class AudioDecoder(nn.Module):
    """
    Memory-friendly multi-stage decoder matching the encoder.
    Total upsampling factor = 32.
    """
    def __init__(
        self,
        out_channels: int,
        in_channels: int,
        hidden_channels: int,
        residual_channels: int,
        num_residual_layers: int,
        stride: int = 32,
    ):
        super().__init__()
        self.stride = stride
        self.stage_strides = self._factorize_stride(stride)

        self.proj_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        layers = []
        current_channels = hidden_channels

        # Upsample in reverse order of the encoder stages
        for s in reversed(self.stage_strides):
            for _ in range(num_residual_layers):
                layers.append(ResidualBlock(current_channels, residual_channels))

            layers.append(
                nn.ConvTranspose1d(
                    current_channels,
                    hidden_channels,
                    kernel_size=2 * s,
                    stride=s,
                    padding=s // 2,
                    output_padding=0,
                )
            )
            layers.append(nn.ReLU())  # inplace=False required for residual + AMP
            current_channels = hidden_channels

        self.backbone = nn.Sequential(*layers)
        self.proj_out = nn.Conv1d(hidden_channels, out_channels, kernel_size=7, padding=3)

    @staticmethod
    def _factorize_stride(total: int):
        if total <= 1:
            return [1]
        factors = []
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
        x = self.proj_in(x)
        x = self.backbone(x)
        x = self.proj_out(x)
        return x

