import torch
import torch.nn as nn
from encoder import ResidualBlock

class AudioDecoder(nn.Module):
    def __init__(self, out_channels, in_channels, hidden_channels, residual_channels, num_residual_layers, stride=4):
        super().__init__()
        # Map from encoder/quantizer output channels (in_channels / embedding_dim)
        # into the decoder's working hidden channel size
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        
        self.res_layers = nn.Sequential(*[
            ResidualBlock(hidden_channels, residual_channels) 
            for _ in range(num_residual_layers)
        ])
        
        # Transposed convolution to upscale timeline back to native resolution
        self.deconv_out = nn.ConvTranspose1d(
            hidden_channels, out_channels, 
            kernel_size=4, stride=stride, padding=1
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_layers(x)
        return self.deconv_out(x)
        
        
        
