import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels, residual_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, residual_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(residual_channels, channels, kernel_size=1)
        )
    def forward(self, x):
        return x + self.block(x)

class AudioEncoderOld(nn.Module):
    def __init__(self, in_channels, hidden_channels, residual_channels, num_residual_layers, stride=4):
        super().__init__()
        # Strided convolutions to compress the audio timeline 
        # For a stride of 4, we downsample by a factor of 4
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=4, stride=stride, padding=1)
        # self.conv_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=4, stride=stride, padding=1)
        
        self.res_layers = nn.Sequential(*[
            ResidualBlock(hidden_channels, residual_channels) 
            for _ in range(num_residual_layers)
        ])
        self.conv_out = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_layers(x)
        return self.conv_out(x)
    

# src/models/encoder.py

class AudioEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, residual_channels, num_residual_layers, stride=4, embedding_dim=64):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=4, stride=stride, padding=1)
        
        self.res_layers = nn.Sequential(*[
            ResidualBlock(hidden_channels, residual_channels) 
            for _ in range(num_residual_layers)
        ])
        # FIX: Ensure out_channels matches the quantizer's embedding_dim
        self.conv_out = nn.Conv1d(hidden_channels, embedding_dim, kernel_size=1) 

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_layers(x)
        return self.conv_out(x)
        
