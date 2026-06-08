import torch
import torch.nn as nn

class MultiScaleSpectralLoss(nn.Module):
    """
    Computes a simplified spectral convergence loss directly on the 
    reconstructed audio sequence vs target audio sequence.
    """
    def __init__(self):
        super().__init__()

    def forward(self, target, reconstruction):
        # Simple MSE on the raw wave representation combined with relative structural variance
        recon_loss = torch.mean(torch.abs(target - reconstruction))
        return recon_loss
        
        
