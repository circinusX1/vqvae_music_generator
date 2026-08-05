"""
Lightweight LoRA (Low-Rank Adaptation) implementation for 8GB GPU training.
No external dependencies (pure PyTorch).

Fixed to be compatible with torch.nn.MultiheadAttention / TransformerEncoderLayer
which access .weight and .bias directly on projection modules.
"""

import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """
    LoRA wrapper around a linear layer.
    Original weights are frozen; only low-rank matrices A and B are trained.

    Exposes .weight and .bias so that PyTorch's MultiheadAttention and
    TransformerEncoderLayer can still access them.
    """
    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.in_features = in_features
        self.out_features = out_features

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    # ------------------------------------------------------------------
    # Compatibility properties required by MultiheadAttention etc.
    # ------------------------------------------------------------------
    @property
    def weight(self):
        """Return the effective weight (original + LoRA delta) for external access."""
        if self.rank > 0:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            return self.original.weight + delta
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    def forward(self, x):
        # Original path (frozen)
        result = self.original(x)
        # LoRA path
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result + lora_out * self.scaling

    def merge(self):
        """Merge LoRA weights into the original linear for inference."""
        if self.rank > 0:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.original.weight.data += delta
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()


def apply_lora_to_linear(module: nn.Module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05, target_modules=None):
    """
    Recursively replace nn.Linear layers with LoRALinear.
    target_modules: list of name substrings to match.
    """
    if target_modules is None:
        # Prefer FFN layers; out_proj is also supported thanks to the .weight property
        target_modules = ['linear1', 'linear2', 'out_proj']

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if any(t in name.lower() for t in target_modules) or target_modules == ['*']:
                setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
        else:
            apply_lora_to_linear(child, rank=rank, alpha=alpha, dropout=dropout, target_modules=target_modules)


def get_lora_parameters(model: nn.Module):
    """Return only the LoRA parameters (for the optimizer)."""
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_params.append(module.lora_A)
            lora_params.append(module.lora_B)
    return lora_params


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def merge_lora_weights(model: nn.Module):
    """Merge all LoRA adapters into base weights (for clean inference checkpoints)."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()

