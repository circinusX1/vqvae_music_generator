import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from lora import LoRALinear, apply_lora_to_linear, get_lora_parameters, merge_lora_weights


class MusicTransformer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, num_layers, num_heads,
                 use_lora: bool = False, lora_rank: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.05):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_lora = use_lora

        self.token_embeddings = nn.Embedding(num_embeddings + 1, embedding_dim)
        self.style_proj = nn.Linear(64, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        self.fc_out = nn.Linear(embedding_dim, num_embeddings + 1)

        if use_lora:
            # Apply LoRA to the heaviest linear layers inside the Transformer
            apply_lora_to_linear(
                self.transformer,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=['linear1', 'linear2', 'out_proj']
            )
            # Also wrap the final output projection
            self.fc_out = LoRALinear(self.fc_out, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

            # Freeze all non-LoRA parameters
            for name, param in self.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False

    def forward(self, x, ref_latent=None):
        seq_len = x.size(1)
        pos = self._positional_encoding(seq_len, x.device)
        out = self.token_embeddings(x) + pos

        if ref_latent is not None:
            style_feat = self.style_proj(ref_latent).unsqueeze(1)
            out = out + style_feat

        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        out = self.transformer(out, mask=mask, is_causal=True)
        return self.fc_out(out)

    def _positional_encoding(self, seq_len: int, device=None) -> torch.Tensor:
        pe = torch.zeros(seq_len, self.embedding_dim, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2, dtype=torch.float, device=device) * 
                           (-torch.log(torch.tensor(10000.0, device=device)) / self.embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)

    def get_trainable_params(self):
        if self.use_lora:
            return get_lora_parameters(self)
        return [p for p in self.parameters() if p.requires_grad]

    def merge_and_save(self, path):
        """Merge LoRA adapters into base weights and save a clean checkpoint for inference."""
        if self.use_lora:
            merge_lora_weights(self)
        torch.save(self.state_dict(), path)

