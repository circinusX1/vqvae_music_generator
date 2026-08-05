
import torch
import torch.nn as nn

class MusicTransformer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, num_layers, num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embedding_dim, num_embeddings + 1)

    def forward(self, x, ref_latent=None):
        seq_len = x.size(1)
        pos = self._positional_encoding(seq_len, x.device)
        out = self.token_embeddings(x) + pos

        if ref_latent is not None:
            style_feat = self.style_proj(ref_latent).unsqueeze(1)
            out = out + style_feat

        # Generate causal mask (required when is_causal=True)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        # Pass the mask explicitly
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
    
    