import torch
import torch.nn as nn

class MusicTransformer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, num_layers, num_heads):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Standard codebook embeddings + SOS token
        self.token_embeddings = nn.Embedding(num_embeddings + 1, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer.enable_checkpointing = True
        self.fc_out = nn.Linear(embedding_dim, num_embeddings + 1)

    def forward(self, x):
        # x is now strictly tokens in range [0, 512]
        seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        pos = self._positional_encoding(seq_len, x.device)
        
        out = self.token_embeddings(x) + pos
        out = self.transformer(out, mask=mask, is_causal=True)
        return self.fc_out(out)

    def _positional_encoding(self, seq_len: int, device=None) -> torch.Tensor:
        pe = torch.zeros(seq_len, self.embedding_dim, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2, dtype=torch.float, device=device) * (-torch.log(torch.tensor(10000.0, device=device)) / self.embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)
    

    