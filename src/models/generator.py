import torch
import torch.nn as nn

class MusicTransformer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, num_layers, num_heads):
        super().__init__()
        # +1 added for a specialized Start-Of-Sequence (SOS) token index
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.token_embeddings = nn.Embedding(num_embeddings + 1, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embedding_dim, num_embeddings + 1)

    def forward(self, x):
        # x shape: [B, T_tokens]
        seq_len = x.size(1)
        
        # Generate causal attention mask to prevent model from looking into the future
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        
        # Compute matching sinusoidal wave offsets for the current timeline slice
        pos = self._positional_encoding(seq_len, x.device)
        
        # Combine identity tokens with relative time positions
        out = self.token_embeddings(x) + pos
        
        # Pass through the causal autoregressive transformer block
        out = self.transformer(out, mask=mask, is_causal=True)
        
        return self.fc_out(out) # Output shape: [B, T_tokens, Codebook_Size]

    def _positional_encoding(self, seq_len: int, device=None) -> torch.Tensor:
        """
        Generates standard continuous sinusoidal positional encodings 
        with shape [1, seq_len, embedding_dim]
        """
        pe = torch.zeros(seq_len, self.embedding_dim, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        
        # Divide term scales frequencies across channels smoothly
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, dtype=torch.float, device=device) *
            (-torch.log(torch.tensor(10000.0, device=device)) / self.embedding_dim)
        )
        
        # Assign alternating sine and cosine patterns
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Safe handling for both odd and even embedding dimensions
        if self.embedding_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
            
        return pe.unsqueeze(0) # Output shape: [1, seq_len, embedding_dim]
    