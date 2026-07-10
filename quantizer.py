import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
        # self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        # fix 1
        self.embedding.weight.data.normal_(0, 0.1)


    def forward(self, inputs):
            # input shape: [B, C, T] -> transpose to channels-last: [B, T, C]
            b, c, t = inputs.shape
            inputs = inputs.permute(0, 2, 1).contiguous()
            flat_input = inputs.view(-1, self.embedding_dim)
            
            # 1. FIXED: Added .t() to match inner matrix dimensions (64 vs 64)
            distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                        + torch.sum(self.embedding.weight**2, dim=1)
                        - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
                
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

            usage = torch.unique(encoding_indices).numel() # fix 1
            # fixx 3 print(f"Codebook usage: {usage}/{self.num_embeddings}")

            encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)
            
            quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
            
            # Losses
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
            loss = e_latent_loss + self.commitment_cost * q_latent_loss
            
            # Straight-Through Estimator
            quantized = inputs + (quantized - inputs).detach()
            
            # 2. FIXED: Dynamically capture the compressed time-dimension size (t) 
            # instead of assuming it matches the batch shape directly
            ### >FIX encoding_indices = encoding_indices.view(b, t)
            encoding_indices = encoding_indices.view(b, -1)
            
            return quantized.permute(0, 2, 1).contiguous(), loss, encoding_indices

    def oldforward(self, inputs):
        # input shape: [B, C, T] -> transpose to channels-last: [B, T, C]
        inputs = inputs.permute(0, 2, 1).contiguous()
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Distance calculation: ||z_e(x) - e||^2
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        
        # Losses
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = e_latent_loss + self.commitment_cost * q_latent_loss
        
        # Straight-Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Convert indices back to match timeline resolution: [B, T_compressed]
        encoding_indices = encoding_indices.view(inputs.shape[0], inputs.shape[1])
        
        return quantized.permute(0, 2, 1).contiguous(), loss, encoding_indices
        
