import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with EMA codebook updates.
    Fully AMP / autocast safe: codebook always stays float32.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        restart_threshold: float = 1.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.restart_threshold = restart_threshold

        embed = torch.randn(num_embeddings, embedding_dim)
        embed = F.normalize(embed, dim=1) * 0.1
        self.register_buffer("embedding", embed)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embed_sum", embed.clone())
        self.register_buffer("usage_count", torch.zeros(num_embeddings))

    def forward(self, inputs):
        b, c, t = inputs.shape
        flat = inputs.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim).float()

        distances = (
            torch.sum(flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding ** 2, dim=1)
            - 2 * torch.matmul(flat, self.embedding.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        quantized = torch.matmul(encodings, self.embedding).view(b, t, c)

        if self.training:
            with torch.no_grad():
                cluster_size = encodings.sum(0)
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1.0 - self.decay)

                embed_sum = torch.matmul(encodings.t(), flat)
                self.ema_embed_sum.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)

                n = self.ema_cluster_size.sum()
                cluster_size_smooth = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon)
                    * n
                )
                embed_normalized = self.ema_embed_sum / cluster_size_smooth.unsqueeze(1)
                self.embedding.copy_(embed_normalized)

                self.usage_count.add_(cluster_size)

                if self.restart_threshold > 0:
                    dead = (self.ema_cluster_size < self.restart_threshold).nonzero(as_tuple=False).view(-1)
                    if dead.numel() > 0 and flat.size(0) > 0:
                        rand_idx = torch.randint(0, flat.size(0), (dead.numel(),), device=flat.device)
                        replacement = flat[rand_idx].float()
                        self.embedding[dead] = replacement
                        self.ema_embed_sum[dead] = replacement
                        self.ema_cluster_size[dead] = 1.0
                        self.usage_count[dead] = 0

        e_latent_loss = F.mse_loss(quantized.detach(), flat.view(b, t, c))
        loss = self.commitment_cost * e_latent_loss

        quantized = flat.view(b, t, c) + (quantized - flat.view(b, t, c)).detach()
        quantized = quantized.to(dtype=inputs.dtype)
        quantized = quantized.permute(0, 2, 1).contiguous()
        encoding_indices = encoding_indices.view(b, t)

        return quantized, loss, encoding_indices

    def get_codebook_usage(self):
        return int((self.usage_count > 0).sum().item())

