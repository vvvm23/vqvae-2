import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import accelerate

from .utils import HelperModule

class VQLayer(HelperModule):
    def build(self,
        in_dim: int,
        embedding_dim: int,
        codebook_size: int,
        decay: float = 0.99,
        eps: float = 1e-5,
        embedding_dtype: torch.dtype = torch.float32
    ):
        self.codebook_size = codebook_size
        self.decay = decay
        self.eps = eps
        self.conv_in = nn.Conv2d(in_dim, embedding_dim, 1) 

        # TODO: smarter init?
        embeddings = torch.randn(embedding_dim, codebook_size, dtype=embedding_dtype)
        self.register_buffer('embeddings', embeddings)
        self.register_buffer('cluster_sizes', torch.zeros(codebook_size, dtype=torch.float32)) 
        self.register_buffer('embeddings_avg', embeddings.clone())

    # TODO: generally, check for efficiency and correctness
    def forward(self, x):
        x = rearrange(self.conv_in(x), 'N c h w -> N h w c')
        z = rearrange(x, 'N h w c -> (N h w) c')
        cluster_distances = (
            z.pow(2).sum(dim=-1, keepdim=True)
            - 2 * z @ self.embeddings
            + self.embeddings.pow(2).sum(dim=0, keepdim=True) # TODO: can this square be cached using reparam?
        )
        _, embedding_idx = (-cluster_distances).max(dim=-1)

        embedding_onehot = F.one_hot(embedding_idx, self.codebook_size).to(z.dtype) # TODO: is onehot needed in eval mode?
        
        embedding_idx = rearrange(embedding_idx, '(N h w) -> N h w', N=x.shape[0], h=x.shape[1], w=x.shape[2])
        q = F.embedding(embedding_idx, self.embeddings.T)

        if self.training:
            # TODO: can this be replaced with counter on idx? can avoid building one hot matrix maybe?
            embedding_onehot_sum = accelerate.utils.reduce(embedding_onehot.sum(dim=0))
            embedding_sum = accelerate.utils.reduce(z.T @ embedding_onehot)

            self.cluster_sizes.data.mul_(self.decay).add_(
                embedding_onehot_sum, alpha=1-self.decay
            )
            self.embeddings_avg.data.mul_(self.decay).add_(embedding_sum, alpha=1-self.decay)
            n = self.cluster_sizes.sum()
            cluster_sizes = (
                (self.cluster_sizes + self.eps) / (n + self.codebook_size * self.eps) * n
            ).unsqueeze(0)

            self.embeddings.data.copy_(self.embeddings_avg / cluster_sizes)

        diff = (q.detach() - x).pow(2).mean() 
        q = x + (q - x).detach() # allows gradient flow through `x`

        return rearrange(q, 'N h w c -> N c h w'), embedding_idx, diff


if __name__ == '__main__':
    codebook = VQLayer(16, 8, 1024)
    x = torch.randn(4, 16, 14, 14)

    z, idx, diff = codebook(x)

    print(x.shape, z.shape)