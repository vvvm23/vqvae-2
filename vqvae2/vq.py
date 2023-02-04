import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import accelerate

from .utils import HelperModule
from typing import Literal

import logging
from accelerate.logging import get_logger

logger = get_logger(__file__)


def l2norm(t: torch.Tensor):
    return F.normalize(t, p=2, dim=-1)


# https://github.com/lucidrains/vector-quantize-pytorch/blob/61b821d876249977be5c8b6f7fac710f860f05db/vector_quantize_pytorch/vector_quantize_pytorch.py
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t: torch.Tensor):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t: torch.Tensor, temperature: float = 0.0, dim=-1):
    if temperature == 0.0:
        return t.argmax(dim=dim)
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


class VQLayer(HelperModule):
    def build(
        self,
        in_dim: int,
        embedding_dim: int,
        codebook_size: int,
        decay: float = 0.99,
        eps: float = 1e-5,
        embedding_dtype: torch.dtype = torch.float32,
        init_type: Literal["normal", "kaiming_uniform"] = "kaiming_uniform",
        cosine: bool = True,
        gumbel_temperature: float = 0.0,
    ):
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.eps = eps
        self.conv_in = nn.Conv2d(in_dim, embedding_dim, 1)
        self.cosine = cosine
        self.gumbel_temperature = gumbel_temperature

        if init_type == "normal":
            embeddings = torch.normal(mean=0.0, std=0.1, size=(embedding_dim, codebook_size)).to(embedding_dtype)
        elif init_type == "kaiming_uniform":
            embeddings = torch.empty((embedding_dim, codebook_size), dtype=embedding_dtype)
            nn.init.kaiming_uniform_(embeddings)
        else:
            raise ValueError("unrecognised `init_type`")

        if self.cosine:
            embeddings = l2norm(embeddings)

        self.register_buffer("embeddings", embeddings)
        self.register_buffer("cluster_sizes", torch.zeros(codebook_size, dtype=torch.float32))
        self.register_buffer("embeddings_avg", embeddings.clone())

    # TODO: generally, check for efficiency and correctness
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        dtype = x.dtype
        x = rearrange(self.conv_in(x.float()), "N c h w -> N h w c")
        z = rearrange(x, "N h w c -> (N h w) c")
        if self.cosine:
            z = l2norm(z)

        norm_embeddings = l2norm(self.embeddings) if self.cosine else self.embeddings
        cluster_distances = (
            z.pow(2).sum(dim=-1, keepdim=True)
            - 2 * z @ norm_embeddings
            + norm_embeddings.pow(2).sum(dim=0, keepdim=True)  # TODO: can this square be cached using reparam?
        )
        # TODO: do we need to change this in eval mode?
        embedding_idx = gumbel_sample(-cluster_distances, temperature=self.gumbel_temperature, dim=-1)

        embedding_onehot = F.one_hot(embedding_idx, self.codebook_size).to(
            z.dtype
        )  # TODO: is onehot needed in eval mode?

        embedding_idx = embedding_idx.reshape(*x.shape[:-1])
        q = F.embedding(embedding_idx, norm_embeddings.T)

        if self.training:
            # TODO: adding the all reduce breaks things.. unsure why
            # embedding_onehot_sum = accelerate.utils.reduce(embedding_onehot.sum(dim=0), reduction='sum')
            # embedding_sum = accelerate.utils.reduce(z.T @ embedding_onehot, reduction='sum')

            # TODO: can this be replaced with counter on idx? can avoid building one hot matrix maybe?
            embedding_onehot_sum = embedding_onehot.sum(dim=0)
            embedding_sum = z.T @ embedding_onehot

            self.cluster_sizes.data.mul_(self.decay).add_(embedding_onehot_sum, alpha=1 - self.decay)

            if self.cosine:
                embedding_sum = l2norm(embedding_sum)
            self.embeddings_avg.data.mul_(self.decay).add_(embedding_sum, alpha=1 - self.decay)
            n = self.cluster_sizes.sum()
            cluster_sizes = ((self.cluster_sizes + self.eps) / (n + self.codebook_size * self.eps) * n).unsqueeze(0)

            self.embeddings.data.copy_(self.embeddings_avg / cluster_sizes)

        diff = (q.detach() - x).pow(2).mean()
        q = x + (q - x).detach()  # allows gradient flow through `x`

        return rearrange(q, "N h w c -> N c h w").to(dtype), embedding_idx, diff


if __name__ == "__main__":
    codebook = VQLayer(16, 8, 1024)
    x = torch.randn(4, 16, 14, 14)

    z, idx, diff = codebook(x)

    print(x.shape, z.shape)
