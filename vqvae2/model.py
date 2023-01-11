import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Callable

from .utils import HelperModule, identity_activation
from .conv import ConvDown, ConvUp
from .vq import VQLayer

# Regular VQ-VAE without any hierarchical bits
class VQVAE(HelperModule):
    def build(self,
        in_dim: int,
        hidden_dim: int,
        codebook_dim: int,
        codebook_size: int,
        codebook_decay: float = 0.99,
        codebook_eps: float = 1e-5,
        codebook_dtype: torch.dtype = torch.float32,
        num_residual_layers: int = 2,
        residual_dim: int = 256,
        residual_kernel_size: int = 3,
        residual_stride: int = 1,
        residual_padding: int = 1,
        residual_bias: bool = True,
        resample_method: str = 'conv',
        resample_factor: int = 4,
        use_batch_norm: bool = True,
        use_rezero: bool = False,
        activation: Callable = F.silu,
        output_activation: Callable = identity_activation,
    ):
        # TODO: refactor arg passing a la lucidrains style
        self.encoder = ConvDown(
            in_dim=in_dim,
            out_dim=hidden_dim,
            residual_dim=residual_dim,
            resample_factor=resample_factor,
            num_residual_layers=num_residual_layers,
            resample_method=resample_method,
            residual_kernel_size=residual_kernel_size,
            residual_stride=residual_stride,
            residual_padding=residual_padding,
            residual_bias=residual_bias,
            use_batch_norm=use_batch_norm,
            use_rezero=use_rezero,
            activation=activation
        )

        self.codebook = VQLayer(
            in_dim=hidden_dim,
            embedding_dim=codebook_dim,
            codebook_size=codebook_size,
            decay=codebook_decay,
            eps=codebook_eps,
            embedding_dtype=codebook_dtype
        )

        # TODO: refactor args here too
        self.decoder = ConvUp(
            in_dim=hidden_dim,
            residual_dim=residual_dim,
            resample_factor=resample_factor,
            num_residual_layers=num_residual_layers,
            resample_method=resample_method,
            residual_kernel_size=residual_kernel_size,
            residual_stride=residual_stride,
            residual_padding=residual_padding,
            residual_bias=residual_bias,
            use_batch_norm=use_batch_norm,
            use_rezero=use_rezero,
            activation=activation
        )

        # TODO: parameterize?
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1),
            output_activation,
        )

    def encode(self, x):
        z = self.encoder(x)
        return self.codebook(z)

    def decode(self, z):
        z = self.decoder(z)
        return self.out_conv(z)

    def decode_discrete(self, idx):
        z = F.embedding(idx, self.embeddings.T)
        return self.decode(z)

    def forward(self, x):
        z, idx, diff = self.encode(x)
        return self.decode(z)

class VQVAE2(HelperModule):
    def build(self):
        pass

    def encode(self, x):
        pass

    def decode(self, x):
        pass

    def decode_discrete(self, x):
        pass

    def forward(self):
        pass
