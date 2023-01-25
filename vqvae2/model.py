import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from .utils import HelperModule
from .conv import ConvDown, ConvUp
from .vq import VQLayer

# Regular VQ-VAE without any hierarchical bits
class VQVAE(HelperModule):
    def build(
        self,
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
        resample_method: str = "conv",
        resample_factor: int = 4,
        use_batch_norm: bool = True,
        use_rezero: bool = False,
        activation: nn.Module = nn.SiLU,
        output_activation: nn.Module = nn.Identity,
    ):
        # TODO: store args that will be needed by higher level VQ-VAE-2
        # TODO: refactor arg passing a la lucidrains style
        self.encoder = ConvDown(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
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
            activation=activation,
        )

        self.codebook = VQLayer(
            in_dim=hidden_dim,
            embedding_dim=codebook_dim,
            codebook_size=codebook_size,
            decay=codebook_decay,
            eps=codebook_eps,
            embedding_dtype=codebook_dtype,
        )

        # TODO: refactor args here too
        self.decoder = ConvUp(
            in_dim=codebook_dim,
            hidden_dim=hidden_dim,
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
            activation=activation,
        )

        # TODO: parameterize?
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, in_dim, 3, stride=1, padding=1),
            output_activation(),
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
        return self.decode(z), idx, diff


class VQVAE2(HelperModule):
    def build(self, vqvaes: Tuple[VQVAE]):
        self.num_levels = len(vqvaes)
        self.vqvaes = nn.ModuleList(*vqvaes)

    def encode(self, x):
        zs, idx, total_diff = [], [], 0.0
        z = x
        for vqvae in self.vqvaes:
            z, id, diff = vqvae.encode(z)
            zs.append(z)
            idx.append(id)
            total_diff += diff

        return zs, idx, diff

    def decode(self, zs):
        pass

    def decode_discrete(self, x):
        pass

    def forward(self, x):
        zs, idx, diff = self.encode(x)
        return self.decode(zs), idx, diff


if __name__ == "__main__":
    vqvae = VQVAE(3, 256, 256, 256, residual_dim=256)
    count = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)
    print(f"Number of parameters: {count:,}")

    x = torch.randn(4, 3, 32, 32)
    y, idx, diff = vqvae(x)
    print(x.shape)
    print(y.shape)
    print(idx.shape)
    print(diff)
