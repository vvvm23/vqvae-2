import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List, Optional, Literal

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
        codebook_init_type: Literal["normal", "kaiming_uniform"] = "kaiming_uniform",
        codebook_gumbel_temperature: float = 0.0,
        codebook_cosine: bool = True,
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
            init_type=codebook_init_type,
            cosine=codebook_cosine,
            gumbel_temperature=codebook_gumbel_temperature,
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

    def decode(self, z, c: Optional[torch.FloatTensor] = None):
        z = self.decoder(z, c)
        return self.out_conv(z)

    def decode_discrete(self, idx, c: Optional[torch.FloatTensor] = None):
        z = F.embedding(idx, self.embeddings.T)
        return self.decode(z, c)


class VQVAE2(HelperModule):
    def build(self, vqvaes: Tuple[VQVAE]):
        self.num_levels = len(vqvaes)
        self.vqvaes = nn.ModuleList(vqvaes)

    @classmethod
    def build_from_config(cls, config, **kwargs):
        config = dict(config)
        return cls.build_from_kwargs(config.pop("resample_factors"), **config, **kwargs)

    @classmethod
    def build_from_kwargs(cls, resample_factors: List[int] = None, **kwargs):
        assert resample_factors is not None
        vqvaes = []
        in_dim = kwargs.pop("in_dim")
        for f in resample_factors:
            vqvaes.append(VQVAE(in_dim=in_dim, **kwargs, resample_factor=f))
            in_dim = kwargs["hidden_dim"]

        return cls(vqvaes)

    # TODO: currently limited to only two levels as we only cascade condition from next highest
    @staticmethod
    def _hierarchical_forward(vqvae, x, vqvaes: List):
        z = vqvae.encoder.downsample(x)
        c, idx, diff = 0.0, [], 0.0
        if len(vqvaes):
            c, idx, diff = VQVAE2._hierarchical_forward(vqvaes[0], z, vqvaes[1:])
        # TODO: try concat conditioning later
        z = vqvae.encoder.residual(z + c)
        z, id, d = vqvae.codebook(z)

        z = vqvae.decode(z, c)
        return z, [id, *idx], diff + d

    def forward(self, x):
        return VQVAE2._hierarchical_forward(self.vqvaes[0], x, self.vqvaes[1:])


if __name__ == "__main__":
    device = torch.device("cuda")
    vqvae2 = VQVAE2.build_from_kwargs(
        in_dim=3, hidden_dim=128, codebook_dim=64, codebook_size=512, residual_dim=128, resample_factors=[4, 2]
    ).to(device=device, dtype=torch.float16)

    count = sum(p.numel() for p in vqvae2.parameters() if p.requires_grad)
    print(f"Number of parameters: {count:,}")

    x = torch.randn(4, 3, 256, 256).to(device=device, dtype=torch.float16)
    y, idx, diff = vqvae2(x)

    print(y.shape)
    print([i.shape for i in idx])
    print(diff)
