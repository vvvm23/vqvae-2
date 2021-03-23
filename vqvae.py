import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log2

from helper import HelperModule

class ReZero(HelperModule):
    def build(self, in_channels: int, res_channels: int):
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, res_channels, 3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(res_channels),
            nn.ReLU(),

            nn.Conv2d(res_channels, in_channels, 3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.layers(x) * self.alpha + x

class ResidualStack(HelperModule):
    def build(self, in_channels: int, res_channels: int, nb_layers: int):
        self.stack = nn.Sequential(*[ReZero(in_channels, res_channels) 
                        for _ in range(nb_layers)
                    ])

    def forward(self, x):
        return self.stack(x)

class Encoder(HelperModule):
    def build(self, 
            in_channels: int, hidden_channels: int, 
            res_channels: int, nb_res_layers: int,
            downscale_factor: int,
        ):
        assert log2(downscale_factor) % 1 == 0, "Downscale must be a power of 2"
        downscale_steps = int(log2(downscale_factor))
        layers = []
        c_channel, n_channel = in_channels, hidden_channels // 2
        for _ in range(downscale_steps):
            layers.append(nn.Sequential(
                nn.Conv2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.ReLU(),
            ))
            c_channel, n_channel = n_channel, hidden_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(ResidualStack(n_channel, res_channels, nb_res_layers))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Decoder(HelperModule):
    def build(self, 
            in_channels: int, hidden_channels: int, out_channels: int,
            res_channels: int, nb_res_layers: int,
            upscale_factor: int,
        ):
        assert log2(upscale_factor) % 1 == 0, "Downscale must be a power of 2"
        upscale_steps = int(log2(upscale_factor))
        layers = [nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1)]
        layers.append(ResidualStack(hidden_channels, res_channels, nb_res_layers))
        c_channel, n_channel = hidden_channels, hidden_channels // 2
        for _ in range(upscale_steps):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.ReLU(),
            ))
            c_channel, n_channel = n_channel, out_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

"""
    Almost directly taken from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
    No reason to reinvent this rather complex mechanism.

    Essentially handles the "discrete" part of the network, and training through EMA rather than 
    third term in loss function.

    TODO: Merge input conv layer into this CodeLayer
"""
class CodeLayer(HelperModule):
    def build(self, embed_dim: int, nb_entries: int):
        self.dim = embed_dim
        self.n_embed = nb_entries
        self.decay = 0.99
        self.eps = 1e-5

        embed = torch.randn(embed_dim, nb_entries)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(nb_entries))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, x):
        flatten = x.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # TODO: Replace this? Or can we simply comment out?
            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

"""
    Main VQ-VAE-2 Module, capable of support arbitrary number of levels

    TODO: Add forward function
    TODO: Cascade latent codes down correctly. Current channel counts are incorrect
"""
class VQVAE(HelperModule):
    def build(self,
            in_channels: int                = 3,
            hidden_channels: int            = 128,
            res_channels: int               = 32,
            nb_res_layers: int              = 2,
            nb_levels: int                  = 2,
            embed_dim: int                  = 64,
            nb_entries: int                 = 512,
            scaling_rates: list[int]        = [4, 2]
        ):
        self.nb_levels = nb_levels
        assert len(scaling_rates) == nb_levels, "Number of scaling rates not equal to number of levels!"

        self.encoders = nn.ModuleList([Encoder(in_channels, hidden_channels, res_channels, nb_res_layers, scaling_rates[0])])
        self.decoders = nn.ModuleList([Decoder(hidden_channels, hidden_channels, in_channels, res_channels, nb_res_layers, scaling_rates[0])])
        for sr in scaling_rates[1:]:
            self.encoders.append(Encoder(hidden_channels, hidden_channels, res_channels, nb_res_layers, sr))
            self.decoders.append(Decoder(hidden_channels, hidden_channels, hidden_channels, res_channels, nb_res_layers, sr))

        self.codebooks = nn.ModuleList([CodeLayer(embed_dim, nb_entries)])

    def forward(self, x):
        return x

if __name__ == '__main__':
    net = VQVAE()
