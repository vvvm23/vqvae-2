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
    def build(self):
        pass

    def forward(self, x):
        pass

class VQLayer(HelperModule):
    def build(self):
        pass

    def forward(self, x):
        pass

class VQVAELevel(HelperModule):
    def build(self):
        pass

    def forward(self, x):
        pass

class VQVAE(HelperModule):
    def build(self,
            in_channels: int        = 3,
            hidden_channels: int    = 128,
            res_channels: int       = 32,
            nb_res_layers: int      = 2,
            nb_levels: int          = 2,
            embed_dim: int          = 64,
            nb_entries: int         = 512,
        ):
        pass

    def forward(self, x):
        return x

if __name__ == '__main__':
    net = Encoder(3, 64, 32, 2, 4)
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.shape)
