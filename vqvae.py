import torch
import torch.nn as nn
import torch.nn.functional as F

from helper import HelperModule

class ReZero(HelperModule):
    def build(self, in_channels: int, res_channels: int):
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, res_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(res_channels),
            nn.ReLU(),

            nn.Conv2d(res_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
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
    def build(self):
        pass

    def forward(self, x):
        pass

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
            in_channels: int = 3,
            nb_levels: int = 2,
        ):
        pass

    def forward(self, x):
        return x

if __name__ == '__main__':
    net = ResidualStack(3, 1, 8)
    x = torch.randn(1, 3, 8, 8)
    y = net(x)

    print(y)
    print(y.shape)
