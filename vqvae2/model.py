import torch
import torch.nn as nn

from typing import List

from .utils import HelperModule
from .conv import ConvDown, ConvUp
from .vq import VQLayer

# Regular VQ-VAE without any hierarchical bits
class VQVAE(HelperModule):
    def build(self):
        pass

    def encode(self, x):
        pass

    def decode(self, x):
        pass

    def decode_discrete(self, x):
        pass

    def forward(self, x):
        pass

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
