import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log2

from .utils import HelperModule
from typing import Callable, Dict
from functools import partial

class Residual(HelperModule):
    def build(self, 
            net: nn.Module, 
            use_rezero: bool = False, 
            rezero_init: float = 0.0
        ):
        self.net = net
        self.alpha = nn.Parameter(torch.FloatTensor(rezero_init)) if use_rezero else 1.0

    def forward(self, x):
        return self.net(x) * self.alpha + x

class ResidualLayer(HelperModule):
    def build(self,
        in_dim: int,
        residual_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1, # TODO: padding='same'
        bias: bool = True,
        use_batch_norm: bool = True,
        use_rezero: bool = False,
        activation: Callable = F.silu
    ):
        self.activation = activation
        layers = nn.Sequential(
            nn.Conv2d(in_dim, residual_dim, kernel_size, 
                stride=stride, 
                padding=padding, 
                bias=bias
            ),
            nn.BatchNorm2d(residual_dim) if use_batch_norm else nn.Identity(),
            activation,

            nn.Conv2d(residual_dim, in_dim, 1,
                stride=1,
                bias=bias
            ),
            nn.BatchNorm2d(residual_dim) if use_batch_norm else nn.Identity(),
        )
        self.layers = Residual(layers, use_rezero=use_rezero, rezero_init=0.0)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.activation(self.layers(x))

class ConvDown(HelperModule):
    def build(self,
        num_residual_layers: int,
        in_dim: int,
        out_dim: int,
        residual_dim: int,
        resample_factor: int,
        resample_method: str = 'conv', # 'max', 'conv', 'auto'
        residual_kernel_size: int = 3,
        residual_stride: int = 1,
        residual_padding: int = 1,
        residual_bias: int = 1,
        use_batch_norm: bool = True,
        use_rezero: bool = False,
        activation: Callable = F.silu,
    ):
        assert log2(resample_factor).is_integer(), f"Downsample factor must be a power of 2! Got '{resample_factor}'"

        if resample_method == 'auto':
            resample_method = 'max' if resample_factor == 2 else 'conv'

        if resample_method == 'conv':
            self.downsample = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(in_dim if i == 0 else out_dim, out_dim, 4, stride=2, padding=1), 
                    nn.BatchNorm2d(out_dim) if use_batch_norm else nn.Identity(), 
                    activation
                )
            for i in range(log2(resample_factor))])
        elif resample_method == 'max':
            self.downsample = nn.MaxPool2d(resample_factor, stride=resample_factor)
        else:
            raise ValueError(f"Unknown resample method '{resample_method}'!")

        self.residual = nn.Sequential(*[
            ResidualLayer(
                in_dim=out_dim, 
                residual_dim=residual_dim,
                kernel_size=residual_kernel_size,
                stride=residual_stride,
                padding=residual_padding,
                bias=residual_bias,
                use_batch_norm=use_batch_norm,
                use_rezero=use_rezero,
                activation=activation
            )
        for _ in range(num_residual_layers)])

    def forward(self, x):
        x = self.downsample(x)
        return self.residual(x)

class ConvUp(HelperModule):
    def build(self,
        num_residual_layers: int,
        in_dim: int,
        residual_dim: int,
        resample_factor: int,
        resample_method: str = 'conv', # 'max', 'conv', 'auto'
        residual_kernel_size: int = 3,
        residual_stride: int = 1,
        residual_padding: int = 1,
        residual_bias: int = 1,
        use_batch_norm: bool = True,
        use_rezero: bool = False,
        activation: Callable = F.silu,
    ):
        assert log2(resample_factor).is_integer(), f"Downsample factor must be a power of 2! Got '{resample_factor}'"

        self.residual = nn.Sequential(*[
            ResidualLayer(
                in_dim=in_dim, 
                residual_dim=residual_dim,
                kernel_size=residual_kernel_size,
                stride=residual_stride,
                padding=residual_padding,
                bias=residual_bias,
                use_batch_norm=use_batch_norm,
                use_rezero=use_rezero,
                activation=activation
            )
        for _ in range(num_residual_layers)])

        if resample_method == 'auto':
            resample_method = 'conv' # interpolate probably isn't good, so just make 'auto' be conv on upsample

        if resample_method == 'conv':
            self.upsample = nn.Sequential(*[
                nn.Sequential(
                    nn.ConvTranspose2d(in_dim, in_dim, 4, stride=2, padding=1, output_padding=1), 
                    nn.BatchNorm2d(in_dim) if use_batch_norm else nn.Identity(), 
                    activation
                )
            for _ in range(log2(resample_factor))])
        elif resample_method == 'interpolate':
            self.upsample = partial(F.interpolate, scale_factor=(resample_factor, resample_factor), mode='bilinear')
        else:
            raise ValueError(f"Unknown resample method '{resample_method}'!")

    def forward(self, x):
        x = self.residual(x)
        return self.upsample(x)