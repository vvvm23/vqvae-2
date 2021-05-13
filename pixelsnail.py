"""
    Based heavily off this implementation: https://github.com/rosinality/vq-vae-2-pytorch/blob/master/pixelsnail.py

    Changes:
        - Some naming conventions are changed (don't use input as a variable!!!)
        - support for $n$ conditioning variables.
"""
from math import sqrt, prod
from functools import partial, lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from helper import HelperModule

def wn_linear(in_dim, out_dim):
    return nn.utils.weight_norm(nn.Linear(in_dim, out_dim))

class WNConv2d(HelperModule):
    def build(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        activation=None,
    ):
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        )
        self.out_channel = out_channel
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        self.kernel_size = kernel_size
        self.activation = activation

    def forward(self, x):
        out = self.conv(x)
        if self.activation is not None:
            out = self.activation(out)
        return out

def shift_down(x, size=1):
    return F.pad(x, [0, 0, size, 0])[:, :, : x.shape[2], :]
def shift_right(x, size=1):
    return F.pad(x, [size, 0, 0, 0])[:, :, :, : x.shape[3]]

class CausalConv2d(HelperModule):
    def build(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding='downright',
        activation=None,
    ):
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 2
        self.kernel_size = kernel_size

        if padding == 'downright':
            pad = [kernel_size[1] - 1, 0, kernel_size[0] - 1, 0]
        elif padding == 'down' or padding == 'causal':
            pad = kernel_size[1] // 2
            pad = [pad, pad, kernel_size[0] - 1, 0]

        self.causal = 0
        if padding == 'causal':
            self.causal = kernel_size[1] // 2
        self.pad = nn.ZeroPad2d(pad)

        self.conv = WNConv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=0,
            activation=activation,
        )

    def forward(self, x):
        out = self.pad(x)
        if self.causal > 0:
            self.conv.conv.weight_v.data[:, :, -1, self.causal :].zero_()
        out = self.conv(out)
        return out


class GatedResBlock(HelperModule):
    def build(
        self,
        in_channel,
        channel,
        kernel_size,
        conv='wnconv2d',
        activation=nn.ELU,
        dropout=0.1,
        auxiliary_channel=0,
        condition_dim=0,
    ):
        if conv == 'wnconv2d':
            conv_module = partial(WNConv2d, padding=kernel_size // 2)
        elif conv == 'causal_downright':
            conv_module = partial(CausalConv2d, padding='downright')
        elif conv == 'causal':
            conv_module = partial(CausalConv2d, padding='causal')

        self.activation = activation()
        self.conv1 = conv_module(in_channel, channel, kernel_size)

        if auxiliary_channel > 0:
            self.aux_conv = WNConv2d(auxiliary_channel, channel, 1)

        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv_module(channel, in_channel * 2, kernel_size)

        if condition_dim > 0:
            self.condition = WNConv2d(condition_dim, in_channel * 2, 1, bias=False)

        self.gate = nn.GLU(1)

    def forward(self, x, aux_input=None, condition=None):
        out = self.conv1(self.activation(x))

        if aux_input is not None:
            out = out + self.aux_conv(self.activation(aux_input))

        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if condition is not None:
            condition = self.condition(condition)
            out += condition

        out = self.gate(out)
        out += x

        return out

@lru_cache(maxsize=64)
def causal_mask(size):
    shape = [size, size]
    mask = np.triu(np.ones(shape), k=1).astype(np.uint8).T
    start_mask = np.ones(size).astype(np.float32)
    start_mask[0] = 0

    return (
        torch.from_numpy(mask).unsqueeze(0),
        torch.from_numpy(start_mask).unsqueeze(1),
    )

class CausalAttention(HelperModule):
    def build(self, query_channel, key_channel, channel, n_head=8, dropout=0.1):

        self.query = wn_linear(query_channel, channel)
        self.key = wn_linear(key_channel, channel)
        self.value = wn_linear(key_channel, channel)

        self.dim_head = channel // n_head
        self.n_head = n_head

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key):
        batch, _, height, width = key.shape

        def reshape(x):
            return x.view(batch, -1, self.n_head, self.dim_head).transpose(1, 2)

        query_flat = query.view(batch, query.shape[1], -1).transpose(1, 2)
        key_flat = key.view(batch, key.shape[1], -1).transpose(1, 2)
        query = reshape(self.query(query_flat))
        key = reshape(self.key(key_flat)).transpose(2, 3)
        value = reshape(self.value(key_flat))

        attn = torch.matmul(query, key) / sqrt(self.dim_head)
        mask, start_mask = causal_mask(height * width)
        mask = mask.type_as(query)
        start_mask = start_mask.type_as(query)
        attn = attn.masked_fill(mask == 0, -1e4)
        attn = torch.softmax(attn, 3) * start_mask
        attn = self.dropout(attn)

        out = attn @ value
        out = out.transpose(1, 2).reshape(
            batch, height, width, self.dim_head * self.n_head
        )
        out = out.permute(0, 3, 1, 2)

        return out

class PixelBlock(HelperModule):
    def build(
        self,
        in_channel,
        channel,
        kernel_size,
        nb_res_block,
        attention=True,
        dropout=0.1,
        condition_dim=0,
    ):
        resblocks = []
        for i in range(nb_res_block):
            resblocks.append(
                GatedResBlock(
                    in_channel,
                    channel,
                    kernel_size,
                    conv='causal',
                    dropout=dropout,
                    condition_dim=condition_dim,
                )
            )

        self.resblocks = nn.ModuleList(resblocks)
        self.attention = attention

        if attention:
            self.key_resblock = GatedResBlock(
                in_channel * 2 + 2, in_channel, 1, dropout=dropout
            )
            self.query_resblock = GatedResBlock(
                in_channel + 2, in_channel, 1, dropout=dropout
            )

            self.causal_attention = CausalAttention(
                in_channel + 2, in_channel * 2 + 2, in_channel // 2, dropout=dropout
            )

            self.out_resblock = GatedResBlock(
                in_channel,
                in_channel,
                1,
                auxiliary_channel=in_channel // 2,
                dropout=dropout,
            )
        else:
            self.out = WNConv2d(in_channel + 2, in_channel, 1)

    def forward(self, x, background, condition=None):
        out = x
        for resblock in self.resblocks:
            out = resblock(out, condition=condition)

        if self.attention:
            key_cat = torch.cat([x, out, background], 1)
            key = self.key_resblock(key_cat)
            query_cat = torch.cat([out, background], 1)
            query = self.query_resblock(query_cat)
            attn_out = self.causal_attention(query, key)
            out = self.out_resblock(out, attn_out)
        else:
            bg_cat = torch.cat([out, background], 1)
            out = self.out(bg_cat)

        return out


class CondResNet(HelperModule):
    def build(self, in_channel, channel, kernel_size, nb_res_block):
        blocks = [WNConv2d(in_channel, channel, kernel_size, padding=kernel_size // 2)]
        for i in range(nb_res_block):
            blocks.append(GatedResBlock(channel, channel, kernel_size))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class PixelSNAIL(HelperModule):
    def build(
        self,
        shape,
        nb_entries,
        channel=256,
        kernel_size=5,
        nb_block=4,
        nb_res_block=4,
        nb_res_channel=256,
        attention=True,
        dropout=0.1,
        nb_cond=0,
        scaling_rates=[],
        nb_cond_res_block=0,
        nb_cond_res_channel=0,
        cond_res_kernel=3,
        nb_out_res_block=0,
    ):
        height, width = shape

        self.nb_entries = nb_entries
        self.nb_cond = nb_cond
        self.scaling_rates = scaling_rates

        if kernel_size % 2 == 0:
            kernel = kernel_size + 1

        else:
            kernel = kernel_size

        self.horizontal = CausalConv2d(
            nb_entries, channel, [kernel // 2, kernel], padding='down'
        )
        self.vertical = CausalConv2d(
            nb_entries, channel, [(kernel + 1) // 2, kernel // 2], padding='downright'
        )

        coord_x = (torch.arange(height).float() - height / 2) / height
        coord_x = coord_x.view(1, 1, height, 1).expand(1, 1, height, width)
        coord_y = (torch.arange(width).float() - width / 2) / width
        coord_y = coord_y.view(1, 1, 1, width).expand(1, 1, height, width)
        self.register_buffer('background', torch.cat([coord_x, coord_y], 1))

        self.blocks = nn.ModuleList()

        for i in range(nb_block):
            self.blocks.append(
                PixelBlock(
                    channel,
                    nb_res_channel,
                    kernel_size,
                    nb_res_block,
                    attention=attention,
                    dropout=dropout,
                    condition_dim=nb_cond_res_channel,
                )
            )

        if nb_cond > 0:
            self.cond_nets = nn.ModuleList()
            for _ in range(nb_cond):
                self.cond_nets.append(CondResNet(
                    nb_entries, nb_cond_res_channel, cond_res_kernel, nb_cond_res_block
                ))
            self.cond_out = nn.Sequential(
                WNConv2d(nb_cond_res_channel*nb_cond, nb_cond_res_channel, cond_res_kernel, padding=1),
                nn.ELU(),
            )

        out = []

        for i in range(nb_out_res_block):
            out.append(GatedResBlock(channel, nb_res_channel, 1))

        out.extend([nn.ELU(inplace=True), WNConv2d(channel, nb_entries, 1)])

        self.out = nn.Sequential(*out)

    def forward(self, x, *condition, cache=None):
        assert len(condition) == self.nb_cond, "Expected {self.nb_cond} conditions! Got {len(condition)}!"
        if cache is None:
            cache = {}
        batch, height, width = x.shape
        x = (
            F.one_hot(x, self.nb_entries).permute(0, 3, 1, 2).type_as(self.background)
        )
        horizontal = shift_down(self.horizontal(x))
        vertical = shift_right(self.vertical(x))
        out = horizontal + vertical

        background = self.background[:, :, :height, :].expand(batch, 2, height, width)

        # if condition is not None:
        if len(condition) > 0:
            if 'condition' in cache:
                condition = cache['condition']
                condition = condition[:, :, :height, :]

            else:
                cx = []
                for i, c in enumerate(condition):
                    c = F.one_hot(c, self.nb_entries).permute(0, 3, 1, 2).type_as(self.background)
                    c = self.cond_nets[i](c)
                    c = F.interpolate(c, scale_factor=prod(self.scaling_rates[:i+1]))
                    cx.append(c)

                condition = self.cond_out(torch.cat(cx, dim=1))
                cache['condition'] = condition.detach().clone()
                condition = condition[:, :, :height, :]

                # condition = (
                    # F.one_hot(condition, self.nb_entries)
                    # .permute(0, 3, 1, 2)
                    # .type_as(self.background)
                # )
                # condition = self.cond_resnet(condition)
                # condition = F.interpolate(condition, scale_factor=2)
                # cache['condition'] = condition.detach().clone() condition = condition[:, :, :height, :]
        else:
            condition = None

        for block in self.blocks:
            out = block(out, background, condition=condition)

        out = self.out(out)
        return out, cache

if __name__ == '__main__':
    device = torch.device('cuda')
    net = PixelSNAIL(
        shape=(32,32),
        nb_entries=256,
        channel=128,
        kernel_size=5,
        nb_block=2,
        nb_res_block=2,
        nb_res_channel=128,
        nb_cond=2,
        scaling_rates=[2, 4],
        nb_cond_res_block=2,
        nb_cond_res_channel=128,
    ).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        x = torch.randint(0, 256, (1, 32, 32)).to(device)
        c1 = torch.randint(0, 256, (1, 16, 16))
        c2 = torch.randint(0, 256, (1, 4, 4))
        l, _ = net(x, c1, c2)
    print(l.shape, l.dtype)
    
    p = torch.softmax(l, dim=1)
    print(p.shape)
    y = torch.argmax(p, dim=1)
    print(y.shape)
