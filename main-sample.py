import torch
import torchvision
from torchvision.utils import save_image

from tqdm import tqdm
import argparse
import datetime
import time
from pathlib import Path
from math import sqrt

from datasets import get_dataset
from hps import HPS_VQVAE, HPS_PIXEL
from helper import get_device, get_parameter_count

def load_vqvae(path, cfg, device):
    net = VQVAE(in_channels=cfg.in_channels, 
            hidden_channels=cfg.hidden_channels, 
            embed_dim=cfg.embed_dim, 
            nb_entries=cfg.nb_entries, 
            nb_levels=cfg.nb_levels, 
            scaling_rates=cfg.scaling_rates).to(device)
    net.load_state_dict(torch.load(path))
    net.eval()
    return net

def load_pixelsnail(path, cfg, level, device):
    lcfg = cfg.level[args.level]
    nb_cond = len(cfg.level) - level - 1
    net = PixelSnail(
        shape =                 cfg.code_shape, # TODO: unavailable in this script, calculate from scaling rates?
        nb_class =              cfg.nb_entries,
        channel =               lcfg.channel,
        kernel_size =           lcfg.kernel_size,
        nb_pixel_block =        lcfg.nb_block,
        nb_res_block =          lcfg.nb_res_block,
        res_channel =           lcfg.nb_res_channel,
        dropout =               lcfg.dropout,

        nb_cond =               nb_cond,
        nb_cond_res_block =     lcfg.nb_cond_res_block if nb_cond else 0,
        cond_res_channel =      lcfg.nb_cond_res_channel if nb_cond else 0,

        nb_out_res_block =      lcfg.nb_out_res_block,
    ).to(device)
    net.load_state_dict(torch.load(path))
    net.eval()
    return net

@torch.no_grad()
@torch.cuda.amp.autocast()
def vqvae_decode(net, codes):
    pass

@torch.no_grad()
@torch.cuda.amp.autocast()
def pixelsnail_sample(net, cs, nb_samples):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vqvae_path', type=str)
    parser.add_argument('pixelsnail_path', type=str, nargs='+')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--task', type=str, default='cifar10')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--no-tqdm', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--save-jpg', action='store_true')
    args = parser.parse_args()

    hps_vqvae, hps_pixel = HPS_VQVAE[args.task], HPS_PIXEL[args.task]

    save_id = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if args.batch_size:
        cfg.batch_size = args.batch_size

    device = get_device(args.cpu)
    vqvae = load_vqvae(args.vqvae_path, hps_vqvae, device)
    pixelsnails = [load_pixelsnail(p, hps_pixel, l, device) for l, p in enumerate(args.pixelsnail_path)]
