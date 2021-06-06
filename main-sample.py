import torch
import torchvision
from torchvision.utils import save_image

from tqdm import tqdm
import argparse
import datetime
import time
from pathlib import Path
from math import sqrt, prod

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
    # TODO: This, needs some functions in main class though
    pass

@torch.no_grad()
@torch.cuda.amp.autocast()
def pixelsnail_sample(net, cs, shape, nb_samples, device):
    sample = torch.zeros(nb_samples, *shape, dtype=torch.int64).to(device)
    cache = {}
    pb = tqdm(total=prod(shape)) # TODO: add option to disable
    for i in range(shape[0]):
        for j in range(shape[1]):
            pred, cache = model(sample, cs=cs cache={})
            pred = F.softmax(pred[:, :, i, j], dim=1) # TODO: needs temperature parameter
            sample[:, i, j] = torch.multinomial(pred, 1).squeeze()
            pb.update(1)

    return sample.cpu()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vqvae_path', type=str)
    parser.add_argument('pixelsnail_path', type=str, nargs='+')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--task', type=str, default='cifar10')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--nb-samples', type=int, default=None)
    parser.add_argument('--no-tqdm', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--save-jpg', action='store_true')
    args = parser.parse_args()

    hps_vqvae, hps_pixel = HPS_VQVAE[args.task], HPS_PIXEL[args.task]

    save_id = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    device = get_device(args.cpu)

    if args.batch_size:
        cfg.mini_batch_size = args.batch_size
    _, *shape  = hps_vqvae.image_shape

    print("> Loading VQ-VAE-2")
    vqvae = load_vqvae(args.vqvae_path, hps_vqvae, device)
    print("> Loading PixelSnail priors")
    pixelsnails = [load_pixelsnail(p, hps_pixel, l, device) for l, p in enumerate(args.pixelsnail_path)]

    codes = []
    for l in range(hps_vqvae.nb_levels-1, -1, -1):
        print(f"> Sampling from PixelSnail level {l}")
        scale_product = prod(hps_vqvae.scaling_rates[:l+1])
        latent_shape = (shape[0] // scale_product, shape[1] // scale_product)
        sample = pixelsnail_sample(pixelsnails[l], codes, args.nb_samples, device)
        codes.append(sample)
        print()

    print(f"> Decoding sampled latent codes using VQ-VAE")
    img = vqvae_decode(vqvae, codes)

    save_path = f"sample-{save_id}.{'jpg' if args.save_jpg else 'png'}"
    print(f"> Saving image to {save_path}")
    save_image(img, save_path, nrow=int(sqrt(args.nb_samples)), normalize=True, value_range=(-1,1))
