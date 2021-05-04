import torch
import torch.nn.functional as F

import argparse
import datetime
import time
from tqdm import tqdm
from pathlib import Path

from vqvae import VQVAE
from datasets import get_dataset
from hps import HPS
from helper import get_device, get_parameter_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--task', type=str, default='cifar10')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--no-tqdm', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    args = parser.parse_args()
    cfg = HPS[args.task]

    save_id = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    device = get_device(args.cpu)

    print(f"> Loading VQ-VAE-2 model")
    net = VQVAE(in_channels=cfg.in_channels, 
                hidden_channels=cfg.hidden_channels, 
                embed_dim=cfg.embed_dim, 
                nb_entries=cfg.nb_entries, 
                nb_levels=cfg.nb_levels, 
                scaling_rates=cfg.scaling_rates).to(device)
    net.load_state_dict(torch.load(args.path))
    print(f"> Number of parameters: {get_parameter_count(net)}")

    if args.batch_size:
        cfg.batch_size = args.batch_size

    if not args.no_save:
        latent_dir = Path(f"latent-data")
        latent_dir.mkdir(exist_ok=True)

    print(f"> Loading {cfg.display_name} dataset")
    train_loader, test_loader = get_dataset(args.task, cfg, shuffle_train=False, shuffle_test=False)

