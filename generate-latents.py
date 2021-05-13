import torch
import torch.nn.functional as F

from torchvision.utils import save_image

import argparse
import datetime
import time
import math
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
        dataset_path = latent_dir / f"{save_id}-latent-dataset.pt"

    print(f"> Loading {cfg.display_name} dataset")
    (train_loader, test_loader), (train_dataset, test_dataset) = get_dataset(
        args.task, cfg, 
        shuffle_train=False, shuffle_test=False,
        return_dataset=True
    )
    train_dataset_len, test_dataset_len = len(train_dataset), len(test_dataset)
    img_shape = train_dataset[0][0].shape
    print(f"> Image shape: {list(img_shape)}")
    spatial_dim = img_shape[-1]

    assert cfg.nb_levels == len(cfg.scaling_rates), "Number of levels does not match number of scaling rates!"

    code_dims = [spatial_dim // math.prod(cfg.scaling_rates[:i+1]) for i in range(cfg.nb_levels)]
    print(f"> Latent code shapes:")
    for i, c in enumerate(code_dims):
        print(f"\tLevel {i+1}: [{c}, {c}]")

    
    # TODO: Allocating all space into memory at the start. Could run out of memory!
    print("> Allocating memory to latent datasets")
    latent_dataset = {
        'train':    [torch.zeros((train_dataset_len, c, c), dtype=torch.int64) for c in code_dims],
        'test':     [torch.zeros((test_dataset_len, c, c), dtype=torch.int64) for c in code_dims]
    }

    print("> Generating latent train dataset")
    pb = tqdm(train_loader, disable=args.no_tqdm)
    for i, (x, _) in enumerate(pb):
        with torch.no_grad(), torch.cuda.amp.autocast():
            x = x.to(device)
            idx = net(x)[-1][::-1]
        for ci in range(cfg.nb_levels):
            latent_dataset['train'][ci][i*cfg.batch_size:(i+1)*cfg.batch_size] = idx[ci]

    print("> Generating latent test dataset")
    pb = tqdm(test_loader, disable=args.no_tqdm)
    for i, (x, _) in enumerate(pb):
        with torch.no_grad(), torch.cuda.amp.autocast():
            x = x.to(device)
            idx = net(x)[-1][::-1]
        for ci in range(cfg.nb_levels):
            latent_dataset['test'][ci][i*cfg.batch_size:(i+1)*cfg.batch_size] = idx[ci]

    if not args.no_save:
        print("> Saving latent dataset to disk")
        torch.save(latent_dataset, dataset_path)
