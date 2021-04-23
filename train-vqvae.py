import torch
import torchvision
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse
from math import sqrt

from vqvae import VQVAE
from hps import HPS
from helper import NoLabelImageFolder, get_device

# TODO: Function to generate train / test splits
def get_dataset(task: str, cfg):
    if task == 'ffhq1024':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        # dataset = torchvision.datasets.ImageFolder('data/ffhq1024', transform=transforms)
        dataset = NoLabelImageFolder('data/ffhq1024', transform=transforms)
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, num_workers=8, shuffle=True)
    elif task == 'cifar10':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = torchvision.datasets.CIFAR10('data', transform=transforms, download=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, num_workers=8, shuffle=True)
    elif task == 'stl10':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = torchvision.datasets.STL10('data', split='unlabeled', transform=transforms, download=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, num_workers=8, shuffle=True)
    elif task == 'mnist':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.MNIST('data', train=True, transform=transforms, download=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, num_workers=8, shuffle=True)
    elif task == 'kmnist':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.KMNIST('data', train=True, transform=transforms, download=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, num_workers=8, shuffle=True)
    else:
        print("> Unknown dataset. Terminating")
        exit()

    return dataset, loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--task', type=str, default='cifar10')
    args = parser.parse_args()
    cfg = HPS[args.task]

    print(f"Loading {cfg.display_name} dataset")
    dataset, loader = get_dataset(args.task, cfg)
    device = get_device(args.cpu)
    net = VQVAE(in_channels=cfg.in_channels, 
                hidden_channels=cfg.hidden_channels, 
                embed_dim=cfg.embed_dim, 
                nb_entries=cfg.nb_entries, 
                nb_levels=cfg.nb_levels, 
                scaling_rates=cfg.scaling_rates).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)

    for eid in range(cfg.max_epochs):
        epoch_loss, epoch_r_loss, epoch_l_loss = 0.0, 0.0, 0.0
        pb = tqdm(loader)
        net.train()
        for i, (x, _) in enumerate(pb):
            optim.zero_grad()
            x = x.to(device)
            y, d, _, _ = net(x)

            r_loss, l_loss = ((y-x)**2).mean(), sum(d)

            loss = r_loss + cfg.beta*l_loss
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            epoch_r_loss += r_loss.item()
            epoch_l_loss += l_loss.item()
            pb.set_description(f"training_loss: {epoch_loss / (i+1)} [r_loss: {epoch_r_loss/ (i+1)}, l_loss: {epoch_l_loss / (i+1)}]")

            if i % cfg.image_frequency == 0:
                save_image(y, f"samples/recon-{i}.png", nrow=int(sqrt(cfg.batch_size)), normalize=True, value_range=(-1,1))

        print(f"Loss: {epoch_loss / len(loader)}")
