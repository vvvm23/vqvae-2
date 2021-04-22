import torch
import torchvision
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from tqdm import tqdm

from vqvae import VQVAE
from hps import HPS
from helper import NoLabelImageFolder

# TODO: Function to generate train / test splits
def get_dataset(task: str, batch_size: int):
    if task == 'ffhq1024':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        # dataset = torchvision.datasets.ImageFolder('data/ffhq1024', transform=transforms)
        dataset = NoLabelImageFolder('data/ffhq1024', transform=transforms)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    elif task == 'cifar10':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = torchvision.datasets.CIFAR10('data', transform=transforms, download=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    elif task == 'stl10':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = torchvision.datasets.STL10('data', split='unlabeled', transform=transforms, download=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    elif task == 'mnist':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.MNIST('data', train=True, transform=transforms, download=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    elif task == 'kmnist':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.KMNIST('data', train=True, transform=transforms, download=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    else:
        print("> Unknown dataset. Terminating")
        exit()

    return dataset, loader

if __name__ == '__main__':
    print("Loading FFHQ1024 dataset")
    # dataset, loader = get_dataset('ffhq1024', 8)
    dataset, loader = get_dataset('ffhq1024', 8)
    device = torch.device('cuda')
    # net = VQVAE().to(device)
    net = VQVAE(in_channels=3, hidden_channels=256, embed_dim=64, nb_entries=256, nb_levels=3, scaling_rates=[8, 2, 2]).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=2.5e-4)

    for eid in range(100):
        epoch_loss, epoch_r_loss, epoch_l_loss = 0.0, 0.0, 0.0
        pb = tqdm(loader)
        net.train()
        # for i, (x, _) in enumerate(pb):
        for i, x in enumerate(pb):
            optim.zero_grad()
            x = x.to(device)
            y, d, _, _ = net(x)

            r_loss, l_loss = ((y-x)**2).mean(), sum(d)

            loss = r_loss + 100.0*l_loss
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            epoch_r_loss += r_loss.item()
            epoch_l_loss += l_loss.item()
            pb.set_description(f"training_loss: {epoch_loss / (i+1)} [r_loss: {epoch_r_loss/ (i+1)}, l_loss: {epoch_l_loss / (i+1)}]")

            if i % 100 == 0:
                save_image(y, f"samples/recon-{i}.png", nrow=8, normalize=True, value_range=(-1,1))

        print(f"Loss: {epoch_loss / len(loader)}")
