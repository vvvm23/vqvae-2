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
            torchvision.transforms.ToTensor()
        ])
        # dataset = torchvision.datasets.ImageFolder('data/ffhq1024', transform=transforms)
        dataset = NoLabelImageFolder('data/ffhq1024', transform=transforms)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    elif task == 'cifar10':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        dataset = torchvision.datasets.CIFAR10('data', transform=transforms, download=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    elif task == 'stl10':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        dataset = torchvision.datasets.STL10('data', split='unlabeled', transform=transforms, download=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True)

    return dataset, loader

if __name__ == '__main__':
    print("Loading FFHQ1024 dataset")
    # dataset, loader = get_dataset('ffhq1024', 8)
    dataset, loader = get_dataset('stl10', 32)
    device = torch.device('cuda')
    # net = VQVAE().to(device)
    net = VQVAE(hidden_channels=64, embed_dim=32, nb_entries=256, nb_levels=2, scaling_rates=[4,2]).to(device)
    crit = torch.nn.MSELoss()
    optim = torch.optim.Adam(net.parameters())

    for eid in range(100):
        epoch_loss = 0.
        pb = tqdm(loader)
        for i, (x, _) in enumerate(pb):
            optim.zero_grad()
            x = x.to(device)
            y, d, _, _ = net(x)

            loss = crit(y, x) + 0.25 * sum(d)
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            pb.set_description(f"training_loss: {epoch_loss / (i+1)}")

            if i % 100 == 0:
                save_image(y, f"samples/recon-{i}.png", nrow=8)

        print(f"Loss: {epoch_loss / len(loader)}")
