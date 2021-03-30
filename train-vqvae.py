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
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    return dataset, loader

if __name__ == '__main__':
    print("Loading FFHQ1024 dataset")
    dataset, loader = get_dataset('ffhq1024', 8)
    device = torch.device('cuda')
    net = VQVAE().to(device)
    crit = torch.nn.MSELoss()
    optim = torch.optim.Adam(net.parameters())

    for eid in range(100):
        epoch_loss = 0.
        for x in tqdm(loader):
            optim.zero_grad()
            x = x.to(device)
            y, d, _, _ = net(x)

            loss = crit(y, x) + 0.25 * sum(d)
            loss.backward()
            optim.step()

            epoch_loss += loss.item()

        print(f"Loss: {epoch_loss / len(loader)}")
