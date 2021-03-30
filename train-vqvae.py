import torch
import torchvision
from torchvision.utils import save_image

import matplotlib.pyplot as plt

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
    dataset, loader = get_dataset('ffhq1024', 1)
    device = torch.device('cuda')
    net = VQVAE(nb_levels=3, scaling_rates=[2]*3).to(device)

    image = dataset.__getitem__(123).to(device).unsqueeze(0)
    y = net(image)[-1][-1]

    fig, axs = plt.subplots(2)
    axs[0].imshow(image.squeeze().detach().cpu().permute(1, 2, 0))
    axs[1].imshow(y.squeeze().detach().cpu().permute(1, 2, 0))
    print(y)
    plt.show()
