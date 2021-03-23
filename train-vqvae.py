import torch
import torchvision

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
    print(dataset.__getitem__(0))
