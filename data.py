import torch
import torchvision
import torchvision.transforms as T

from PIL import Image
from typing import Optional, List, Union
from pathlib import Path

import logging
from accelerate.logging import get_logger

logger = get_logger(__file__)


class FFHQDataset(torch.utils.data.Dataset):
    TRAIN_SPLIT_SIZE = 65_000

    def __init__(self, root: Union[str, Path] = "data/ffhq1024", train: bool = True, transform: T = None):
        if isinstance(root, str):
            root = Path(root)

        assert root.is_dir()
        paths = list(root.glob("*.png"))

        if train:
            self.paths = paths[: FFHQDataset.TRAIN_SPLIT_SIZE]
        else:
            self.paths = paths[FFHQDataset.TRAIN_SPLIT_SIZE :]

        self.transform = transform if transform else T.Compose()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        return self.transform(img)


def build_transforms(cfg):
    train_transforms = [T.Resize((cfg.data.height, cfg.data.width)), T.ToTensor()]
    test_transforms = [T.Resize((cfg.data.height, cfg.data.width)), T.ToTensor()]

    if cfg.data.preprocess.vflip:
        train_transforms.append(T.RandomVerticalFlip())
    if cfg.data.preprocess.hflip:
        train_transforms.append(T.RandomHorizontalFlip())
    if cfg.data.preprocess.normalise:
        assert len(cfg.data.preprocess.normalise.mean) == cfg.data.channels
        assert len(cfg.data.preprocess.normalise.std) == cfg.data.channels
        train_transforms.append(T.Normalize(**cfg.data.preprocess.normalise))
        test_transforms.append(T.Normalize(**cfg.data.preprocess.normalise))

    train_transforms, test_transforms = T.Compose(train_transforms), T.Compose(test_transforms)

    return train_transforms, test_transforms


def get_dataset(cfg):
    train_transforms, test_transforms = build_transforms(cfg)
    if cfg.data.name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10("data", train=True, transform=train_transforms, download=True)
        test_dataset = torchvision.datasets.CIFAR10("data", train=False, transform=test_transforms, download=True)
    elif cfg.data.name == "mnist":
        train_dataset = torchvision.datasets.MNIST("data", train=True, transform=train_transforms, download=True)
        test_dataset = torchvision.datasets.MNIST("data", train=False, transform=test_transforms, download=True)
    elif cfg.data.name in ["ffhq1024", "ffhq256", "ffhq128"]:
        train_dataset = FFHQDataset(f"data/{cfg.data.name}", train=True, transform=train_transforms)
        test_dataset = FFHQDataset(f"data/{cfg.data.name}", train=False, transform=test_transforms)
    else:
        logging.error(f"Unknown dataset {cfg.data.name}. Terminating")
        exit()

    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")

    batch_size = cfg.vqvae.training.batch_size
    workers = cfg.data.num_workers

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=workers, shuffle=False)

    return train_loader, test_loader
