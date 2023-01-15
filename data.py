import torch
import torchvision
import torchvision.transforms as T

import logging
from accelerate.logging import get_logger
logger = get_logger(__file__)

def build_transforms(cfg):
    train_transforms = [
        T.Resize((cfg.data.height, cfg.data.width)),
        T.ToTensor()
    ]
    test_transforms = [
        T.Resize((cfg.data.height, cfg.data.width)),
        T.ToTensor()
    ]

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
    if cfg.data.name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10('data', train=True, transform=train_transforms, download=True)
        test_dataset = torchvision.datasets.CIFAR10('data', train=False, transform=test_transforms, download=True)
    elif cfg.data.name == 'mnist':
        train_dataset = torchvision.datasets.MNIST('data', train=True, transform=train_transforms, download=True)
        test_dataset = torchvision.datasets.MNIST('data', train=False, transform=test_transforms, download=True)
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
