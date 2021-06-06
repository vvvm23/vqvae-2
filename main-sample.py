import torch
import torchvision
from torchvision.utils import save_image

from tqdm import tqdm
import argparse
import datetime
import time
from pathlib import Path
from math import sqrt

from datasets import get_dataset
from hps import HPS_VQVAE, HPS_PIXEL
from helper import get_device, get_parameter_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vqvae_path', type=str)
    parser.add_argument('pixelsnail_path', type=str, nargs='+')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--task', type=str, default='cifar10')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--no-tqdm', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--save-jpg', action='store_true')
    args = parser.parse_args()

    print(args)
