import torch
import torch.nn.functional as F

from torchvision.utils import save_image

import argparse
import datetime
import time
from tqdm import tqdm
from pathlib import Path

from trainer import PixelTrainer
from hps import HPS
from helper import get_device, get_parameter_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--task', type=str, default='cifar10')
    parser.add_argument('--load-path', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--no-tqdm', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--save-jpg', action='store_true')
    parser.add_argument('--level', action='store_true')
    args = parser.parse_args()

    # TODO: retrieve based on task and prior level
    cfg = HPS[args.task]

    save_id = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    trainer = PixelTrainer(cfg, args)

    if args.load_path:
        print(f"> Loading model parameters from checkpoint")
        trainer.load_checkpoint(args.load_path)

    if args.batch_size:
        cfg.batch_size = args.batch_size

    if not args.no_save:
        runs_dir = Path(f"runs")
        root_dir = runs_dir / f"pixelsnail-{args.level}-{args.task}-{save_id}"
        chk_dir = root_dir / "checkpoints"
        img_dir = root_dir / "images"
        log_dir = root_dir / "logs"

        runs_dir.mkdir(exist_ok=True)
        root_dir.mkdir(exist_ok=True)
        chk_dir.mkdir(exist_ok=True)
        img_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)

    # TODO: get latent dataset
    # TODO: train-eval loop over tandem latent dataset (level0, level1, ..., levelN)
