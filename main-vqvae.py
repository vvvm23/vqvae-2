import torch
import torchvision
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse
import datetime
import time
from pathlib import Path
from math import sqrt

from trainer import Trainer
from datasets import get_dataset
from hps import HPS
from helper import get_device, get_parameter_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--task', type=str, default='cifar10')
    parser.add_argument('--load-path', type=str, default=None)
    parser.add_argument('--no-tqdm', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    cfg = HPS[args.task]

    save_id = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    print(f"> Initialising VQ-VAE-2 model")
    trainer = Trainer(cfg, args.cpu)
    print(f"> Number of parameters: {get_parameter_count(trainer.net)}")

    if args.load_path:
        print(f"> Loading model parameters from checkpoint")
        trainer.load_checkpoint(args.load_path)

    if args.evaluate:
        # TODO: might leak train data into test given a random split
        # TODO: worth to split ffhq1024 beforehand
        # TODO: apparently first 60_000 are training, remaining are test
        print(f"> Loading {cfg.display_name} dataset")
        _, test_loader = get_dataset(args.task, cfg, shuffle_test=True)
        print(f"> Generating evaluation batch of reconstructions")
        file_name = f"./recon-{save_id}-eval.png"
        for x, _ in test_loader:
            *_, y = trainer.eval(x)
            save_image(y, file_name, nrow=int(sqrt(cfg.batch_size)), normalize=True, value_range=(-1,1))
            break
        print(f"> Saved to {file_name}")
        exit()
        
    if not args.no_save:
        runs_dir = Path(f"runs")
        root_dir = runs_dir / f"{args.task}-{save_id}"
        chk_dir = root_dir / "checkpoints"
        img_dir = root_dir / "images"
        log_dir = root_dir / "logs"

        runs_dir.mkdir(exist_ok=True)
        root_dir.mkdir(exist_ok=True)
        chk_dir.mkdir(exist_ok=True)
        img_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)

    print(f"> Loading {cfg.display_name} dataset")
    train_loader, test_loader = get_dataset(args.task, cfg)

    for eid in range(cfg.max_epochs):
        print(f"> Epoch {eid+1}/{cfg.max_epochs}:")
        epoch_loss, epoch_r_loss, epoch_l_loss = 0.0, 0.0, 0.0
        epoch_start_time = time.time()
        pb = tqdm(train_loader, disable=args.no_tqdm)
        for i, (x, _) in enumerate(pb):
            loss, r_loss, l_loss = trainer.train(x)
            epoch_loss += loss
            epoch_r_loss += r_loss
            epoch_l_loss += l_loss
            pb.set_description(f"training_loss: {epoch_loss / (i+1)} [r_loss: {epoch_r_loss/ (i+1)}, l_loss: {epoch_l_loss / (i+1)}]")
        print(f"> Training loss: {epoch_loss / len(train_loader)} [r_loss: {epoch_r_loss / len(train_loader)}, l_loss: {epoch_l_loss / len(train_loader)}]")
        
        epoch_loss, epoch_r_loss, epoch_l_loss = 0.0, 0.0, 0.0
        pb = tqdm(test_loader, disable=args.no_tqdm)
        for i, (x, _) in enumerate(pb):
            loss, r_loss, l_loss, y = trainer.eval(x)
            epoch_loss += loss
            epoch_r_loss += r_loss
            epoch_l_loss += l_loss
            pb.set_description(f"evaluation: {epoch_loss / (i+1)} [r_loss: {epoch_r_loss/ (i+1)}, l_loss: {epoch_l_loss / (i+1)}]")
            if i == 0 and not args.no_save and eid % cfg.image_frequency == 0:
                save_image(y, img_dir / f"recon-{str(eid).zfill(4)}.png", nrow=int(sqrt(cfg.batch_size)), normalize=True, value_range=(-1,1))

        if eid % cfg.checkpoint_frequency == 0 and not args.no_save:
            trainer.save_checkpoint(chk_dir / f"{args.task}-state-dict-{str(eid).zfill(4)}.pt")

        print(f"> Evaluation loss: {epoch_loss / len(test_loader)} [r_loss: {epoch_r_loss / len(test_loader)}, l_loss: {epoch_l_loss / len(test_loader)}]")
        print(f"> Epoch time taken: {time.time() - epoch_start_time:.2f} seconds.")
        print()
