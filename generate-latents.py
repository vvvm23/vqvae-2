import hydra
from omegaconf import DictConfig, OmegaConf

import logging
from accelerate.logging import get_logger
from accelerate.utils import set_seed

logger = get_logger(__file__)

set_seed(0xAAAA)

import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from pathlib import Path

from vqvae2 import VQVAE, VQVAE2
from data import get_dataset

import wandb as wandb_module

import argparse

# TODO: with class condtioned experiments, need to save class
def encode_loop(loader, net, cfg, out_dir, device):
    out_dir.mkdir()

    count = 0
    for batch in tqdm(loader):
        if isinstance(batch, (list, tuple)):
            batch, *_ = batch
        _, idx, _ = net(batch.to(device))

        for i, id in enumerate(idx):
            for bi, b in enumerate(id):
                np.save(out_dir / f"{count+bi:05}.{i}.npy", b.cpu().numpy().astype(np.uint16))
        count += id.shape[0]


@torch.no_grad()
@torch.inference_mode()
def main(cfg: DictConfig, args):
    logger.info("Loaded Hydra config:")
    logger.info(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Initialising VQVAE model.")
    net = VQVAE2.build_from_config(
        cfg.vqvae.model, codebook_gumbel_temperature=0.0, codebook_init_type="kaiming_uniform", codebook_cosine=False
    )
    net.load_state_dict(torch.load(args.model_path))
    net.eval()
    net = net.to(device)

    out_dir = args.out_dir
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)
    out_dir.mkdir()

    logging.info("Loading dataset.")
    train_loader, test_loader = get_dataset(cfg)

    logging.info("Encoding train set.")
    encode_loop(train_loader, net, cfg, out_dir / "train", device)

    logging.info("Encoding test set.")
    encode_loop(test_loader, net, cfg, out_dir / "eval", device)


# TODO: hydra is a bit tricky with custom args. need to find a way to integrate this better as I can't override now
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    hydra.initialize(version_base=None, config_path="config")
    cfg = hydra.compose(config_name="config", overrides=[f"data={args.config_name}", f"vqvae={args.config_name}"])

    main(cfg, args)
