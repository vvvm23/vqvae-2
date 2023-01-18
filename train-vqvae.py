import hydra
from omegaconf import DictConfig, OmegaConf

import logging
from accelerate.logging import get_logger
from accelerate import Accelerator
from accelerate.utils import set_seed

logger = get_logger(__file__)

set_seed(0xAAAA)
accelerator = Accelerator()

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from tqdm import tqdm

from vqvae2 import VQVAE
from data import get_dataset
from utils import init_wandb, MetricGroup, setup_directory

import wandb as wandb_module

def save_model(net, path):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        net = accelerator.unwrap_model(net)
        accelerator.save(net.state_dict(), path)

@hydra.main(version_base=None, config_path='config', config_name="config")
def main(cfg: DictConfig):
    logger.info("Loaded Hydra config:")
    logger.info(OmegaConf.to_yaml(cfg))

    assert cfg.vqvae.training.batch_size % accelerator.num_processes == 0, "Batch size must be divisible by number of Acccelerate processes"
    cfg.vqvae.training.batch_size //= accelerator.num_processes

    exp_dir = setup_directory()
    checkpoint_dir = exp_dir / 'checkpoints'
    recon_dir = exp_dir / 'recon'

    checkpoint_dir.mkdir(exist_ok=True)
    recon_dir.mkdir(exist_ok=True)

    if accelerator.is_main_process:
        wandb = init_wandb(cfg, exp_dir)
    accelerator.wait_for_everyone()

    def loss_fn(net, batch, eval=False):
        x, *_ = batch
        recon, idx, diff = net(x)
        if eval:
            x, recon = accelerator.gather_for_metrics((x, recon))
        mse_loss = F.mse_loss(recon, x)
        return mse_loss + diff * cfg.vqvae.training.beta, mse_loss, diff, recon, idx

    net = VQVAE(**cfg.vqvae.model, activation=torch.nn.ReLU)
    optim = torch.optim.AdamW(net.parameters(), lr=cfg.vqvae.training.lr)
    train_loader, test_loader = get_dataset(cfg)

    net, optim, train_loader, test_loader = accelerator.prepare(net, optim, train_loader, test_loader)
    idx_history = []

    steps = 0
    max_steps = cfg.vqvae.training.max_steps
    while steps <= max_steps:
        it = train_loader
        if accelerator.is_local_main_process:
            it = tqdm(train_loader)

        metrics = MetricGroup('loss', 'mse_loss', 'kl_loss')
        total_idx = torch.zeros(cfg.vqvae.model.codebook_size).cpu().long()
        net.train()
        for batch in it:
            optim.zero_grad()
            loss, *m, _, idx = loss_fn(net, batch)
            accelerator.backward(loss)
            optim.step()

            total_idx += torch.bincount(idx.cpu().detach().flatten(), minlength=cfg.vqvae.model.codebook_size)

            metrics.log(loss, *m)

            if steps > max_steps:
                save_model(net, checkpoint_dir / f'state_dict_final.pt')
                break

            steps += 1

            if steps % cfg.vqvae.training.save_frequency == 0:
                save_model(net, checkpoint_dir / f'state_dict_{steps:06}.pt')



        if steps <= max_steps:
            metrics.print_summary(f"training {steps}/{max_steps}")
            if accelerator.is_main_process:
                wandb.log({'train': metrics.summarise()}, commit=False)
                wandb.log({'train': {'unused_codewords_proportion': 
                    (total_idx == 0).sum() / cfg.vqvae.model.codebook_size
                }}, commit=False)
                idx_history.append(total_idx / (idx.numel() * len(train_loader)))

        metrics = MetricGroup('loss', 'mse_loss', 'kl_loss')
        net.eval()
        with torch.no_grad():
            for batch in test_loader:
                loss, *m, recon, _ = loss_fn(net, batch)
                metrics.log(loss, *m)

            if accelerator.is_local_main_process:
                save_image(
                    torch.concat([batch[0], recon], axis=0),
                    recon_dir / f'recon_{steps:06}.png', 
                    nrow=len(recon), normalize=True
                )
            accelerator.wait_for_everyone()
        
        metrics.print_summary(f"evaluation {steps}/{max_steps}")
        if accelerator.is_main_process:
            wandb.log({'eval': metrics.summarise()}, commit=True)

    if accelerator.is_main_process:
        wandb.log({'codebook_usage':
            wandb_module.plot.line_series(
                xs=list(range(len(idx_history))),
                ys=torch.stack(idx_history, dim=-1).numpy(),
                keys=[f'codeword_{i:04}' for i in range(cfg.vqvae.model.codebook_size)],
                xname="Epochs"
            )
        })


if __name__ == '__main__':
    main(None)
