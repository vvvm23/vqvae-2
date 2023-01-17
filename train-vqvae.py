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
from pathlib import Path
from datetime import datetime 

from vqvae2 import VQVAE
from data import get_dataset
from utils import init_wandb

import wandb as wandb_module

def save_model(net, path):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        net = accelerator.unwrap_model(net)
        accelerator.save(net.state_dict(), path)

def setup_directory(base='exp'):
    root_dir = Path(base)
    root_dir.mkdir(exist_ok=True)

    save_id = 'vqvae_' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    exp_dir = (root_dir / save_id)
    exp_dir.mkdir(exist_ok=True)
    return exp_dir

@hydra.main(version_base=None, config_path='config', config_name="config")
def main(cfg: DictConfig):
    logger.info("Loaded Hydra config:")
    logger.info(OmegaConf.to_yaml(cfg))

    cfg.vqvae.training.batch_size //= accelerator.num_processes

    exp_dir = setup_directory()
    checkpoint_dir = exp_dir / 'checkpoints'
    recon_dir = exp_dir / 'recon'

    checkpoint_dir.mkdir(exist_ok=True)
    recon_dir.mkdir(exist_ok=True)

    if accelerator.is_main_process:
        wandb = init_wandb(cfg.wandb.entity, exp_dir, cfg.wandb.project)

    accelerator.wait_for_everyone()

    def loss_fn(net, batch, eval=False):
        x, *_ = batch
        recon, idx, diff = net(x)
        if eval: # TODO: don't overload inbuilt function
            x, recon = accelerator.gather_for_metrics((x, recon))
        mse_loss = F.mse_loss(recon, x)
        return mse_loss + diff * cfg.vqvae.training.beta, mse_loss, diff, recon, idx

    net = VQVAE(**cfg.vqvae.model, activation=torch.nn.ReLU)
    optim = torch.optim.AdamW(net.parameters(), lr=cfg.vqvae.training.lr)
    train_loader, test_loader = get_dataset(cfg)

    net, optim, train_loader, test_loader = accelerator.prepare(net, optim, train_loader, test_loader)
    codebook_history = []

    steps = 0
    max_steps = cfg.vqvae.training.max_steps
    while steps <= max_steps:
        it = train_loader
        if accelerator.is_local_main_process:
            it = tqdm(train_loader)

        total_loss, total_mse_loss, total_kl_loss = 0.0, 0.0, 0.0
        idx_total = torch.zeros(cfg.vqvae.model.codebook_size).cpu().long()
        net.train()
        for batch in it:
            optim.zero_grad()
            loss, mse_loss, kl_loss, _, idx = loss_fn(net, batch)
            accelerator.backward(loss)
            optim.step()

            total_loss += loss
            total_mse_loss += mse_loss
            total_kl_loss += kl_loss

            idx_total += torch.bincount(idx.long().flatten(), minlength=cfg.vqvae.model.codebook_size).detach().cpu()

            if steps > max_steps:
                save_model(net, checkpoint_dir / f'state_dict_final.pt')
                break

            steps += 1

            if steps % cfg.vqvae.training.save_frequency == 0:
                save_model(net, checkpoint_dir / f'state_dict_{steps:06}.pt')


        if steps <= max_steps:
            logging.info(f"[training {steps}/{max_steps}] loss: {total_loss/len(train_loader)}, mse_loss: {total_mse_loss/len(train_loader)}, kl_loss: {total_kl_loss/len(train_loader)}")
            if accelerator.is_main_process:
                wandb.log({
                    'train': {
                        'loss': total_loss/len(train_loader),
                        'mse_loss': total_mse_loss/len(train_loader),
                        'kl_loss': total_kl_loss/len(train_loader),
                    }
                }, commit=False)
                codebook_history.append(idx_total)

        total_loss, total_mse_loss, total_kl_loss = 0.0, 0.0, 0.0
        net.eval()
        with torch.no_grad():
            for batch in test_loader:
                loss, mse_loss, kl_loss, recon, idx = loss_fn(net, batch)

                # TODO: refactor metrics with helper class
                total_loss += loss
                total_mse_loss += mse_loss
                total_kl_loss += kl_loss

            if accelerator.is_local_main_process:
                save_image(
                    torch.concat([batch[0], recon], axis=0),
                    recon_dir / f'recon_{steps:06}.png', 
                    nrow=len(recon), normalize=True
                )
            accelerator.wait_for_everyone()
        
        logging.info(f"[evaluation {steps}/{max_steps}] (eval) loss: {total_loss/len(test_loader)}, mse_loss: {total_mse_loss/len(test_loader)}, kl_loss: {total_kl_loss/len(test_loader)}")
        if accelerator.is_main_process:
            wandb.log({
                'eval': {
                    'loss': total_loss/len(train_loader),
                    'mse_loss': total_mse_loss/len(train_loader),
                    'kl_loss': total_kl_loss/len(train_loader),
                }
            }, commit=True)
    if accelerator.is_main_process:
        wandb.log({'codebook_usage': wandb_module.plot.line_series(
            xs=list(range(len(codebook_history))),
            ys=torch.stack(codebook_history),
            keys=[str(i) for i in range(cfg.vqvae.model.codebook_size)],
            title="Codebook Usage",
            xname="Epochs"
        )})



if __name__ == '__main__':
    main(None)
