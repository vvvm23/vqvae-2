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

from vqvae2 import VQVAE, VQVAE2
from data import get_dataset
from utils import init_wandb, MetricGroup, setup_directory

import wandb as wandb_module


def save_model(net, path):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        net = accelerator.unwrap_model(net)
        accelerator.save(net.state_dict(), path)


# TODO: in general, all the logging needs a rework
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    logger.info("Loaded Hydra config:")
    logger.info(OmegaConf.to_yaml(cfg))

    assert (
        cfg.vqvae.training.batch_size % accelerator.num_processes == 0
    ), "Batch size must be divisible by number of Acccelerate processes"
    cfg.vqvae.training.batch_size //= accelerator.num_processes

    exp_dir = setup_directory()
    checkpoint_dir = exp_dir / "checkpoints"
    recon_dir = exp_dir / "recon"

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

    net = VQVAE2.build_from_config(cfg.vqvae.model)
    optim = torch.optim.AdamW(net.parameters(), lr=cfg.vqvae.training.lr)
    train_loader, test_loader = get_dataset(cfg)

    net, optim, train_loader, test_loader = accelerator.prepare(net, optim, train_loader, test_loader)

    columns = [f"codeword_{i:04}" for i in range(cfg.vqvae.model.codebook_size)]
    idx_table = [
        # wandb_module.Table(columns=[f"codeword_{i:04}" for i in range(cfg.vqvae.model.codebook_size)])
        []
        for _ in cfg.vqvae.model.resample_factors
    ]

    steps = 0
    max_steps = cfg.vqvae.training.max_steps
    while steps <= max_steps:
        wandb_log = {}
        it = train_loader
        if accelerator.is_local_main_process:
            it = tqdm(train_loader)

        metrics = MetricGroup("loss", "mse_loss", "kl_loss")
        total_idx = [torch.zeros(cfg.vqvae.model.codebook_size).cpu().long() for _ in cfg.vqvae.model.resample_factors]
        net.train()
        for batch in it:
            optim.zero_grad()
            loss, *m, _, idx = loss_fn(net, batch)
            accelerator.backward(loss)
            optim.step()

            for i in range(len(idx)):
                total_idx[i] += torch.bincount(idx[i].cpu().detach().flatten(), minlength=cfg.vqvae.model.codebook_size)

            metrics.log(loss, *m)

            if steps > max_steps:
                save_model(net, checkpoint_dir / f"state_dict_final.pt")
                break

            steps += 1

            if steps % cfg.vqvae.training.save_frequency == 0:
                save_model(net, checkpoint_dir / f"state_dict_{steps:06}.pt")

        if steps <= max_steps:
            metrics.print_summary(f"training {steps}/{max_steps}")
            if accelerator.is_main_process:
                wandb_log["train"] = {}
                for i in range(len(total_idx)):
                    # idx_table[i].add_data(*((total_idx[i] / (idx[i].numel() * len(train_loader))).tolist()))
                    idx_table[i].append((total_idx[i] / (idx[i].numel() * len(train_loader))))

                    wandb_log.update(
                        {
                            f"codebook_usage.{i}": wandb_module.plot.line_series(
                                xs=list(range(len(idx_table[i]))),
                                ys=torch.stack(idx_table[i], dim=-1).numpy(),
                                keys=columns,
                                xname="Epochs",
                            )
                        }
                    )
                    wandb_log["train"].update(
                        {f"unused_codewords_proportion.{i}": (total_idx[i] == 0).sum() / cfg.vqvae.model.codebook_size}
                    )

                wandb_log["train"].update(metrics.summarise())

        metrics = MetricGroup("loss", "mse_loss", "kl_loss")
        net.eval()
        with torch.no_grad():
            total_idx = [
                torch.zeros(cfg.vqvae.model.codebook_size).cpu().long() for _ in cfg.vqvae.model.resample_factors
            ]
            for batch in test_loader:
                loss, *m, recon, idx = loss_fn(net, batch)
                metrics.log(loss, *m)
                for i in range(len(idx)):
                    total_idx[i] += torch.bincount(idx[i].cpu().flatten(), minlength=cfg.vqvae.model.codebook_size)

        metrics.print_summary(f"evaluation {steps}/{max_steps}")

        if accelerator.is_main_process:
            wandb_log["eval"] = {}
            save_image(
                torch.concat([batch[0], recon], axis=0),
                recon_dir / f"recon_{steps:06}.png",
                nrow=len(recon),
                normalize=True,
            )
            wandb_log.update(
                {
                    "input": wandb_module.Image(batch[0], caption="Input Image"),
                    "recon": wandb_module.Image(recon, caption="Reconstruction"),
                }
            )
            for i in range(len(total_idx)):
                wandb_log["eval"].update(
                    {f"unused_codewords_proportion.{i}": (total_idx[i] == 0).sum() / cfg.vqvae.model.codebook_size}
                )
            wandb_log["eval"].update(metrics.summarise())
            wandb.log(wandb_log)
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main(None)
