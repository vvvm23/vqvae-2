from accelerate import Accelerator
from accelerate.utils import set_seed

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

def main(args):
    batch_size = 128 // accelerator.num_processes
    lr = 1e-4
    beta = 0.25
    nb_workers = 4

    max_steps = 300_000
    save_frequency = 10_000

    root_dir = setup_directory()

    def loss_fn(net, batch, eval=False):
        x, *_ = batch
        recon, _, diff = net(x)
        if eval:
            x, recon = accelerator.gather_for_metrics((x, recon))
        mse_loss = F.mse_loss(recon, x)
        return mse_loss + diff*beta, mse_loss, diff

    net = VQVAE(
        in_dim=3, 
        hidden_dim=128, 
        codebook_dim=64,
        codebook_size=1024,
        residual_dim=256
    )
    optim = torch.optim.AdamW(net.parameters(), lr=lr)
    train_loader, test_loader = get_dataset('cifar10', batch_size, nb_workers)

    net, optim, train_loader, test_loader = accelerator.prepare(net, optim, train_loader, test_loader)

    steps = 0
    while steps <= max_steps:
        it = train_loader
        if accelerator.is_local_main_process:
            it = tqdm(train_loader)

        total_loss, total_mse_loss, total_kl_loss = 0.0, 0.0, 0.0
        for batch in it:
            optim.zero_grad()
            loss, mse_loss, kl_loss = loss_fn(net, batch)
            accelerator.backward(loss)
            optim.step()

            total_loss += loss
            total_mse_loss += mse_loss
            total_kl_loss += kl_loss

            if steps > max_steps:
                save_model(net, root_dir / f'state_dict_final.pt')
                break

            steps += 1

            if steps % save_frequency == 0:
                save_model(net, root_dir / f'state_dict_{steps:06}.pt')

        if steps <= max_steps:
            accelerator.print(f"[training {steps}/{max_steps}] loss: {total_loss/len(train_loader)}, mse_loss: {total_mse_loss/len(train_loader)}, kl_loss: {total_kl_loss/len(train_loader)}")

        total_loss, total_mse_loss, total_kl_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in test_loader:
                loss, mse_loss, kl_loss = loss_fn(net, batch)

                total_loss += loss
                total_mse_loss += mse_loss
                total_kl_loss += kl_loss
        
        accelerator.print(f"[evaluation {steps}/{max_steps}] (eval) loss: {total_loss/len(test_loader)}, mse_loss: {total_mse_loss/len(test_loader)}, kl_loss: {total_kl_loss/len(test_loader)}")



if __name__ == '__main__':
    main(None)