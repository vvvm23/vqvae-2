import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from vqvae2 import VQVAE
from data import get_dataset

from accelerate import Accelerator
accelerator = Accelerator()

def main(args):
    steps = 300_000
    batch_size = 128
    lr = 1e-4
    beta = 0.25
    nb_workers = 4

    def loss_fn(net, batch):
        x, *_ = batch
        recon, _, diff = net(x)
        mse_loss = F.mse_loss(recon, x)
        return mse_loss + diff*beta, mse_loss, diff

    net = VQVAE(
        in_dim=3, 
        hidden_dim=256, 
        codebook_dim=256,
        codebook_size=512,
        residual_dim=256
    )
    optim = torch.optim.AdamW(net.parameters(), lr=lr)
    train_loader, test_loader = get_dataset('cifar10', batch_size, nb_workers)

    net, optim, train_loader = accelerator.prepare(net, optim, train_loader)

    for batch in train_loader:
        optim.zero_grad()
        loss, mse_loss, kl_loss = loss_fn(net, batch)
        accelerator.backward(loss)
        optim.step()

        accelerator.print(loss, mse_loss, kl_loss)


if __name__ == '__main__':
    main(None)