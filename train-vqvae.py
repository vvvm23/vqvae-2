import torch.nn.functional as F
from torchvision.utils import save_image

from vqvae2 import VQVAE
from data import get_dataset

from ptpt.trainer import TrainerConfig, Trainer

def main(args):
    batch_size = 16
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

    _, (train_dataset, test_dataset) = get_dataset('cifar10', batch_size, nb_workers, return_dataset=True)

    cfg = TrainerConfig(
        exp_name = 'vqvae-dev',
        batch_size=batch_size,
        learning_rate=lr,
        nb_workers=nb_workers,
        save_outputs=False,
        metric_names=['mse_loss', 'kl_loss'],
        use_amp=False,
    )

    trainer = Trainer(
        net=net,
        loss_fn=loss_fn,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        cfg=cfg
    )

    trainer.train()

if __name__ == '__main__':
    main(None)