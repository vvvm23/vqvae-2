import torch

from vqvae import VQVAE
from helper import get_device
"""
    Might abstract away some training routines here.
    ..or just delete this file
"""
class Trainer:
    def __init__(self, cfg, cpu=False):
        self.device = get_device(cpu)
        self.net = VQVAE(in_channels=cfg.in_channels, 
                    hidden_channels=cfg.hidden_channels, 
                    embed_dim=cfg.embed_dim, 
                    nb_entries=cfg.nb_entries, 
                    nb_levels=cfg.nb_levels, 
                    scaling_rates=cfg.scaling_rates).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self.beta = cfg.beta

    def _calculate_loss(self, x: torch.FloatTensor):
        x = x.to(self.device)
        y, d, _, _ = self.net(x)
        r_loss, l_loss = y.sub(x).pow(2).mean(), sum(d)
        loss = r_loss + self.beta*l_loss
        return loss, r_loss, l_loss, y

    # TODO: maybe train without optim update, just accumulate grads?
    # another function can then call step
    def train(self, x: torch.FloatTensor):
        self.net.train()
        self.opt.zero_grad()
        loss, r_loss, l_loss, _ = self._calculate_loss(x)
        loss.backward()
        self.opt.step()
        return loss.item(), r_loss.item(), l_loss.item()

    @torch.no_grad()
    def eval(self, x: torch.FloatTensor):
        self.net.eval()
        self.opt.zero_grad()
        loss, r_loss, l_loss, y = self._calculate_loss(x)
        return loss.item(), r_loss.item(), l_loss.item(), y

    # TODO: Checkpointing routines
    def save_checkpoint(self):
        pass
    def load_checkpoint(self):
        pass
