import torch

from vqvae import VQVAE
from helper import get_device

class Trainer:
    def __init__(self, cfg, args):
        self.device = get_device(args.cpu)
        self.net = VQVAE(in_channels=cfg.in_channels, 
                    hidden_channels=cfg.hidden_channels, 
                    embed_dim=cfg.embed_dim, 
                    nb_entries=cfg.nb_entries, 
                    nb_levels=cfg.nb_levels, 
                    scaling_rates=cfg.scaling_rates).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self.beta = cfg.beta
        self.scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)

    def _calculate_loss(self, x: torch.FloatTensor):
        x = x.to(self.device)
        y, d, _, _, _ = self.net(x)
        r_loss, l_loss = y.sub(x).pow(2).mean(), sum(d)
        loss = r_loss + self.beta*l_loss
        return loss, r_loss, l_loss, y

    # TODO: maybe train without optim update, just accumulate grads?
    # another function can then call step
    def train(self, x: torch.FloatTensor):
        self.net.train()
        self.opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
            loss, r_loss, l_loss, _ = self._calculate_loss(x)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        return loss.item(), r_loss.item(), l_loss.item()

    @torch.no_grad()
    def eval(self, x: torch.FloatTensor):
        self.net.eval()
        self.opt.zero_grad()
        loss, r_loss, l_loss, y = self._calculate_loss(x)
        return loss.item(), r_loss.item(), l_loss.item(), y

    def save_checkpoint(self, path):
        torch.save(self.net.state_dict(), path)

    def load_checkpoint(self, path):
        self.net.load_state_dict(torch.load(path))
