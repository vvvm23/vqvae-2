import torch
import torch.nn.functional as F
import math

from vqvae import VQVAE
from pixelsnail import PixelSnail
from helper import get_device

class VQVAETrainer:
    def __init__(self, cfg, args):
        self.device = get_device(args.cpu)
        self.net = VQVAE(in_channels=cfg.in_channels, 
                    hidden_channels=cfg.hidden_channels, 
                    embed_dim=cfg.embed_dim, 
                    nb_entries=cfg.nb_entries, 
                    nb_levels=cfg.nb_levels, 
                    scaling_rates=cfg.scaling_rates).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self.opt.zero_grad()

        self.beta = cfg.beta
        self.scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)

        self.update_frequency = math.ceil(cfg.batch_size / cfg.mini_batch_size)
        self.train_steps = 0

    def _calculate_loss(self, x: torch.FloatTensor):
        x = x.to(self.device)
        y, d, _, _, _ = self.net(x)
        r_loss, l_loss = y.sub(x).pow(2).mean(), sum(d)
        loss = r_loss + self.beta*l_loss
        return loss, r_loss, l_loss, y

    # another function can then call step
    def train(self, x: torch.FloatTensor):
        self.net.train()
        # self.opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
            loss, r_loss, l_loss, _ = self._calculate_loss(x)
        self.scaler.scale(loss / self.update_frequency).backward()
        # self.scaler.step(self.opt)
        # self.scaler.update()

        self.train_steps += 1
        if self.train_steps % self.update_frequency == 0:
            self._update_parameters()

        return loss.item(), r_loss.item(), l_loss.item()

    """
        Use accumulated gradients to step `self.opt`, updating parameters.
    """
    def _update_parameters(self):
        self.scaler.step(self.opt)
        self.opt.zero_grad()
        self.scaler.update()

    @torch.no_grad()
    def eval(self, x: torch.FloatTensor):
        self.net.eval()
        # self.opt.zero_grad()
        loss, r_loss, l_loss, y = self._calculate_loss(x)
        return loss.item(), r_loss.item(), l_loss.item(), y

    def save_checkpoint(self, path):
        torch.save(self.net.state_dict(), path)

    def load_checkpoint(self, path):
        self.net.load_state_dict(torch.load(path))

class PixelTrainer:
    def __init__(self, cfg, args):
        self.device = get_device(args.cpu)
        # self.net = PixelSNAIL((32, 32), 256).to(self.device)

        lcfg = cfg.level[args.level]
        nb_cond = len(cfg.level) - args.level

        self.net = PixelSNAIL(
            shape =                 cfg.code_shape,
            nb_entries =            cfg.nb_entries,
            channel =               lcfg.channel,
            kernel_size =           lcfg.kernel_size,
            nb_block =              lcfg.nb_block,
            nb_res_block =          lcfg.nb_res_block,
            nb_res_channel =        lcfg.nb_res_channel,
            attention =             lcfg.attention,
            dropout =               lcfg.dropout,

            # TODO: Figure out correct nb_cond and scaling_rate based on args.level
            nb_cond =               nb_cond,
            scaling_rates =         [],
            nb_cond_res_block =     lcfg.nb_cond_res_block if nb_cond else 0,
            nb_cond_res_channel =   lcfg.nb_cond_res_channel if nb_cond else 0,

            nb_out_res_block =      lcfg.nb_out_res_block,
        )
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self.opt.zero_grad()
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)

        self.update_frequency = math.ceil(cfg.batch_size / cfg.mini_batch_size)
        self.train_steps = 0

    @torch.cuda.amp.autocast()
    def _calculate_loss(self, x: torch.LongTensor, *condition):
        x = x.to(self.device)
        y, _ = self.net(x, *condition)
        loss = F.cross_entropy(y, x)

        y_max = torch.argmax(y, dim=1)
        accuracy = (y_max == x) / torch.numel(x)

        return loss, accuracy

    def _update_parameters(self):
        self.scaler.step(self.opt)
        self.opt.zero_grad()
        self.scaler.update()
    
    def train_step(self, x: torch.LongTensor, *condition):
        self.net.train()
        loss, accuracy = self._calculate_loss(x, *condition)
        self.scaler.scale(loss / self.update_frequency).backward()

        self.train_steps += 1
        if self.train_steps % self.update_frequency == 0:
            self._update_parameters()

        return loss.item(), accuracy.item()

    @torch.no_grad()
    def eval_step(self, x: torch.LongTensor, *condition):
        self.net.eval()
        loss, accuracy = self._calculate_loss(x, *condition)
        return loss.item(), accuracy.item()

    def save_checkpoint(self, path):
        torch.save(self.net.state_dict(), path)

    def load_checkpoint(self, path):
        self.net.load_state_dict(torch.load(path))
