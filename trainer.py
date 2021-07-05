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
                    scaling_rates=cfg.scaling_rates)
        if torch.cuda.device_count() > 1:
            self.net = torch.nn.DataParallel(self.net)
        self.net = self.net.to(self.device)

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
        with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
            loss, r_loss, l_loss, y = self._calculate_loss(x)
        self.scaler.scale(loss / self.update_frequency).backward()

        self.train_steps += 1
        if self.train_steps % self.update_frequency == 0:
            self._update_parameters()

        return loss.item(), r_loss.item(), l_loss.item(), y

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
    def __init__(self, cfg_pixel, cfg_vqvae, args):
        self.device = get_device(args.cpu)

        lcfg = cfg_pixel.level[args.level]
        nb_cond = len(cfg_pixel.level) - args.level - 1
        self.prior = PixelSnail(
            shape =                 cfg_pixel.code_shape,
            nb_class =              cfg_pixel.nb_entries,
            channel =               lcfg.channel,
            kernel_size =           lcfg.kernel_size,
            nb_pixel_block =        lcfg.nb_block,
            nb_res_block =          lcfg.nb_res_block,
            res_channel =           lcfg.nb_res_channel,
            dropout =               lcfg.dropout,

            nb_cond =               nb_cond,
            nb_cond_res_block =     lcfg.nb_cond_res_block if nb_cond else 0,
            nb_cond_in_res_block =  lcfg.nb_cond_in_res_block if nb_cond else 0,
            cond_embedding_dim =    cfg_vqvae.embed_dim,
            cond_res_channel =      lcfg.nb_cond_res_channel if nb_cond else 0,

            nb_out_res_block =      lcfg.nb_out_res_block,
            attention =             lcfg.attention,
        ).to(self.device)

        self.opt = torch.optim.Adam(self.prior.parameters(), lr=cfg_pixel.learning_rate)
        self.opt.zero_grad()

        self.vqvae = VQVAE(
            in_channels=cfg_vqvae.in_channels, 
            hidden_channels=cfg_vqvae.hidden_channels, 
            embed_dim=cfg_vqvae.embed_dim, 
            nb_entries=cfg_vqvae.nb_entries, 
            nb_levels=cfg_vqvae.nb_levels, 
            scaling_rates=cfg_vqvae.scaling_rates
        ).to(self.device)
        
        self.vqvae.load_state_dict(torch.load(args.vqvae_path))
        self.vqvae.eval()

        self.update_frequency = math.ceil(cfg_pixel.batch_size / cfg_pixel.mini_batch_size)
        self.train_steps = 0

        self.level = args.level

    # inplace
    @torch.no_grad()
    def _dequantize_condition(self, condition):
        for i, c in enumerate(condition):
            condition[i] = self.vqvae.codebooks[self.level+i+1].embed_code(condition[i]).permute(0, 3, 1, 2)

    def _calculate_loss(self, x: torch.LongTensor, condition):
        x = x.to(self.device)
        condition = [c.to(self.device) for c in condition]
        self._dequantize_condition(condition)

        y, _ = self.prior(x, cs=condition)

        # for some reason, setting reduction='none' THEN doing mean prevents inf loss during AMP
        loss = F.cross_entropy(y, x, reduction='none').mean()

        y_max = torch.argmax(y, dim=1)
        accuracy = (y_max == x).sum() / torch.numel(x)

        return loss, accuracy, y

    def _update_parameters(self):
        self.opt.step()
        self.opt.zero_grad()
        # self.scaler.step(self.opt)
        # self.opt.zero_grad()
        # self.scaler.update()
    
    def train(self, x: torch.LongTensor, condition):
        self.prior.train()
        loss, accuracy, y = self._calculate_loss(x, condition)
        # self.scaler.scale(loss / self.update_frequency).backward()
        (loss / self.update_frequency).backward()

        self.train_steps += 1
        if self.train_steps % self.update_frequency == 0:
            self._update_parameters()

        return loss.item(), accuracy.item(), y

    @torch.no_grad()
    def eval(self, x: torch.LongTensor, condition):
        self.prior.eval()
        loss, accuracy, y = self._calculate_loss(x, condition)
        return loss.item(), accuracy.item(), y

    def save_checkpoint(self, path):
        torch.save(self.prior.state_dict(), path)

    def load_checkpoint(self, path):
        self.prior.load_state_dict(torch.load(path))
