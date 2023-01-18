import torch
import wandb

import logging
from accelerate.logging import get_logger
from pathlib import Path
from datetime import datetime 

logger = get_logger(__file__)

def get_parameter_count(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def get_device(cpu):
    if cpu or not torch.cuda.is_available(): 
        return torch.device('cpu')
    return torch.device('cuda')

def init_wandb(cfg, root_dir):
    if 'wandb' not in cfg:
        return wandb.init(mode='disabled')
    return wandb.init(
        entity = cfg.wandb.entity,
        project = cfg.wandb.project,
        dir = root_dir,
        resume = 'auto',
    )

def setup_directory(base='exp'):
    root_dir = Path(base)
    root_dir.mkdir(exist_ok=True)

    save_id = 'vqvae_' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    exp_dir = (root_dir / save_id)
    exp_dir.mkdir(exist_ok=True)
    return exp_dir

class Metric:
    def __init__(self, name):
        self.name = name
        self.reset()

    def log(self, value):
        self.total += value
        self.steps += 1

    def reset(self):
        self.total = 0.0
        self.steps = 0

    def summarise(self):
        avg = self.total / self.steps
        return avg

class MetricGroup:
    def __init__(self, *names):
        self.names = names
        self.metrics = {n: Metric(n) for n in names}

    def log(self, *values):
        for v, m in zip(values, self.metrics.values()):
            m.log(v)

    def reset(self):
        for m in self.metrics.values():
            m.reset()

    def summarise(self):
        return {n: m.summarise() for n, m in self.metrics.items()}

    def print_summary(self, header):
        summary = self.summarise()

        msg = (
            f"[{header}] " +
            ', '.join(f"{n}: {v}" for n, v in summary.items())
        )
        logging.info(msg)