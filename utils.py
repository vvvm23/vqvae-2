import torch
import wandb

def get_parameter_count(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def get_device(cpu):
    if cpu or not torch.cuda.is_available(): 
        return torch.device('cpu')
    return torch.device('cuda')

def init_wandb(
    entity: str,
    root_dir: str,
    project: str = 'vqvae-dev',
):
    return wandb.init(
        entity = entity,
        project = project,
        dir = root_dir,
        resume = 'auto',
    )
