import torch
import torchvision

class HelperModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.build(*args, **kwargs)

    def build(self, *args, **kwargs):
        raise NotImplementedError

class NoLabelImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, idx):
        return super().__getitem__(idx)[0]

def get_parameter_count(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
