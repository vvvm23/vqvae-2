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
