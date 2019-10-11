__author__ = "Aditya Singh"
__version__ = "0.1"

import torch
import torch.nn as nn


class MirrorActivationUnit(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)
