# -*- coding:utf-8 -*-
"""
作者：itongggg
***
date：2023年06月05日
"""
import torch
from .types import *
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseVAE


class VanillaVAE(BaseVAE):
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List, **kwargs):
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim


