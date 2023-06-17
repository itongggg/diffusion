# -*- coding:utf-8 -*-
"""
作者：itongggg
***
date：2023年06月05日
"""

from .types import *
import torch.nn as nn
from abc import abstractmethod


class BaseVAE(nn.Module):
    def __init__(self):
        super(BaseVAE, self).__init__()

    def encode(self, t: Tensor):
        raise NotImplementedError

    def decode(self, t: Tensor):
        raise NotImplementedError

    def sample(self, batch_size: int, device, **kwargs):
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor):
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs):
        pass
