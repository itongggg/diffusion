# -*- coding:utf-8 -*-
"""
作者：itongggg
***
date：2023年06月07日
"""
import torch
import torch.nn as nn

class A(nn.Module):
    def __init__(self):
        super(A, self).__init__()
        c = torch.ones(10)
        self.register_buffer('d', c)

    def display(self):
        print(self.d)


a = A()
a.display()
