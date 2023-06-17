# -*- coding:utf-8 -*-
"""
作者：itongggg
***
date：2023年05月30日
"""
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List


class Swish(nn.Module):
    """
    ### swish activation function
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    ### Embedding for time
    """
    def __init__(self, n_channels: int):
        """
        :param n_channels: the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # first linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device)*-emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb


class ResidualBlock(nn.Module):
    """
    ### Residual Block

    A residual block has two convolution layers with GN each resolution is processed with two residual blocks
    """
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32, dropout: float = 0.1):
        """
        :param in_channels: the number of input channels
        :param out_channels: the number of output channels
        :param time_channels: the number channels in time step embedddings
        :param n_groups: the number of groups for GN
        :param dropout: dropout rate
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # if in_channels != out_channels we have to project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """

        :param x: [batch_size, in_channels, height, width]
        :param t: [batch_size, time_channels]
        """
        h = self.conv1(self.act1(self.norm1(x)))
        # add time embeddings
        h += self.time_emb(self.time_act(t))[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(x)))

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    Attention block
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """

        :param n_channels: channels of input
        :param n_heads: number of heads in multi-head attention
        :param d_k: number of dimension in each head
        :param n_groups: number of groups for GN
        """
        super().__init__()

        if d_k is None:
            d_k = n_channels

        self.norm = nn.GroupNorm(n_groups, n_channels)
        # projection for q, k ,v
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** (-0.5)
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """

        :param x: [batch_size, in_channels, height, width]
        :param t: [batch_size, time_channels]
        t is not used but kept for the attention layer function signature to match with ResidualBlock
        """

        _ = t
        batch_size, n_channels, height, width = x.shape
        # change x to [batch_size, seq, n_channels]
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get q k v (concatenated) shape it to [batch_size, seq, n_heads, 3 * d_k]
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # [batch_size, seq, n_heads, d_k]
        # Calculate scaled dot-product q*k^T
        attn = torch.einsum('bihd, bjhd->bijh', q, k) * self.scale
        # softmax along the sequence dimension
        attn = attn.softmax(dim=2)
        # Multiply by values
        res = torch.einsum('bijh, bjhd->bihd', attn, v)
        # reshape [batch_size, seq, n_heads*d_k]
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)

        res += x
        # change shape to [batch_size, in_channels, height, width]
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        return res


class DownBlock(nn.Module):
    """
    This combines ResidualBlock and AttentionBlock These are used in the first half of U-Net at each resolution
    """
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """
     This combines ResidualBlock and AttentionBlock These are used in the second half of U-Net at each resolution
    """
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        # the input hasi in_channel + out_channel because we concatenate the output of the same resolution
        # from the first half of Unet
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """
    It combines a ResidualBlock, AttentionBlock followed by another ResidualBlock
    this block is applied at the lowest resolution of the U-Net
    """
    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    """
    scale up the feature map by two times
    """
    def __init__(self, n_channels: int):
        super().__init__()
        # kernel size = 4
        # stride = 2
        # padding = 1
        # m = (n-1)s - 2p + k = ns
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    scale down the feature map by 1/2 times
    """
    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net
    """
    def __init__(self, image_channels: int, n_channels: int, ch_mults: Union[Tuple[int, ...], List[int]],
                 is_attn: Union[Tuple[bool, ...], List[int]], n_blocks: int):
        """

        :param image_channels: number of channels of the imgs
        :param n_channels: number of channels in the initial feature map
        :param ch_mults: list of channels numbers at each resolution
        :param is_attn: whether to use attention at each resolution
        :param n_blocks: number of UpDownBlocks at each resolution
        """
        super().__init__()
        # Number of resolutions
        n_resolutions = len(ch_mults)

        # project image into feature map
        self.img_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3,3), padding=(1,1))

        # the embedding layer time embedding has n_channels * 4 channels
        self.time_emb = TimeEmbedding(n_channels * 4)

        # First half of U-Net -- decreasing resolution
        down = []
        out_channels = in_channels = n_channels
        # for each resolution
        for i in range(n_resolutions):
            # number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # add n_blocks
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            # down sample at all resolution except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # combine the set of modules
        self.down = nn.ModuleList(down)

        # middle block
        self.middle = MiddleBlock(out_channels, n_channels*4)

        # second half of U-net increasing resolution
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            # n_blocks at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels*4, is_attn[i]))
            # final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels*4, is_attn[i]))
            in_channels = out_channels
            # up sample at all resolution except last
            if i > 0:
                up.append(Upsample(in_channels))

        self.up = nn.ModuleList(up)
        # final normalization and conv layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        :param x: [batch_size, in_channels, height, width]
        :param t: [batch_size]
        """
        # get time-step embeddings
        t = self.time_emb(t)
        x = self.img_proj(x)

        # h will store outputs at each resolution for skip connection
        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Middle
        x = self.middle(x, t)

        # second half
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # get skip connection from first half and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)

        return self.final(self.act(self.norm(x)))

