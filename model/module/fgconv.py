#!/usr/bin/env python
# coding: utf-8

'''
Fixed group convolution
'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FGConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, in_groups=None, out_groups=None):
        super(FGConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                              padding=padding, dilation=dilation, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_groups = list(np.copy(in_groups))
        self.out_groups = list(np.copy(out_groups))
        
        if in_groups is None or out_groups is None:
            U = torch.ones(self.out_channels, self.in_channels)
        else:
            self.in_groups.insert(0, 0)
            self.out_groups.insert(0, 0)
            self.in_groups = np.cumsum(self.in_groups)
            self.out_groups = np.cumsum(self.out_groups)
            U = torch.zeros(self.out_channels, self.in_channels)
            for i in range(1, len(self.in_groups)):
                U[self.out_groups[i-1]:self.out_groups[i], self.in_groups[i-1]:self.in_groups[i]] = 1
        
        U = U.view(self.out_channels, self.in_channels, 1, 1)
        self.register_buffer('U', U)

    def forward(self, x):
        masked_weight = self.conv.weight * self.U
        x = F.conv2d(x, masked_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)
        return x

