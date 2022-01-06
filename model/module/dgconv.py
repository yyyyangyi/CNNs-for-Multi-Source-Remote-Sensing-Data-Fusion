#!/usr/bin/env python
# coding: utf-8

'''
This implementation of Dynamic Grouping Convolution (DGConv) is based on 
https://github.com/d-li14/dgconv.pytorch
'''


import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def _aggregate(gate, D, I, K, sort=True):
    if sort:
        _, ind = gate.sort(descending=True)
        gate = gate[:, ind[0, :]]

    U = [(gate[0, i] * D + gate[1, i] * I) for i in range(K)]
    while len(U) != 1:
        temp = []
        for i in range(0, len(U) - 1, 2):
            temp.append(_kronecker_product(U[i], U[i + 1]))
        if len(U) % 2 != 0:
            temp.append(U[-1])
        del U
        U = temp

    return U[0], gate

def _kronecker_product(mat1, mat2):
    return torch.ger(mat1.view(-1), mat2.view(-1)).reshape(*(mat1.size() + mat2.size())).permute(
        [0, 2, 1, 3]).reshape(mat1.size(0) * mat2.size(0), mat1.size(1) * mat2.size(1))


class DGConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, sort=True, groups=1):
        super(DGConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                              padding=padding, dilation=dilation, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer('D', torch.eye(2))
        self.register_buffer('I', torch.ones(2, 2))
        
        self.groups = groups
        if groups > 1:
            self.register_buffer('group_mask', _kronecker_product(torch.ones(out_channels//groups, in_channels//groups), torch.eye(groups)))
        
        if self.out_channels // self.in_channels >= 2:    # Group-up
            self.K = int(np.ceil(math.log2(in_channels)))  # U: [in_channels, in_channels]
            r = int(np.ceil(self.out_channels / self.in_channels))
            _I = _kronecker_product(torch.eye(self.in_channels), torch.ones(r,1))
            self._I = nn.Parameter(_I, requires_grad=False)
        elif self.in_channels // self.out_channels >= 2:  # Group-down
            self.K = int(np.ceil(math.log2(out_channels)))  # U: [out_channels, out_channels]
            r = int(np.ceil(self.in_channels / self.out_channels))
            _I = _kronecker_product(torch.eye(self.out_channels), torch.ones(1,r))
            self._I = nn.Parameter(_I, requires_grad=False)
        else:
            # in_channels=out_channels, or either one is not the multiple of the other
            self.K = int(np.ceil(math.log2(max(in_channels, out_channels))))
            
        eps = 1e-8
        gate_init = [eps * random.choice([-1, 1]) for _ in range(self.K)]
        self.register_parameter('gate', nn.Parameter(torch.Tensor(gate_init)))
        self.sort = sort

    def forward(self, x):
        setattr(self.gate, 'org', self.gate.data.clone())
        self.gate.data = ((self.gate.org - 0).sign() + 1) / 2.
        U_regularizer =  2 ** (self.K  + torch.sum(self.gate))
        gate = torch.stack((1 - self.gate, self.gate))
        self.gate.data = self.gate.org # Straight-Through Estimator
        U, gate = _aggregate(gate, self.D, self.I, self.K, sort=self.sort)
        if self.out_channels // self.in_channels >= 2:    # Group-up
            U = torch.mm(self._I, U)
        elif self.in_channels // self.out_channels >= 2:  # Group-down
            U = torch.mm(U, self._I)

        U = U[:self.out_channels, :self.in_channels]
        if self.groups > 1:
            U = U * self.group_mask
        masked_weight = self.conv.weight * U.view(self.out_channels, self.in_channels, 1, 1)
            
        x = F.conv2d(x, masked_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)
        return x

