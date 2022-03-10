#!/usr/bin/env python
# coding: utf-8


# This implementation of UNet is based on 
# https://github.com/milesial/Pytorch-UNet


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.module.dgconv import DGConv2d
from model.module.fgconv import FGConv2d


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, use_dgconv=True, groups=1, in_groups=None, out_groups=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if use_dgconv:
            if in_groups is not None:
                self.double_conv = nn.Sequential(
                    FGConv2d(in_channels, mid_channels, kernel_size=3, padding=1, 
                             in_groups=in_groups, out_groups=out_groups),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    DGConv2d(mid_channels, out_channels, kernel_size=3, padding=1, groups=groups),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                self.double_conv = nn.Sequential(
                    DGConv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=groups),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    DGConv2d(mid_channels, out_channels, kernel_size=3, padding=1, groups=groups),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
        else:
            if in_groups is not None:
                self.double_conv = nn.Sequential(
                    FGConv2d(in_channels, mid_channels, kernel_size=3, padding=1, 
                             in_groups=in_groups, out_groups=out_groups),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, groups=groups),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=groups),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, groups=groups),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )

    def forward(self, x):
        return self.double_conv(x)

    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, use_dgconv=True, groups=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_dgconv=use_dgconv, groups=groups)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, use_dgconv=True, groups=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, use_dgconv=use_dgconv, groups=groups)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2, groups=groups)
            self.conv = DoubleConv(in_channels, out_channels, use_dgconv=use_dgconv, groups=groups)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
    
class UNet(nn.Module):
    def __init__(self, input_channels, n_classes, bilinear=False, use_dgconv=True, use_init=False, fix_groups=1):
        super(UNet, self).__init__()
        self.n_channels = input_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        if fix_groups > 1:
            self.inc = DoubleConv(self.n_channels, 64, use_dgconv=use_dgconv, groups=fix_groups, 
                                 in_groups=[3,48,3,4], out_groups=[16,16,16,16])
#             self.inc = DoubleConv(self.n_channels, 64, use_dgconv=use_dgconv, groups=fix_groups, 
#                                  in_groups=[14,16,14,14], out_groups=[16,16,16,16])
        else:
            self.inc = DoubleConv(self.n_channels, 64, use_dgconv=use_dgconv)
        self.down1 = Down(64, 128, use_dgconv=use_dgconv, groups=fix_groups)
        self.down2 = Down(128, 256, use_dgconv=use_dgconv, groups=fix_groups)
        self.down3 = Down(256, 512, use_dgconv=use_dgconv, groups=fix_groups)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, use_dgconv=use_dgconv, groups=fix_groups)
        self.up1 = Up(1024, 512 // factor, bilinear, use_dgconv=use_dgconv, groups=fix_groups)
        self.up2 = Up(512, 256 // factor, bilinear, use_dgconv=use_dgconv, groups=fix_groups)
        self.up3 = Up(256, 128 // factor, bilinear, use_dgconv=use_dgconv, groups=fix_groups)
        self.up4 = Up(128, 64, bilinear, use_dgconv=use_dgconv, groups=fix_groups)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

