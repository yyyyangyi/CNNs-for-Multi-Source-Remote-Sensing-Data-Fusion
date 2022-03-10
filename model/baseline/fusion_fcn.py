#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Fusion_FCN(nn.Module):
    
    def __init__(self, in_channel_branch=[6,7,55], n_classes=1000, use_init=False):
        super(Fusion_FCN, self).__init__()
        self.in_channel_branch = in_channel_branch
        
        # VHRI+LiDAR branch
        self.branch1_conv1 = nn.Conv2d(in_channel_branch[0], 64, kernel_size=3, padding=1)
        self.branch1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.branch1_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # NDSM branch
        self.branch2_conv1 = nn.Conv2d(in_channel_branch[1]-in_channel_branch[0], 64, kernel_size=3, padding=1)
        self.branch2_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.branch2_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.out_conv = nn.Conv2d(176, n_classes, kernel_size=1)
        
        if use_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        # VHRI+LiDAR branch
        x1 = self.branch1_conv1(x[:,:self.in_channel_branch[0],...])
        x1 = self.relu(x1)
        x1 = F.pad(x1, (0,1,0,1), mode='replicate')
        x11 = x1 = self.avgpool(x1)
        x1 = self.branch1_conv2(x1)
        x1 = self.relu(x1)
        x1 = F.pad(x1, (0,1,0,1), mode='replicate')
        x12 = x1 = self.avgpool(x1)
        x1 = self.branch1_conv3(x1)
        x1 = self.relu(x1)
        x1 = F.pad(x1, (0,1,0,1), mode='replicate')
        x13 = x1 = self.avgpool(x1)
        x1 = (x11 + x12 + x13) / 3
        # NDSM branch
        x2 = self.branch2_conv1(x[:,self.in_channel_branch[0]:self.in_channel_branch[1],...])
        x2 = self.relu(x2)
        x2 = F.pad(x2, (0,1,0,1), mode='replicate')
        x21 = x2 = self.avgpool(x2)
        x2 = self.branch2_conv2(x2)
        x2 = self.relu(x2)
        x2 = F.pad(x2, (0,1,0,1), mode='replicate')
        x22 = x2 = self.avgpool(x2)
        x2 = self.branch2_conv3(x2)
        x2 = self.relu(x2)
        x2 = F.pad(x2, (0,1,0,1), mode='replicate')
        x23 = x2 = self.avgpool(x2)
        x2 = (x21 + x22 + x23) / 3
        # output
        output = torch.cat((x1, x2, x[:,self.in_channel_branch[1]:self.in_channel_branch[2],...]), dim=1)
        output = self.out_conv(output)
        output = self.relu(output)
        return output



