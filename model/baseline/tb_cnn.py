#!/usr/bin/env python
# coding: utf-8


# Based on the official Tensorflow implementation
# https://github.com/Hsuxu/Two-branch-CNN-Multisource-RS-classification


import numpy as np
import torch
import torch.nn as nn


class CascadeBlock(nn.Module):
    
    def __init__(self, in_channels, nb_filters, kernel_size=3):
        super(CascadeBlock, self).__init__()
        
        self.conv1_1 = nn.Conv2d(in_channels, nb_filters*2, kernel_size=kernel_size, padding=1)
        self.bn1_1 = nn.BatchNorm2d(nb_filters*2)
        self.conv1_2 = nn.Conv2d(nb_filters*2, nb_filters, kernel_size=1)
        self.bn1_2 = nn.BatchNorm2d(nb_filters)
        
        self.mid_conv = nn.Conv2d(in_channels, nb_filters*2, kernel_size=1, bias=False)
        
        self.conv2_1 = nn.Conv2d(nb_filters, nb_filters*2, kernel_size=kernel_size, padding=1)
        self.bn2_1 = nn.BatchNorm2d(nb_filters*2)
        self.conv2_2 = nn.Conv2d(nb_filters*2, nb_filters, kernel_size=kernel_size, padding=1)
        self.bn2_2 = nn.BatchNorm2d(nb_filters)
        
        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x0 = x
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = x1 = self.bn1_2(x)
        x = self.lrelu(x)
        
        x = self.conv2_1(x) + self.mid_conv(x0)
        x = self.bn2_1(x)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.lrelu(x+x1)

        return x
    
    
class CascadeNet(nn.Module):    
    
    def __init__(self, in_channels):
        super(CascadeNet, self).__init__()
    
        filters = [16, 32, 64, 96, 128, 192, 256, 512]
        self.conv0 = nn.Conv2d(in_channels, filters[2], kernel_size=3, padding=1)
        self.cascade_block1 = CascadeBlock(in_channels=filters[2], nb_filters=filters[2])
        self.cascade_block2 = CascadeBlock(in_channels=filters[2], nb_filters=filters[4])
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.maxpool = nn.MaxPool2d(2, padding=1)
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.lrelu(x)
        x = self.cascade_block1(x)
        x = self.maxpool(x)
        x = self.lrelu(x)
        x = self.cascade_block2(x)
        return torch.flatten(x, start_dim=1)
    
    
class SimpleCNNBranch(nn.Module):
    
    def __init__(self, in_channels):
        super(SimpleCNNBranch, self).__init__()
    
        self.conv0 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.maxpool = nn.MaxPool2d(2, padding=1)
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.maxpool(x)
        return torch.flatten(x, start_dim=1)

    
class PixelBranch(nn.Module):
    
    def __init__(self, in_channels, patch_size=5):
        super(PixelBranch, self).__init__()
    
        filters = [8, 16, 32, 64, 96, 128]
        self.patch_size = patch_size
        self.conv0 = nn.Conv1d(in_channels, filters[3], kernel_size=11)
        self.bn0 = nn.BatchNorm1d(filters[3])
        self.conv3 = nn.Conv1d(filters[3], filters[5], kernel_size=3)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.maxpool = nn.MaxPool1d(2, padding=1)
        
    def forward(self, x):
        x = x[:, :, self.patch_size:self.patch_size+1, self.patch_size:self.patch_size+1]
        x = x.squeeze(-1)
        x = torch.transpose(x, 1, 2)
        
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.lrelu(x)
        x = self.conv3(x)
        x = self.lrelu(x)
        x = self.maxpool(x)
        return torch.flatten(x, start_dim=1)
    
    
class LidarBranch(nn.Module):
    
    def __init__(self, in_channels, out_channels=64, standalone=False):
        super(LidarBranch, self).__init__()
        
        self.standalone = standalone
        self.net = CascadeNet(in_channels)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(128, out_channels)
        
    def forward(self, x):
        x = self.net(x)
        if self.standalone: 
            x = self.dropout(x)
#             print("lidar branch, fc", x.shape)
            x = self.fc(x)
        return x
   

class HsiBranch(nn.Module):
    
    def __init__(self, in_channels, out_channels=64, patch_size=5, standalone=False):
        super(HsiBranch, self).__init__()
    
        self.standalone = standalone
        self.cnn_net = SimpleCNNBranch(in_channels)
        self.pixel_net = PixelBranch(in_channels=1, patch_size=patch_size)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(128, out_channels)
        
    def forward(self, x):
        x1 = self.cnn_net(x)
        x2 = self.pixel_net(x)
        x = torch.cat((x1,x2), dim=1)
        if self.standalone:
            x = self.dropout(x)
#             print("hsi branch, fc", x.shape)
            x = self.fc(x)
        return x
    
    
class TB_CNN(nn.Module):
    
    def __init__(self, in_channel_branch=[64,2], n_classes=10, patch_size=5):
        super(TB_CNN, self).__init__()
        
        self.in_channel_branch = in_channel_branch
        self.hsi_branch = HsiBranch(in_channels=in_channel_branch[0], patch_size=patch_size)
        self.lidar_branch = LidarBranch(in_channels=in_channel_branch[1])
        self.bn = nn.BatchNorm1d(26496)
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(26496, 128)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(128, n_classes)
        
    def forward(self, x):
        x1 = self.hsi_branch(x[:,:self.in_channel_branch[0],...])
        x2 = self.lidar_branch(x[:,self.in_channel_branch[0]:,...])
        x = torch.cat((x1,x2), dim=1)
#         print("tb_cnn, fc: ", x.shape)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.fc2(x)
        return x
    

