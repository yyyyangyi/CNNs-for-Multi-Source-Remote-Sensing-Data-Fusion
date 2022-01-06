#!/usr/bin/env python
# coding: utf-8


# Modified from torchvision implementation of ResNet


import numpy as np
import torch
import torch.nn as nn

from model.module.dgconv import DGConv2d
from model.module.fgconv import FGConv2d


class BasicBlock(nn.Module):
    '''
    DGConv block: (3x3 DGConv => 3x3 DGConv)
    Standard block: (3x3 Conv => 3x3 Conv)
    '''
    expansion = 1
    
    def __init__(self, inplanes, out_channels, stride=1, dilation=1, downsample=None, use_dgconv=True, groups=1):
        super(BasicBlock, self).__init__()
        if use_dgconv:
            self.conv1 = DGConv2d(inplanes, out_channels, 3, stride, padding=1, bias=False, groups=groups)
            self.conv2 = DGConv2d(out_channels, out_channels, 3, 1, padding=1, bias=False, groups=groups)
        else:
            self.conv1 = nn.Conv2d(inplanes, out_channels, 3, stride, padding=1, bias=False, groups=groups)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x
    
    
class Bottleneck(nn.Module):
    '''
    DGConv block: (1x1 depthwise_Conv => 3x3 DGConv => 1x1 depthwise_Conv)
    Standard block: (1x1 Conv => 3x3  Conv => 1x1 Conv)
    '''
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_dgconv=True, groups=1):
        super(Bottleneck, self).__init__()
        if use_dgconv:
            # Requires inplanes>planes & inplanes%planes=0, as in standard resnet50
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, groups=planes)
            self.conv2 = DGConv2d(planes, planes, kernel_size=3, stride=stride, 
                                  padding=dilation, bias=False, dilation=dilation, groups=groups)
            self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False, groups=planes)
        else:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, groups=groups)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=dilation, bias=False, dilation=dilation, groups=groups)
            self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False, groups=groups)
            
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
   

class ResNet(nn.Module):

    def __init__(self, block, num_layer, n_classes=1000, input_channels=3, 
                 replace_stride_with_dilation=None, use_init=False, use_dgconv=True, fix_groups=1):
        super(ResNet, self).__init__()
        self.dilation = 1
        self.inplanes = 64
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
            
        if use_dgconv:
            if fix_groups > 1:
                # muufl - [64,2]
                # berlin - [244,4]
                # houston - [3,48,3,4]
                self.conv1 = FGConv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, 
                                     in_groups=[64,2], out_groups=[32,32])
            else:
                self.conv1 = DGConv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            if fix_groups > 1:
                self.conv1 = FGConv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, 
                                     in_groups=[64,2], out_groups=[32,32])
            else:
                self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_layer[0], use_dgconv=use_dgconv, groups=fix_groups)
        self.layer2 = self._make_layer(block, 128, num_layer[1], stride=2, 
                                       dilate=replace_stride_with_dilation[0], use_dgconv=use_dgconv, groups=fix_groups)
        self.layer3 = self._make_layer(block, 256, num_layer[2], stride=2, 
                                       dilate=replace_stride_with_dilation[1], use_dgconv=use_dgconv, groups=fix_groups)
        self.layer4 = self._make_layer(block, 512, num_layer[3], stride=2, 
                                       dilate=replace_stride_with_dilation[2], use_dgconv=use_dgconv, groups=fix_groups)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(block.expansion*512, n_classes)
        
        if use_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, out_channels, num_block, stride=1, dilate=False, use_dgconv=True, groups=1):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != out_channels*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_channels*block.expansion, 1, stride=stride, bias=False, groups=groups),
                nn.BatchNorm2d(out_channels*block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, out_channels, stride=stride, downsample=downsample, 
                            dilation=previous_dilation, use_dgconv=use_dgconv, groups=groups))
        self.inplanes = out_channels*block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.inplanes, out_channels, dilation=self.dilation, use_dgconv=use_dgconv, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

