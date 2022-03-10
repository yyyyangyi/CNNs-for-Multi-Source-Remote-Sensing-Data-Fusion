#!/usr/bin/env python
# coding: utf-8

# Some helper functions

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

    
def _get_dataset(ds_name, ds_dir):
    if ds_name == 'houston':
        from data.dataset_houston import load_dataset
    elif ds_name == 'berlin':
        from data.dataset_berlin import load_dataset
    elif ds_name == 'muufl':
        from data.dataset_muufl import load_dataset
    else:
        raise NotImplementedError('Data set not implemented!')
    return load_dataset(ds_dir)
    
def _split_train_val(X, y, train_ratio=1.0):
    X_tensor = torch.Tensor(X)
    y_tensor = torch.LongTensor(y)
    data_set = TensorDataset(X_tensor, y_tensor)
    total_size = len(data_set)
    train_size = int(total_size*train_ratio)
    val_size = total_size - train_size
    if val_size > 0:
        train_set, val_set = torch.utils.data.random_split(data_set, [train_size, val_size])
    else:
        train_set = data_set
        val_set = None
    return train_set, val_set

def _get_class_weights(y, num_classes, mask):
    num_samples = np.zeros((num_classes,))
    if mask:
        for i in range(1, num_classes+1):
            num_samples[i-1] = np.sum(y==i)
    else:
        for i in range(num_classes):
            num_samples[i] = np.sum(y==i)
    class_weights = [1 - (n / sum(num_samples)) for n in num_samples]
    return torch.FloatTensor(class_weights)

def _get_model(model_name='resnet18', ckpt=None, **kwargs):
    if model_name == 'unet':
        from model.unet import UNet
        model = UNet(**kwargs)
    elif model_name == 'resnet18':
        from model.resnet import resnet18
        model = resnet18(**kwargs)
    elif model_name == 'resnet50':
        from model.resnet import resnet50
        model = resnet50(**kwargs)
    # baseline models
    elif model_name == 'fusion_fcn':
        from model.baseline.fusion_fcn import Fusion_FCN
        model = Fusion_FCN(**kwargs)
    elif model_name == 'tb_cnn':
        from model.baseline.tb_cnn import TB_CNN
        model = TB_CNN(**kwargs)
    else:
        raise NotImplementedError('Model not implemented!')
    if ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    return model
        
def _get_optimizer(model, opt_name='adam', lr=0.001, ckpt=None):
    if opt_name == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer not implemented!')
    if ckpt: 
        optim.load_state_dict(ckpt['optimizer_state_dict'])
    return optim

    