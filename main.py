#!/usr/bin/env python
# coding: utf-8


import os
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

from common import Config
from train import train
import test
    

def _seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
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

def _get_model(model_name='resnet18', **kwargs):
    if model_name == 'unet':
        from model.unet import UNet
        return UNet(**kwargs)
    elif model_name == 'resnet18':
        from model.resnet import resnet18
        return resnet18(**kwargs)
    elif model_name == 'resnet50':
        from model.resnet import resnet50
        return resnet50(**kwargs)
    else:
        raise NotImplementedError('Model not implemented!')
        
def _get_optimizer(model, opt_name='adam', lr=0.001):
    if opt_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer not implemented!')
    
def main():
 
    X, y, X_test, y_test = _get_dataset(Config.dataset, Config.data_dir)
    
    num_replicates = Config.num_replicates
    num_classes = Config.num_classes
    epochs = Config.epochs
    seed = Config.seed
    if Config.result_out_dir is not None:
        conf_mats = np.zeros((num_replicates, num_classes, num_classes))
        p_scores = np.zeros((num_replicates, num_classes))
        r_scores = np.zeros((num_replicates, num_classes))
        f1_scores = np.zeros((num_replicates, num_classes))
        k_scores = np.zeros((num_replicates,))
        oa_arr = np.zeros((num_replicates,))
    
    for rep in range(num_replicates):
        _seed = seed + rep
        _seed_everything(_seed)
        
        train_set, val_set = _split_train_val(X, y, train_ratio=1.0)
        train_loader = DataLoader(dataset=train_set, batch_size=Config.batch_size, shuffle=True)
        
        class_weights = _get_class_weights(y, num_classes, Config.mask_undefined)
        if Config.use_gpu:
            class_weights = class_weights.cuda()
        if Config.mask_undefined:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        else:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        model = _get_model(Config.model, input_channels=X_test.shape[0], n_classes=num_classes, 
                           use_dgconv=True, use_init=Config.use_init)
        optimizer = _get_optimizer(model, opt_name=Config.optimizer, lr=Config.lr)
        # train
        model, losses = train(model, train_loader, optimizer, criterion, 
                              num_epochs=epochs, mask_undefined=Config.mask_undefined, 
                              save_ckpt_dir=Config.save_ckpt_dir, use_gpu=Config.use_gpu, 
                              lr_schedule=Config.lr_schedule, verbose=True)
        # test & eval
        if Config.result_out_dir is not None:
            if Config.dataset == 'houston':
                pred_map = test.test_seg(model, X_test, Config.sample_h, Config.sample_w)
            else:
                pred_map = test.test_clf(model, X_test, sample_radius=Config.sample_radius)
            y_pred_all = y_pred_all[y_test>0]
            y_true_all = y_test[y_test>0]
            conf_mats[rep] = confusion_matrix(y_true_all, y_pred_all)
            p_scores[rep] = precision_score(y_true_all, y_pred_all, average=None)
            r_scores[rep] = recall_score(y_true_all, y_pred_all, average=None)
            f1_scores[rep] = f1_score(y_true_all, y_pred_all, average=None)
            k_scores[rep] = cohen_kappa_score(y_true_all, y_pred_all)
            oa_arr[rep] = np.sum(conf_mats[rep]*np.eye(num_classes, num_classes)) / np.sum(conf_mats[rep])
    
    if Config.result_out_dir is not None:
        out_dir = Config.result_out_dir
        np.save(os.path.join(out_dir, 'conf_mats.npy'), conf_mats)
        np.save(os.path.join(out_dir, 'losses.npy'), losses)
        np.save(os.path.join(out_dir, 'p_scores.npy'), p_scores)
        np.save(os.path.join(out_dir, 'r_scores.npy'), r_scores)
        np.save(os.path.join(out_dir, 'f1_scores.npy'), f1_scores)
        np.save(os.path.join(out_dir, 'k_scores.npy'), k_scores)
        np.save(os.path.join(out_dir, 'oa_arr.npy'), oa_arr)
   

if __name__ == '__main__':
    main()

