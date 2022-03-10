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
import _utils
    

def _seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def main():
 
    if Config.dataset != 'muufl':
        X, y, X_test, y_test = _utils._get_dataset(Config.dataset, Config.data_dir)
    
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
        # fix random seeds
        _seed = seed + rep
        _seed_everything(_seed)
        
        # prepare data
        if Config.dataset == 'muufl':
            X, y, X_test, y_test = _utils._get_dataset(Config.dataset, Config.data_dir)
        train_set, val_set = _utils._split_train_val(X, y, train_ratio=1.0)
        train_loader = DataLoader(dataset=train_set, batch_size=Config.batch_size, shuffle=True)
        
        # prepare model and optimizer
        class_weights = _utils._get_class_weights(y, num_classes, Config.mask_undefined)
        if Config.use_gpu:
            class_weights = class_weights.cuda()
        if Config.mask_undefined:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        else:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            
        if Config.ckpt_dir:
            ckpt = torch.load(os.path.join(Config.ckpt_dir, 'ckpt_rep%d_epoch100.pth'%rep))
        else:
            ckpt = None
        if Config.model == 'fusion_fcn':
            model = _utils._get_model(Config.model, ckpt=ckpt, in_channel_branch=[6, 7, 55], n_classes=num_classes, use_init=Config.use_init)
        elif Config.model == 'tb_cnn':
            model = _utils._get_model(Config.model, ckpt=ckpt, in_channel_branch=[64, 2], n_classes=num_classes, patch_size=Config.sample_radius)
        else:
            # ResNets
            model = _utils._get_model(Config.model, ckpt=ckpt, input_channels=X_test.shape[0], n_classes=num_classes, 
                               use_dgconv=Config.use_dgconv, use_init=Config.use_init, fix_groups=Config.fix_groups)
        if Config.use_gpu:
            model = model.cuda()
        optimizer = _utils._get_optimizer(model, opt_name=Config.optimizer, lr=Config.lr, ckpt=ckpt)
        
        # train
        model, losses = train(model, train_loader, optimizer, criterion, 
                              num_epochs=epochs, rep=rep, mask_undefined=Config.mask_undefined, 
                              save_ckpt_dir=Config.save_ckpt_dir, use_gpu=Config.use_gpu, 
                              lr_schedule=Config.lr_schedule, verbose=True)
        # test & eval
        if Config.result_out_dir is not None:
            if Config.dataset == 'houston':
                pred_map = test.test_seg(model, X_test, Config.sample_h, Config.sample_w)
            else:
                pred_map = test.test_clf(model, X_test, sample_radius=Config.sample_radius)
            y_pred_all = pred_map[y_test>0]
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

