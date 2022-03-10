#!/usr/bin/env python
# coding: utf-8


# Model training


import os
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR

def _checkpoint(model, optimizer, ckpt_path, loss=0.0):
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, ckpt_path)
    
def train(model, train_loader, optimizer, criterion, num_epochs=100, rep=0, mask_undefined=True, 
          save_ckpt_dir=None, use_gpu=True, lr_schedule=None, verbose=True):
    
    if lr_schedule is not None:
        scheduler = MultiStepLR(optimizer, milestones=lr_schedule, gamma=0.1)
    losses = np.zeros((num_epochs,))
    model.train()
    for epoch in range(num_epochs):
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, (x, label) in enumerate(train_loader):

            if mask_undefined:
                target = label.clone()
                target[target==0] = 1
                target -= 1
                mask = label!=0
                mask = mask.float()
                if use_gpu:
                    mask = mask.cuda()
                    target = target.cuda()
            if use_gpu:
                x = x.cuda()
                label = label.cuda()

            length = len(train_loader)
            optimizer.zero_grad()

            pred = model(x)
            if mask_undefined:
                loss = criterion(pred, target)
                loss = (loss * mask).sum() / mask.sum()
            else:
                loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            if verbose:
                _, predicted = torch.max(pred.data, 1)
                if mask_undefined:
                    total += mask.sum()
                    predicted += 1
                    correct += ((predicted.eq(label.data))*mask).cpu().sum()
                else:
                    total += len(label)
                    correct += (predicted.eq(label.data)).cpu().sum() 
            
        if verbose:
            if epoch % 10 == 0:                      
                print('[epoch:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, sum_loss / (i + 1), 100. * correct / total))
        
        if save_ckpt_dir:
            if epoch % 100 == 99:
                ckpt_path = os.path.join(save_ckpt_dir, 'ckpt_rep%d_epoch%d.pth'%(rep,epoch))
                _checkpoint(model, optimizer, ckpt_path, sum_loss)
            
        losses[epoch] = sum_loss
        if lr_schedule is not None:
            scheduler.step()
    
    return model, losses

