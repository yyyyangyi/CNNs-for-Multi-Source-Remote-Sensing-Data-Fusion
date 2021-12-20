#!/usr/bin/env python
# coding: utf-8


# Run the model on test sets
# Output is prediction map of same shape as y_test


import numpy as np
import torch


def test_seg(model, X_test, sample_h=128, sample_w=128):
    data_rows = X_test.shape[1]
    data_cols = X_test.shape[2]

    model.eval()
    pred_map = np.zeros((data_rows, data_cols))
    for r in range(0, data_rows, int(sample_h)):
        for c in range(0, data_cols, int(sample_w)):
            if r+sample_h > data_rows:
                bottum = data_rows
                top = data_rows - sample_h
            else:
                bottum = r + sample_h
                top = r
            if c+sample_w > data_cols:
                left = data_cols - sample_w
                right = data_cols
            else:
                left = c
                right = c + sample_w
            x = X_test[:, top:bottum, left:right]
            x = torch.Tensor(x)
            x = x.unsqueeze(0).cuda()
            outputs = model(x)
            _, pred = torch.max(outputs.data, 1)
            pred += 1
            pred_map[top:bottum, left:right] = pred.cpu().numpy()
    return pred_map

def test_clf(model, X_test, y_test=None, sample_radius=8, batch_test=64):
    # input y_test if output test samples only
    data_rows = X_test.shape[1]
    data_cols = X_test.shape[2]

    model.eval()
    pred_all = []
    x_batch = []
    batch_count = 0
    for r in range(sample_radius, data_rows-sample_radius):
        for c in range(sample_radius, data_cols-sample_radius):
            if y_test is not None:
                if y_test[r-sample_radius, c-sample_radius] > 0:
                    x = X_test[:, r-sample_radius:r+sample_radius+1, c-sample_radius:c+sample_radius+1]
                    x_batch.append(x)
                    batch_count += 1
                    if batch_count == batch_test:
                        x_batch = torch.Tensor(np.array(x_batch))
                        x_batch = x_batch.cuda()
                        output = model(x_batch)
                        _, pred = torch.max(output.data, 1)
                        pred += 1
                        pred_all.append(pred.cpu().numpy())
                        batch_count = 0
                        x_batch = []
            else:
                x = X_test[:, r-sample_radius:r+sample_radius+1, c-sample_radius:c+sample_radius+1]
                x_batch.append(x)
                batch_count += 1
                if batch_count == batch_test:
                    x_batch = torch.Tensor(np.array(x_batch))
                    x_batch = x_batch.cuda()
                    output = model(x_batch)
                    _, pred = torch.max(output.data, 1)
                    pred += 1
                    pred_all.append(pred.cpu().numpy())
                    batch_count = 0
                    x_batch = []
                    
    if batch_count > 0:
        x_batch = torch.Tensor(np.array(x_batch))
        x_batch = x_batch.cuda()
        output = model(x_batch)
        _, pred = torch.max(output.data, 1)
        pred += 1
        pred_all.append(pred.cpu().numpy())
        
    output = np.hstack(np.array(pred_all).flatten())
    if y_test is not None:
        return output
    else:
        return output.reshape(X_test.shape[1]-2*sample_radius, X_test.shape[2]-2*sample_radius)

