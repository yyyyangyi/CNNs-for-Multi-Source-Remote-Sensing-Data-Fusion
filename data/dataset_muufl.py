#!/usr/bin/env python
# coding: utf-8

# Load MUUFL data set


import os
import numpy as np
import scipy.io as scio
from common import Config


SAMPLE_RADIUS = Config.sample_radius
NUM_CLASSES = Config.num_classes

def load_dataset(data_dir):
    _TRAIN_SIZE = 100
    data_lidar = scio.loadmat(os.path.join(data_dir, 'lidar_data.mat'))['lidar_data']
    data_hsi = scio.loadmat(os.path.join(data_dir, 'hsi_data.mat'))['hsi_data']
    labels = scio.loadmat(os.path.join(data_dir, 'labels.mat'))['labels']
    
    rows, cols = labels.shape
    X_data = np.zeros((rows+SAMPLE_RADIUS*2, cols+SAMPLE_RADIUS*2, data_hsi.shape[2]+data_lidar.shape[2]))
    X_data[SAMPLE_RADIUS:-SAMPLE_RADIUS, SAMPLE_RADIUS:-SAMPLE_RADIUS, :] = np.concatenate((data_hsi, data_lidar), axis=-1)
    X_data = np.transpose(X_data, (2,0,1))
    
    # channel-wise normalization
    for b in range(X_data.shape[0]):
        band = X_data[b, ...]
        X_data[b, ...] = (band-np.min(band)) / (np.max(band)-np.min(band))
    
    X = []
    y = []
    for r in range(SAMPLE_RADIUS, rows+SAMPLE_RADIUS):
        for c in range(SAMPLE_RADIUS, cols+SAMPLE_RADIUS):
            sample = X_data[:, r-SAMPLE_RADIUS:r+SAMPLE_RADIUS+1, c-SAMPLE_RADIUS:c+SAMPLE_RADIUS+1]
            label = labels[r-SAMPLE_RADIUS, c-SAMPLE_RADIUS]
            if label > 0:
                X.append(sample)
                y.append(label)
    
    X = np.array(X)
    y = np.array(y)-1
    
    idx_train = []
    for i in range(NUM_CLASSES):
        idx_samples = np.arange(len(y))[y==i]
        idx_train.append(np.random.choice(idx_samples, _TRAIN_SIZE, replace=False))
    idx_train = np.array(idx_train).flatten()
            
    return np.array(X[idx_train]), np.array(y[idx_train], dtype=np.int32), X_data, labels

