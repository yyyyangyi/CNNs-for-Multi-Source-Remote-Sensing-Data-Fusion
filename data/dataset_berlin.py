#!/usr/bin/env python
# coding: utf-8


# Load Berlin data set


import os
import numpy as np
import scipy.io as scio
from common import Config

# CATEGORIES = ['UNDEFINED', 'Forest', 'Residential Area', 'Industrial Area', 'Low Plants', 'Soil', 
#              'Allotment', 'Commercial Area', 'Water']
SAMPLE_RADIUS = Config.sample_radius


def load_dataset(data_dir):
    _HSI_DATA = 'data_HS_LR'
    _SAR_DATA = 'data_SAR_HR'
    _TRAIN_LABELS = 'TrainImage'
    _TEST_LABELS = 'TestImage'
    
    data_hsi = scio.loadmat(os.path.join(data_dir, _HSI_DATA+'.mat'))[_HSI_DATA]
    data_sar = scio.loadmat(os.path.join(data_dir, _SAR_DATA+'.mat'))[_SAR_DATA]
    y_train = scio.loadmat(os.path.join(data_dir, _TRAIN_LABELS+'.mat'))[_TRAIN_LABELS]
    y_test = scio.loadmat(os.path.join(data_dir, _TEST_LABELS+'.mat'))[_TEST_LABELS]
    assert data_hsi.shape[0]==data_sar.shape[0]==y_train.shape[0]==y_test.shape[0], 'Dimension of data arrays does not match'
    assert data_hsi.shape[1]==data_sar.shape[1]==y_train.shape[1]==y_test.shape[1], 'Dimension of data arrays does not match'
    
    rows, cols = y_train.shape
    X_data = np.zeros((rows+SAMPLE_RADIUS*2, cols+SAMPLE_RADIUS*2, data_hsi.shape[2]+data_sar.shape[2]))    # zero padding
    X_data[SAMPLE_RADIUS:-SAMPLE_RADIUS, SAMPLE_RADIUS:-SAMPLE_RADIUS, :] = np.concatenate((data_hsi, data_sar), axis=-1)
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
            label = y_train[r-SAMPLE_RADIUS, c-SAMPLE_RADIUS]
            if label > 0:
                X.append(sample)
                y.append(label)
            
    return np.array(X), np.array(y)-1, X_data, y_test

