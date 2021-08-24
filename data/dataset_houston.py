#!/usr/bin/env python
# coding: utf-8


# Load Houston data set from 2018 IEEE Data Fusion Contest


import os
import numpy as np
import gdal

X_TRAIN_FILE = 'X_train.npy'
X_TEST_FILE = 'X_test.npy'
Y_TRAIN_FILE = '2018_IEEE_GRSS_DFC_GT_TR.tif'
Y_TEST_FILE = 'Test_Labels.tif'

SAMPLE_H = 128
SAMPLE_W = 128
STRIDE_ROW = int(SAMPLE_H/2)
STRIDE_COL = int(SAMPLE_W/2)
NUM_CLASSES = 20


def _get_data_set(X_data, y_data):    
    data_rows = X_data.shape[1]
    data_cols = X_data.shape[2]
    assert (data_rows, data_cols) == (y_data.shape[0], y_data.shape[1])

    X = []
    y = []
    for r in range(0, data_rows, STRIDE_ROW):
        for c in range(0, data_cols, STRIDE_COL):
            if r+SAMPLE_H > data_rows:
                bottum = data_rows
                top = data_rows - SAMPLE_H
            else:
                bottum = r + SAMPLE_H
                top = r
            if c+SAMPLE_W > data_cols:
                left = data_cols - SAMPLE_W
                right = data_cols
            else:
                left = c
                right = c + SAMPLE_W
            X.append(X_data[:, top:bottum, left:right])
            y.append(y_data[top:bottum, left:right])
            
    return np.array(X), np.array(y)

def load_dataset(data_dir):
    X_train_data = np.load(os.path.join(data_dir, X_TRAIN_FILE))
    y_train_data = gdal.Open(os.path.join(data_dir, Y_TRAIN_FILE)).ReadAsArray()
    X_test_data = np.load(os.path.join(data_dir, X_TEST_FILE))
    y_test_data = gdal.Open(os.path.join(data_dir, Y_TEST_FILE)).ReadAsArray()
    X, y = _get_data_set(X_train_data, y_train_data)
    return X, y, X_test_data, y_test_data

