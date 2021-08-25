# Multi-source-RS-DGConv

## Description

Implementation of the paper "Multi-stream as Regularization: Rethinking Convolutional 
Neural Network Architecture for Multi-source Remote Sensing Data". (under review)

## Usage
- Requirements: python3, pytorch, gdal, sklearn. 
- Simply run 
```
python3 main.py
```
- Training / model arguments can be changed by modifying ```common.py```. Arguments are automatically loaded to ```main.py```.

## Data
Houston2018 data set is available at https://ieee-dataport.org/open-access/2018-ieee-grss-data-fusion-challenge-%E2%80%93-fusion-multispectral-lidar-and-hyperspectral-data.

Berlin data set is available at https://github.com/danfenghong/ISPRS_S2FL.

MUUFL data set is available at https://github.com/GatorSense/MUUFLGulfport/tree/master/MUUFLGulfportSceneLabels.
