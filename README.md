# CNNs for Multi-Source Remote Sensing Data Fusion

## Description

Pytorch implementation of the paper "Single-stream CNN with Learnable Architecture for Multi-source Remote Sensing Data". (under review) [[arxiv]](http://arxiv.org/abs/2109.06094)

Multi-stream CNNs are commonly used in multi-source remote sensing data fusion. In this work we propose an efficient strategy that enables single-stream CNNs to approximate multi-stream models using group convolution. The proposed method is applied to ResNet and UNet, and evaluated on Houston2018, Berlin, MUUFL data sets, obtaining promising results. An interesting finding is that regularization is very important in these models. 

## Usage
- Requirements: python3, pytorch, gdal, sklearn. 
- Simply run 
```
python3 main.py
```
- To customize training/model arguments, modify ```common.py```. Arguments are automatically loaded to ```main.py```.

## Baseline models

This repository also contains Pytorch implementation of the following models, which we use as baselines: 

_Fusion-FCN_: A three-branch CNN for MS-HSI-LiDAR data fusion. Award-winning model in 2018 IEEE DFC. [[Paper]](https://ieeexplore.ieee.org/abstract/document/8518295/)

_Two-branch CNN_ (_TB-CNN_): A two-branch CNN architecture for feasture fusion with HSI and other remote scensing imagery. [[Paper]](https://ieeexplore.ieee.org/abstract/document/8518295/) [[Official Tensorflow implementation]](https://github.com/Hsuxu/Two-branch-CNN-Multisource-RS-classification)

Implementation of these models can be found at ```models/baseline/```. 

## Data
We made some modifications (merely tif→numpy, stacking) to the original data files. Our data files are available at [this Google Drive site](https://drive.google.com/drive/folders/1urY6Pjba3mStDcRphIfkNf50295aW2o2?usp=sharing), which can be directly used in this code. Please note that we used channel-wise normalization AFTER loading these files, and this step is already implemented in our code. 

Below are links to the original data sets:

[[Houston2018]](https://ieee-dataport.org/open-access/2018-ieee-grss-data-fusion-challenge-%E2%80%93-fusion-multispectral-lidar-and-hyperspectral-data) &nbsp;
[[Berlin]](https://github.com/danfenghong/ISPRS_S2FL) &nbsp;
[[MUUFL]](https://github.com/GatorSense/MUUFLGulfport/tree/master/MUUFLGulfportSceneLabels) &nbsp;

## Results

| Dataset | OA (%) | Kappa |
| --- | ----------- | ----- |
| Houston2018 | 63.74 | 0.62 |
| Berlin | 68.21 | 0.54 |
| MUUFL | 86.44 | 0.83 |
