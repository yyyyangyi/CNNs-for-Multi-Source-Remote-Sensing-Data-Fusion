# Multi-source-RS-DGConv

## Description

Implementation of the paper "Single-stream CNN with Learnable Architecture for Multi-source Remote Sensing Data". (under review)

[[arxiv]](http://arxiv.org/abs/2109.06094)

## Usage
- Requirements: python3, pytorch, gdal, sklearn. 
- Simply run 
```
python3 main.py
```
- To customize training/model arguments, modify ```common.py```. Arguments are automatically loaded to ```main.py```.

## Data
We made some modifications (merely tif→numpy, stacking) to the original data files. Our data files are available at [this Google Drive site](https://drive.google.com/drive/folders/1urY6Pjba3mStDcRphIfkNf50295aW2o2?usp=sharing), which can be directly used in this code. Please note that we used channel-wise normalization AFTER loading these files, and this step is already implemented in our code. 

Below are links to the original data sets:

[Houston2018](https://ieee-dataport.org/open-access/2018-ieee-grss-data-fusion-challenge-%E2%80%93-fusion-multispectral-lidar-and-hyperspectral-data)

[Berlin](https://github.com/danfenghong/ISPRS_S2FL)

[MUUFL](https://github.com/GatorSense/MUUFLGulfport/tree/master/MUUFLGulfportSceneLabels)
