#!/usr/bin/env python
# coding: utf-8

# training arguments are defined in this file


class Config():
    save_ckpt_dir = ''
    result_out_dir = ''
    data_dir = ''
    use_gpu = True
    num_replicates = 5
    seed = 42
    dataset = 'muufl'
    
    if dataset == 'houston':
        model = 'unet'
        mask_undefined = True
        num_classes = 20
        epochs = 300
        lr = 0.001
        lr_schedule = None
        optimizer = 'adam'
        batch_size = 12
        use_init = False
        sample_h = sample_w = 128
    elif dataset == 'berlin':
        model = 'resnet18'
        mask_undefined = False
        num_classes = 8
        epochs = 300
        lr = 0.001
        lr_schedule = None
        optimizer = 'sgd'
        batch_size = 64
        use_init = False
        sample_radius = 8
    elif dataset == 'muufl':
        model = 'resnet18'
        mask_undefined = False
        num_classes = 11
        epochs = 300
        lr = 0.02
        lr_schedule = [200, 240]
        optimizer = 'sgd'
        batch_size = 48
        use_init = True
        sample_radius = 5
    

