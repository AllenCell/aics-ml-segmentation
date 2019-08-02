import numpy as np
import logging
import sys
from aicsimageio import AICSImage
from aicsimageprocessing import resize
import os
from scipy import ndimage as ndi
from scipy import stats
import argparse

import yaml

def load_config(config_path):
    import torch
    config = _load_config_yaml(config_path)
    # Get a device to train on
    device_name = config.get('device', 'cuda:0')
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    return config


def _load_config_yaml(config_file):
    return yaml.load(open(config_file, 'r'))

def get_samplers(num_training_data, validation_ratio, my_seed):
    from torch.utils.data import sampler as torch_sampler
    indices = list(range(num_training_data))
    split = int(np.floor(validation_ratio * num_training_data))

    np.random.seed(my_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = torch_sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch_sampler.SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler

def simple_norm(img, a, b, m_high=-1, m_low=-1):
    idx = np.ones(img.shape, dtype=bool)
    if m_high>0:
        idx = np.logical_and(idx, img<m_high)
    if m_low>0:
        idx = np.logical_and(idx, img>m_low)
    img_valid = img[idx]
    m,s = stats.norm.fit(img_valid.flat)
    strech_min = max(m - a*s, img.min())
    strech_max = min(m + b*s, img.max())
    img[img>strech_max]=strech_max
    img[img<strech_min]=strech_min
    img = (img- strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
    return img

def background_sub(img, r):
    struct_img_smooth = ndi.gaussian_filter(img, sigma=r, mode='nearest', truncate=3.0)
    struct_img_smooth_sub = img - struct_img_smooth
    struct_img = (struct_img_smooth_sub - struct_img_smooth_sub.min())/(struct_img_smooth_sub.max()-struct_img_smooth_sub.min())
    return struct_img

def input_normalization(img, args):

    #from aicsimageio import omeTifWriter
    #writer = omeTifWriter.OmeTifWriter('/allen/aics/assay-dev/Segmentation/DeepLearning/DNA_Labelfree_Evaluation/final_evaluation/test_before_norm.tiff')
    #writer.save(img[0,:,:,:])
    nchannel = img.shape[0]
    args.Normalization = int(args.Normalization)
    for ch_idx in range(nchannel):
        struct_img = img[ch_idx,:,:,:] # note that struct_img is only a view of img, so changes made on struct_img also affects img
        if args.Normalization == 0: # min-max normalization
            struct_img = (struct_img - struct_img.min() + 1e-8)/(struct_img.max() - struct_img.min() + 1e-7)
        elif args.Normalization == 1: # mem: DO NOT CHANGE (FIXED FOR CAAX PRODUCTION)
            m,s = stats.norm.fit(struct_img.flat)
            strech_min = max(m - 2*s, struct_img.min())
            strech_max = min(m + 11 *s, struct_img.max())
            struct_img[struct_img>strech_max]=strech_max
            struct_img[struct_img<strech_min]=strech_min
            struct_img = (struct_img- strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
        elif args.Normalization == 2: # nuc
            #struct_img = simple_norm(struct_img, 2.5, 10, 1000, 300)
            struct_img = simple_norm(struct_img, 2.5, 10)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
        elif args.Normalization == 4:
            struct_img = simple_norm(struct_img, 1, 15)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
        elif args.Normalization == 7: # cardio_wga
            struct_img = simple_norm(struct_img, 1, 6)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
        elif args.Normalization == 10: # lamin hipsc, DO NOT CHANGE (FIXED FOR LAMNB1 PRODUCTION)
            img_valid = struct_img[struct_img>4000]
            m,s = stats.norm.fit(img_valid.flat)
            m,s = stats.norm.fit(struct_img.flat)
            strech_min = struct_img.min()
            strech_max = min(m + 25 *s, struct_img.max())
            struct_img[struct_img>strech_max]=strech_max
            struct_img = (struct_img- strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
        elif args.Normalization == 12: # nuc
            struct_img = background_sub(struct_img,50)
            struct_img = simple_norm(struct_img, 2.5, 10)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
            print('subtracted background')
        elif args.Normalization == 11: 
            struct_img = background_sub(struct_img,50)
            #struct_img = simple_norm(struct_img, 2.5, 10)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
        elif args.Normalization == 13: # cellmask
            struct_img[struct_img>10000] = struct_img.min()
            struct_img = background_sub(struct_img,50)
            struct_img = simple_norm(struct_img, 2, 11)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
        elif args.Normalization == 14:
            struct_img = simple_norm(struct_img, 1, 10)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
        elif args.Normalization == 15: # lamin
            struct_img[struct_img>4000] = struct_img.min()
            struct_img = background_sub(struct_img,50)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
        elif args.Normalization == 16: # lamin/h2b
            struct_img = background_sub(struct_img,50)
            struct_img = simple_norm(struct_img, 1.5, 6)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
        elif args.Normalization == 17: # lamin
            struct_img = background_sub(struct_img,50)
            struct_img = simple_norm(struct_img, 1, 10)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
        elif args.Normalization == 18: # h2b
            struct_img = background_sub(struct_img,50)
            struct_img = simple_norm(struct_img, 1.5, 10)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
        else:
            print('no normalization recipe found')
            quit()
    return img

def image_normalization(img, config):

    if type(config) is dict:
        ops = config['ops']
        nchannel = img.shape[0]
        assert len(ops) == nchannel
        for ch_idx in range(nchannel):
            ch_ops = ops[ch_idx]['ch']
            struct_img = img[ch_idx,:,:,:]
            for transform in ch_ops:
                if transform['name'] == 'background_sub':
                    struct_img = background_sub(struct_img, transform['sigma'])
                elif transform['name'] =='auto_contrast':
                    param = transform['param']
                    if len(param)==2:
                        struct_img = simple_norm(struct_img, param[0], param[1])
                    elif len(param)==4:
                        struct_img = simple_norm(struct_img, param[0], param[1], param[2], param[3])
                    else: 
                        print('bad paramter for auto contrast')
                        quit()
                else: 
                    print(transform['name'])
                    print('other normalization methods are not supported yet')
                    quit()
                
                img[ch_idx,:,:,:] = struct_img[:,:,:]
    else:
        args_norm = lambda:None
        args_norm.Normalization = config

        img = input_normalization(img, args_norm)

    return img

def load_single_image(args, fn, time_flag=False):

    if time_flag:
        img = fn[:,args.InputCh,:,:]
        img = img.astype(float)
        img = np.transpose(img, axes=(1,0,2,3))
    else:
        data_reader = AICSImage(fn)
        img = data_reader.data  #TCZYX
        if img.shape[0]==1:
            img = np.squeeze(img,axis=0)
        elif img.shape[1]==1:
            img = np.squeeze(img,axis=1)
        else:
            print('error in data dimension')
            print(img.shape)
            quit()
        img = img.astype(float)
        if img.shape[1] < img.shape[0]:
                img = np.transpose(img,(1,0,2,3))
        img = img[args.InputCh,:,:,:] #  fancy indexing atually creates a copy, not a view

    # normalization
    if args.mode == 'train':
        for ch_idx in range(args.nchannel):
            struct_img = img[ch_idx,:,:,:] # note that struct_img is only a view of img, so changes made on struct_img also affects img
            struct_img = (struct_img - struct_img.min() )/(struct_img.max() - struct_img.min())
    elif not args.Normalization == 0:
        img = input_normalization(img, args)
    
    # rescale
    if len(args.ResizeRatio)>0:
        img = resize(img, (1, args.ResizeRatio[0], args.ResizeRatio[1], args.ResizeRatio[2]), method='cubic')
        for ch_idx in range(img.shape[0]):
            struct_img = img[ch_idx,:,:,:] # note that struct_img is only a view of img, so changes made on struct_img also affects img
            struct_img = (struct_img - struct_img.min())/(struct_img.max() - struct_img.min())
            img[ch_idx,:,:,:] = struct_img

    return img


def compute_iou(prediction, gt, cmap):

    area_i = np.logical_and(prediction, gt)
    area_i[cmap==0]=False
    area_u = np.logical_or(prediction, gt)
    area_u[cmap==0]=False

    return np.count_nonzero(area_i) / np.count_nonzero(area_u)

def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
