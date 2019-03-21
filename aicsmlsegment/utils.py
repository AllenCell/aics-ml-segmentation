import torch.nn as nn
import numpy as np
import logging
import sys
from torch.utils.data import sampler as torch_sampler
from aicsimageio import AICSImage
from aicsimageprocessing import resize
import os
from scipy import ndimage as ndi
from scipy import stats
import argparse

import torch
import yaml

def load_config(config_path):
    #parser = argparse.ArgumentParser(description='UNet3D training')
    #parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    #args = parser.parse_args()
    config = _load_config_yaml(config_path)
    # Get a device to train on
    device_name = config.get('device', 'cuda:0')
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    return config


def _load_config_yaml(config_file):
    return yaml.load(open(config_file, 'r'))

def get_samplers(num_training_data, validation_ratio, my_seed):

    indices = list(range(num_training_data))
    split = int(np.floor(validation_ratio * num_training_data))

    np.random.seed(my_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = torch_sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch_sampler.SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler

def input_normalization(img, args):

    #from aicsimageio import omeTifWriter
    #writer = omeTifWriter.OmeTifWriter('/allen/aics/assay-dev/Segmentation/DeepLearning/DNA_Labelfree_Evaluation/final_evaluation/test_before_norm.tiff')
    #writer.save(img[0,:,:,:])
    nchannel = img.shape[0]
    for ch_idx in range(nchannel):
        struct_img = img[ch_idx,:,:,:] # note that struct_img is only a view of img, so changes made on struct_img also affects img
        if args.Normalization == 0: # min-max normalization
            #struct_img = (struct_img - np.percentile(struct_img,0.1) + 1e-8)/(np.percentile(struct_img,99.9) - np.percentile(struct_img,0.1) + 1e-8)
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
            #img_valid = struct_img[np.logical_and(struct_img>300,struct_img<1000)]
            #m,s = stats.norm.fit(img_valid.flat)
            m,s = stats.norm.fit(struct_img.flat)
            strech_min = max(m - 2.5*s, struct_img.min())
            strech_max = min(m + 10 *s, struct_img.max())
            struct_img[struct_img>strech_max]=strech_max
            struct_img[struct_img<strech_min]=strech_min
            struct_img = (struct_img- strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
        elif args.Normalization == 3: # lamin high
            m,s = stats.norm.fit(struct_img.flat)
            strech_min = np.percentile(struct_img,0.005)
            strech_max = min(m + 8 *s, struct_img.max())
            print(m)
            print(s)
            print(strech_min)
            print(strech_max)
            struct_img[struct_img>strech_max]=strech_max
            struct_img[struct_img<strech_min]=strech_min
            struct_img = (struct_img - strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
        elif args.Normalization == 4: # lamin low
            m,s = stats.norm.fit(struct_img.flat)
            strech_min = max(m - 1*s, struct_img.min())
            strech_max = min(m + 10 * s, struct_img.max())
            print(m)
            print(s)
            print(strech_min)
            print(strech_max)
            struct_img[struct_img>strech_max]=strech_max
            struct_img[struct_img<strech_min]=strech_min
            struct_img = (struct_img - strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
        elif args.Normalization == 5: # light
            m,s = stats.norm.fit(struct_img.flat)
            strech_min = max(m - 4*s, struct_img.min())
            strech_max = min(m + 4 *s, struct_img.max())
            struct_img[struct_img>strech_max]=strech_max
            struct_img[struct_img<strech_min]=strech_min
            struct_img = (struct_img- strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
        elif args.Normalization == 6: # dna-fast
            m,s = stats.norm.fit(struct_img.flat)
            strech_min = max(m - 2*s, struct_img.min())
            strech_max = min(m + 8 *s, struct_img.max()) # used 6 before transfer learing, use 8 after
            print(strech_max)
            print(struct_img.max())
            print(strech_min)
            print(struct_img.min())
            #strech_min = struct_img.min()
            #strech_max = struct_img.max()
            struct_img[struct_img>strech_max]=strech_max
            struct_img[struct_img<strech_min]=strech_min
            struct_img = (struct_img- strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
            #writer = io.omeTifWriter.OmeTifWriter('test.ome.tiff')
            #writer.save(struct_img)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
        elif args.Normalization == 7: # cardio_wga
            m,s = stats.norm.fit(struct_img.flat)
            strech_min = max(m - 1*s, struct_img.min())
            strech_max = min(m + 6 *s, struct_img.max()) # used 6 before transfer learing, use 8 after
            struct_img[struct_img>strech_max]=strech_max
            struct_img[struct_img<strech_min]=strech_min
            struct_img = (struct_img- strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
        elif args.Normalization == 8: # FBL hipsc
            m,s = stats.norm.fit(struct_img.flat)
            strech_min = struct_img.min()
            strech_max = min(m + 20 *s, struct_img.max()) # used 6 before transfer learing, use 8 after
            struct_img[struct_img>strech_max]=strech_max
            struct_img[struct_img<strech_min]=strech_min
            struct_img = (struct_img- strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
        elif args.Normalization == 9: # nuc
            img_valid = struct_img[struct_img>10000]
            m,s = stats.norm.fit(img_valid.flat)
            m,s = stats.norm.fit(struct_img.flat)
            strech_min = m - 3.5*s
            strech_max = m + 3.5 *s
            struct_img[struct_img>strech_max]=strech_max
            struct_img[struct_img<strech_min]=strech_min
            struct_img = (struct_img- strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
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
            
            from scipy import ndimage as ndi
            struct_img_smooth = ndi.gaussian_filter(struct_img, sigma=50, mode='nearest', truncate=3.0)
            struct_img_smooth_sub = struct_img - struct_img_smooth
            struct_img = (struct_img_smooth_sub - struct_img_smooth_sub.min())/(struct_img_smooth_sub.max()-struct_img_smooth_sub.min())
            
            m,s = stats.norm.fit(struct_img.flat)
            strech_min = max(m - 2.5*s, struct_img.min())
            strech_max = min(m + 10 *s, struct_img.max())
            struct_img[struct_img>strech_max]=strech_max
            struct_img[struct_img<strech_min]=strech_min
            struct_img = (struct_img- strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
            img[ch_idx,:,:,:] = struct_img[:,:,:]
            print('subtracted background')
    
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
