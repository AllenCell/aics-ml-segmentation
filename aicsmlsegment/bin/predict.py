#!/usr/bin/env python

import sys
import argparse
import logging
import traceback
import os
import pathlib
import numpy as np

from skimage.morphology import remove_small_objects
from skimage.io import imsave
from aicsimageio import AICSImage
from scipy.ndimage import zoom

from aicsmlsegment.utils import load_config, load_single_image, input_normalization, image_normalization
from aicsmlsegment.utils import get_logger
from aicsmlsegment.model_utils import build_model, load_checkpoint, model_inference, apply_on_image

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    # declare the model
    model = build_model(config)

    # load the trained model instance
    model_path = config['model_path']
    print(f'Loading model from {model_path}...')
    load_checkpoint(model_path, model)

    # extract the parameters for running the model inference
    args_inference=lambda:None
    args_inference.size_in = config['size_in']
    args_inference.size_out = config['size_out']
    args_inference.OutputCh = config['OutputCh']
    args_inference.nclass =  config['nclass'] 
    if config['RuntimeAug'] <=0:
        args_inference.RuntimeAug = False
    else:
        args_inference.RuntimeAug = True

    # run
    inf_config = config['mode']
    if inf_config['name'] == 'file':
        fn = inf_config['InputFile']
        data_reader = AICSImage(fn)

        if inf_config['timelapse']:
            assert data_reader.shape[1] > 1, "not a timelapse, check you data"

            for tt in range(data_reader.shape[1]):
                # Assume:  dimensions = TCZYX
                img = data_reader.get_image_data("CZYX", S=0, T=tt, C=config['InputCh']).astype(float)
                img = image_normalization(img, config['Normalization'])

                if len(config['ResizeRatio'])>0:
                    img = zoom(img, (1, config['ResizeRatio'][0], config['ResizeRatio'][1], config['ResizeRatio'][2]), order=2, mode='reflect')
                    for ch_idx in range(img.shape[0]):
                        struct_img = img[ch_idx,:,:,:]
                        struct_img = (struct_img - struct_img.min())/(struct_img.max() - struct_img.min())
                        img[ch_idx,:,:,:] = struct_img

                # apply the model
                output_img = apply_on_image(model, img, model.final_activation, args_inference)

                # extract the result and write the output
                if len(config['OutputCh']) == 2:
                    out = output_img[0]
                    out = (out - out.min()) / (out.max()-out.min())
                    if len(config['ResizeRatio'])>0:
                        out = zoom(out, (1.0, 1/config['ResizeRatio'][0], 1/config['ResizeRatio'][1], 1/config['ResizeRatio'][2]), order=2, mode='reflect')
                    out = out.astype(np.float32)
                    if config['Threshold']>0:
                        out = out > config['Threshold']
                        out = out.astype(np.uint8)
                        out[out>0]=255
                    imsave(config['OutputDir'] + os.sep + pathlib.PurePosixPath(fn).stem + '_T_'+ f'{tt:03}' +'_struct_segmentation.tiff', out)
                else:
                    for ch_idx in range(len(config['OutputCh'])//2):
                        out = output_img[ch_idx]
                        out = (out - out.min()) / (out.max()-out.min())
                        if len(config['ResizeRatio'])>0:
                            out = zoom(out, (1.0, 1/config['ResizeRatio'][0], 1/config['ResizeRatio'][1], 1/config['ResizeRatio'][2]), order=2, mode='reflect')
                        out = out.astype(np.float32)
                        if config['Threshold']>0:
                            out = out > config['Threshold']
                            out = out.astype(np.uint8)
                            out[out>0]=255
                        imsave(config['OutputDir'] + os.sep + pathlib.PurePosixPath(fn).stem + '_T_'+ f'{tt:03}' +'_seg_'+ str(config['OutputCh'][2*ch_idx])+'.tiff',out)
        else:
            img = data_reader.get_image_data("CZYX", S=0, T=0, C=config['InputCh']).astype(float)
            img = image_normalization(img, config['Normalization'])

            if len(config['ResizeRatio'])>0:
                img = zoom(img, (1, config['ResizeRatio'][0], config['ResizeRatio'][1], config['ResizeRatio'][2]), order=2, mode='reflect')
                for ch_idx in range(img.shape[0]):
                    struct_img = img[ch_idx,:,:,:] # note that struct_img is only a view of img, so changes made on struct_img also affects img
                    struct_img = (struct_img - struct_img.min())/(struct_img.max() - struct_img.min())
                    img[ch_idx,:,:,:] = struct_img

            # apply the model
            output_img = apply_on_image(model, img, model.final_activation, args_inference)

            # extract the result and write the output
            if len(config['OutputCh']) == 2:
                out = output_img[0] 
                out = (out - out.min()) / (out.max()-out.min())
                if len(config['ResizeRatio'])>0:
                    out = zoom(out, (1.0, 1/config['ResizeRatio'][0], 1/config['ResizeRatio'][1], 1/config['ResizeRatio'][2]), order=2, mode='reflect')
                out = out.astype(np.float32)
                if config['Threshold']>0:
                    out = out > config['Threshold']
                    out = out.astype(np.uint8)
                    out[out>0]=255
                imsave(config['OutputDir'] + os.sep + pathlib.PurePosixPath(fn).stem +'_struct_segmentation.tiff', out)
            else:
                for ch_idx in range(len(config['OutputCh'])//2):
                    out = output_img[ch_idx] 
                    out = (out - out.min()) / (out.max()-out.min())
                    if len(config['ResizeRatio'])>0:
                        out = zoom(out, (1.0, 1/config['ResizeRatio'][0], 1/config['ResizeRatio'][1], 1/config['ResizeRatio'][2]), order=2, mode='reflect')
                    out = out.astype(np.float32)
                    if config['Threshold']>0:
                        out = out > config['Threshold']
                        out = out.astype(np.uint8)
                        out[out>0]=255
                    imsave(config['OutputDir'] + os.sep + pathlib.PurePosixPath(fn).stem +'_seg_'+ str(config['OutputCh'][2*ch_idx])+'.tiff', out)
            print(f'Image {fn} has been segmented')

    elif inf_config['name'] == 'folder':
        from glob import glob
        filenames = glob(inf_config['InputDir'] + '/*' + inf_config['DataType'])
        filenames.sort() #(reverse=True)
        print('files to be processed:')
        print(filenames)

        for _, fn in enumerate(filenames):

            # load data
            data_reader = AICSImage(fn)
            img = data_reader.get_image_data('CZYX', S=0, T=0, C=config['InputCh']).astype(float)
            if len(config['ResizeRatio'])>0:
                img = zoom(img, (1,config['ResizeRatio'][0], config['ResizeRatio'][1], config['ResizeRatio'][2]), order=2, mode='reflect')
            img = image_normalization(img, config['Normalization'])

            # apply the model
            output_img = apply_on_image(model, img, model.final_activation, args_inference)

            # extract the result and write the output
            if len(config['OutputCh'])==2:
                if config['Threshold']<0:
                    out = output_img[0]
                    out = (out - out.min()) / (out.max()-out.min())
                    if len(config['ResizeRatio'])>0:
                        out = zoom(out, (1.0, 1/config['ResizeRatio'][0], 1/config['ResizeRatio'][1], 1/config['ResizeRatio'][2]), order=2, mode='reflect')
                    out = out.astype(np.float32)
                    out = (out - out.min()) / (out.max()-out.min())
                else:
                    out = remove_small_objects(output_img[0] > config['Threshold'], min_size=2, connectivity=1) 
                    out = out.astype(np.uint8)
                    out[out>0]=255
                imsave(config['OutputDir'] + os.sep + pathlib.PurePosixPath(fn).stem + '_struct_segmentation.tiff', out)
            else:
                for ch_idx in range(len(config['OutputCh'])//2):
                    if config['Threshold']<0:
                        out = output_img[ch_idx]
                        out = (out - out.min()) / (out.max()-out.min())
                        out = out.astype(np.float32)
                    else:
                        out = output_img[ch_idx] > config['Threshold']
                        out = out.astype(np.uint8)
                        out[out>0]=255
                    imsave(config['OutputDir'] + os.sep + pathlib.PurePosixPath(fn).stem + '_seg_'+ str(config['OutputCh'][2*ch_idx])+'.ome.tif', out)
            
            print(f'Image {fn} has been segmented')

if __name__ == '__main__':
    
    main()