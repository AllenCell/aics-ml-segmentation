import sys
import argparse
import logging
import traceback
import pathlib
import numpy as np

from aicsimageio import AICSImage, omeTifWriter
from aicsimageprocessing import resize

from aicsmlsegment.utils import load_config, load_single_image, input_normalization
from aicsmlsegment.utils import get_logger
from aicsmlsegment.model_utils import build_model, load_checkpoint, model_inference

def main(args):

    config = load_config(args.config)

    # declare the model
    model = build_model(config)

    # load the trained model instance
    model_path = config['model_path']
    print(f'Loading model from {model_path}...')
    load_checkpoint(model_path, model)

    # extract the parameters for preparing the input image
    args_norm = lambda:None
    args_norm.Normalization = config['Normalization']

    # extract the parameters for running the model inference
    args_inference=lambda:None
    args_inference.size_in = config['size_in']
    args_inference.size_out = config['size_out']
    args_inference.OutputCh = config['OutputCh']
    args_inference.nclass =  config['nclass'] 

    # run
    inf_config = config['mode']
    if inf_config['name'] == 'file':
        fn = inf_config['InputFile']
        data_reader = AICSImage(fn)
        img0 = data_reader.data

        if inf_config['timelapse']:
            assert img0.shape[0]>1

            for tt in range(img0.shape[0]):
                # Assume:  dimensions = TCZYX
                img = img0[tt, config['InputCh'],:,:,:].astype(float)
                img = input_normalization(img, args_norm)

                if len(config['ResizeRatio'])>0:
                    img = resize(img, (1, config['ResizeRatio'][0], config['ResizeRatio'][1], config['ResizeRatio'][2]), method='cubic')
                    for ch_idx in range(img.shape[0]):
                        struct_img = img[ch_idx,:,:,:]
                        struct_img = (struct_img - struct_img.min())/(struct_img.max() - struct_img.min())
                        img[ch_idx,:,:,:] = struct_img

                # apply the model
                output_img = model_inference(model, img, model.final_activation, args_inference)

                # extract the result and write the output
                for ch_idx in range(len(config['OutputCh'].OutputCh)//2):
                    writer = omeTifWriter.OmeTifWriter(config['OutputDir'] + pathlib.PurePosixPath(fn).stem + '_T_'+ f'{tt:03}' +'_seg_'+ str(config['OutputCh'][2*ch_idx])+'.tiff')
                    out = output_img[ch_idx]
                    if len(config['ResizeRatio'])>0:
                        out = resize(out, (1.0, 1/config['ResizeRatio'][0], 1/config['ResizeRatio'][1], 1/config['ResizeRatio'][2]), method='cubic')
                    if config['Threshold']>0:
                        out = out > config['Threshold']
                        out = out.astype(np.uint8)
                        out[out>0]=255
                    writer.save(out)
        else:
            img = img0[0,:,:,:,:].astype(float)
            print(f'processing one image of size {img.shape}')
            if img.shape[1] < img.shape[0]:
                img = np.transpose(img,(1,0,2,3))
            img = img[config['InputCh'],:,:,:]
            img = input_normalization(img, args_norm)

            if len(config['ResizeRatio'])>0:
                img = resize(img, (1, config['ResizeRatio'][0], config['ResizeRatio'][1], config['ResizeRatio'][2]), method='cubic')
                for ch_idx in range(img.shape[0]):
                    struct_img = img[ch_idx,:,:,:] # note that struct_img is only a view of img, so changes made on struct_img also affects img
                    struct_img = (struct_img - struct_img.min())/(struct_img.max() - struct_img.min())
                    img[ch_idx,:,:,:] = struct_img

            # apply the model
            output_img = model_inference(model, img, model.final_activation, args_inference)

            # extract the result and write the output
            for ch_idx in range(len(config['OutputCh'])//2):
                out = output_img[ch_idx] 
                if len(config['ResizeRatio'])>0:
                    out = resize(out, (1.0, 1/config['ResizeRatio'][0], 1/config['ResizeRatio'][1], 1/config['ResizeRatio'][2]), method='cubic')
                if config['Threshold']>0:
                    out = out > config['Threshold']
                    out = out.astype(np.uint8)
                    out[out>0]=255
                writer = omeTifWriter.OmeTifWriter(config['OutputDir'] + pathlib.PurePosixPath(fn).stem +'_seg_'+ str(config['OutputCh'][2*ch_idx])+'.tiff')
                writer.save(out)
            print(f'Image {fn} has been segmented')

    elif inf_config['name'] == 'folder':
        from glob import glob
        filenames = glob(inf_config['InputDir'] + '/*' + inf_config['DataType'])
        filenames.sort()
        #print(filenames)

        for _, fn in enumerate(filenames):

            # load data
            data_reader = AICSImage(fn)
            img0 = data_reader.data
            img = img0[0,:,:,:,:].astype(float)
            if img.shape[1] < img.shape[0]:
                img = np.transpose(img,(1,0,2,3))
            img = img[config['InputCh'],:,:,:]
            img = input_normalization(img, args_norm)


            writer = omeTifWriter.OmeTifWriter(config['OutputDir'] + pathlib.PurePosixPath(fn).stem + '_seg_input.ome.tif')
            writer.save(img)

            if len(config['ResizeRatio'])>0:
                img = resize(img, (1, config['ResizeRatio'][0], config['ResizeRatio'][1], config['ResizeRatio'][2]), method='cubic')
                for ch_idx in range(img.shape[0]):
                    struct_img = img[ch_idx,:,:,:] # note that struct_img is only a view of img, so changes made on struct_img also affects img
                    struct_img = (struct_img - struct_img.min())/(struct_img.max() - struct_img.min())
                    img[ch_idx,:,:,:] = struct_img

            writer = omeTifWriter.OmeTifWriter(config['OutputDir'] + pathlib.PurePosixPath(fn).stem + '_seg_input_2.ome.tif')
            writer.save(img)

            # apply the model
            output_img = model_inference(model, img, model.final_activation, args_inference)

            # extract the result and write the output
            for ch_idx in range(len(config['OutputCh'])//2):
                writer = omeTifWriter.OmeTifWriter(config['OutputDir'] + pathlib.PurePosixPath(fn).stem + '_seg_'+ str(config['OutputCh'][2*ch_idx])+'.ome.tif')
                if config['Threshold']<0:
                    writer.save(output_img[ch_idx].astype(float))
                else:
                    out = output_img[ch_idx] > config['Threshold']
                    out = out.astype(np.uint8)
                    out[out>0]=255
                    writer.save(out)
            
            print(f'Image {fn} has been segmented')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    main(parser.parse_args())