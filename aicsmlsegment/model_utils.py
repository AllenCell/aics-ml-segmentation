import numpy as np
import torch
from torch.autograd import Variable
import time
import logging
import os
import shutil
import sys

from aicsmlsegment.utils import get_logger

SUPPORTED_MODELS = ['unet_xy_zoom', 'unet_xy']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

def apply_on_image(model, input_img, softmax, args):

    if not args.RuntimeAug:
        return model_inference(model, input_img, softmax, args)
    else:
        from PIL import Image
        print('doing runtime augmentation')
        #import pathlib
        #from aicsimageio import omeTifWriter

        input_img_aug = input_img.copy()
        for ch_idx in range(input_img_aug.shape[0]):
            str_im = input_img_aug[ch_idx,:,:,:]
            input_img_aug[ch_idx,:,:,:] = np.flip(str_im, axis=2)
            
        out1 = model_inference(model, input_img_aug, softmax, args)

        '''
        for ch_idx in range(len(args.OutputCh)//2):
            writer = omeTifWriter.OmeTifWriter(args.OutputDir + 'test_seg_aug1_'+ str(args.OutputCh[2*ch_idx])+'.ome.tif')
            if args.Threshold<0:
                writer.save(out1[ch_idx].astype(float))
            else:
                out = out1[ch_idx] > args.Threshold
                out = out.astype(np.uint8)
                out[out>0]=255
                writer.save(out)
        '''

        input_img_aug = []
        input_img_aug = input_img.copy()
        for ch_idx in range(input_img_aug.shape[0]):
            str_im = input_img_aug[ch_idx,:,:,:]
            input_img_aug[ch_idx,:,:,:] = np.flip(str_im, axis=1)
        
        out2 = model_inference(model, input_img_aug, softmax, args)

        '''
        for ch_idx in range(len(args.OutputCh)//2):
            writer = omeTifWriter.OmeTifWriter(args.OutputDir + 'test_seg_aug2_'+ str(args.OutputCh[2*ch_idx])+'.ome.tif')
            if args.Threshold<0:
                writer.save(out2[ch_idx].astype(float))
            else:
                out = out2[ch_idx] > args.Threshold
                out = out.astype(np.uint8)
                out[out>0]=255
                writer.save(out)
        '''

        input_img_aug = []
        input_img_aug = input_img.copy()
        for ch_idx in range(input_img_aug.shape[0]):
            str_im = input_img_aug[ch_idx,:,:,:]
            input_img_aug[ch_idx,:,:,:] = np.flip(str_im, axis=0)
        
        out3 = model_inference(model, input_img_aug, softmax, args)

        '''
        for ch_idx in range(len(args.OutputCh)//2):
            writer = omeTifWriter.OmeTifWriter(args.OutputDir + 'test_seg_aug3_'+ str(args.OutputCh[2*ch_idx])+'.ome.tif')
            if args.Threshold<0:
                writer.save(out3[ch_idx].astype(float))
            else:
                out = out3[ch_idx] > args.Threshold
                out = out.astype(np.uint8)
                out[out>0]=255
                writer.save(out)
        '''

        out0 = model_inference(model, input_img, softmax, args)        

        for ch_idx in range(len(out0)):
            out0[ch_idx] = 0.25*(out0[ch_idx] + np.flip(out1[ch_idx], axis=3) + np.flip(out2[ch_idx], axis=2) + np.flip(out3[ch_idx], axis=1))
        
        return out0

def model_inference(model, input_img, softmax, args):

    model.eval()

    if args.size_in == args.size_out:
        img_pad = np.np.expand_dims(input_img, axis=0) # add batch dimension
    else:  # zero padding on input image
        padding = [(x-y)//2 for x,y in zip(args.size_in, args.size_out)]
        img_pad0 = np.pad(input_img, ((0,0),(0,0),(padding[1],padding[1]),(padding[2],padding[2])), 'symmetric')#'constant')
        img_pad = np.pad(img_pad0, ((0,0),(padding[0],padding[0]),(0,0),(0,0)), 'constant')

    output_img = []
    for ch_idx in range(len(args.OutputCh)//2):
        output_img.append(np.zeros(input_img.shape))

    # loop through the image patch by patch
    num_step_z = int(np.floor(input_img.shape[1]/args.size_out[0])+1)
    num_step_y = int(np.floor(input_img.shape[2]/args.size_out[1])+1)
    num_step_x = int(np.floor(input_img.shape[3]/args.size_out[2])+1)

    with torch.no_grad():
        for ix in range(num_step_x):
            if ix<num_step_x-1:
                xa = ix * args.size_out[2]
            else:
                xa = input_img.shape[3] - args.size_out[2]

            for iy in range(num_step_y):
                if iy<num_step_y-1:
                    ya = iy * args.size_out[1]
                else:
                    ya = input_img.shape[2] - args.size_out[1]

                for iz in range(num_step_z):
                    if iz<num_step_z-1:
                        za = iz * args.size_out[0]
                    else:
                        za = input_img.shape[1] - args.size_out[0]

                    # build the input patch
                    input_patch = img_pad[ : ,za:(za+args.size_in[0]) ,ya:(ya+args.size_in[1]) ,xa:(xa+args.size_in[2])]
                    input_img_tensor = torch.from_numpy(input_patch.astype(float)).float()

                    tmp_out = model(Variable(input_img_tensor.cuda()).unsqueeze(0))
                    assert len(args.OutputCh)//2 <= len(tmp_out), print('the parameter OutputCh is not compatible with the number of output tensors')

                    for ch_idx in range(len(args.OutputCh)//2):
                        label = tmp_out[args.OutputCh[ch_idx*2]]
                        prob = softmax(label)

                        out_flat_tensor = prob.cpu().data.float()
                        out_tensor = out_flat_tensor.view(args.size_out[0], args.size_out[1], args.size_out[2], args.nclass[ch_idx])
                        out_nda = out_tensor.numpy()

                        output_img[ch_idx][0,za:(za+args.size_out[0]), ya:(ya+args.size_out[1]), xa:(xa+args.size_out[2])] = out_nda[:,:,:,args.OutputCh[ch_idx*2+1]]

    return output_img

def save_checkpoint(state, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.
    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)
    
    file_path = checkpoint_dir + os.sep + 'checkpoint_epoch_' + str(state['epoch']) + '.pytorch'
    log_info(f"Saving checkpoint at epoch {state['epoch']} to '{file_path}'")
    torch.save(state, file_path)


def load_checkpoint(checkpoint_path, model):
    """Loads model from a given checkpoint_path
    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    #device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)

    #TODO: add an option to load training status

    return state


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def build_model(config):

    assert 'model' in config, 'Could not find model configuration'
    model_config = config['model']
    name = model_config['name']
    assert name in SUPPORTED_MODELS, f'Invalid model: {name}. Supported models: {SUPPORTED_MODELS}'

    if name == 'unet_xy':
        from aicsmlsegment.Net3D.unet_xy import UNet3D as DNN
        model = DNN(config['nchannel'], config['nclass'])
    elif name =='unet_xy_zoom':
        from aicsmlsegment.Net3D.unet_xy_enlarge import UNet3D as DNN
        model = DNN(config['nchannel'], config['nclass'], model_config.get('zoom_ratio',3))
    
    model = model.apply(weights_init)
    print('model initialization succeeds !')
    model = model.to(config['device'])
    return model