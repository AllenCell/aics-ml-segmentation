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
        torch.nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()

def apply_on_image(model, input_img, softmax, args):
    print('here')
    if args.RuntimeAug == 'None':
        return model_inference(model, input_img, softmax, args)
    else:
        from PIL import Image
        print('doing runtime augmentation')
        import pathlib
        from aicsimageio import omeTifWriter

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

    # zero padding on input image
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

                tmp_out = model(Variable(input_img_tensor.cuda(), volatile=True).unsqueeze(0))

                
                assert len(args.OutputCh)//2 <= len(tmp_out)

                if args.model == 'cascade':
                    label = tmp_out[0]
                elif args.model == 'DSU' or args.model == 'HDSU':
                    label = tmp_out[1]
                else:
                    label = tmp_out

                for ch_idx in range(len(args.OutputCh)//2):
                    label = tmp_out[args.OutputCh[ch_idx*2]]
                    prob = softmax(label)

                    out_flat_tensor = prob.cpu().data.float()
                    out_tensor = out_flat_tensor.view(args.size_out[0], args.size_out[1], args.size_out[2], args.nclass[ch_idx])
                    out_nda = out_tensor.numpy()

                    output_img[ch_idx][0,za:(za+args.size_out[0]), ya:(ya+args.size_out[1]), xa:(xa+args.size_out[2])] = out_nda[:,:,:,args.OutputCh[ch_idx*2+1]]

    return output_img

def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.
    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.
    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied
    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])

    return state


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def find_maximum_patch_size(model, device):
    """Tries to find the biggest patch size that can be send to GPU for inference
    without throwing CUDA out of memory"""
    logger = get_logger('PatchFinder')
    in_channels = model.in_channels

    patch_shapes = [(64, 128, 128), (96, 128, 128),
                    (64, 160, 160), (96, 160, 160),
                    (64, 192, 192), (96, 192, 192)]

    for shape in patch_shapes:
        # generate random patch of a given size
        patch = np.random.randn(*shape).astype('float32')

        patch = torch \
            .from_numpy(patch) \
            .view((1, in_channels) + patch.shape) \
            .to(device)

        logger.info(f"Current patch size: {shape}")
        model(patch)


def unpad(probs, index, shape, pad_width=8):
    def _new_slices(slicing, max_size):
        if slicing.start == 0:
            p_start = 0
            i_start = 0
        else:
            p_start = pad_width
            i_start = slicing.start + pad_width

        if slicing.stop == max_size:
            p_stop = None
            i_stop = max_size
        else:
            p_stop = -pad_width
            i_stop = slicing.stop - pad_width

        return slice(p_start, p_stop), slice(i_start, i_stop)

    D, H, W = shape

    i_c, i_z, i_y, i_x = index
    p_c = slice(0, probs.shape[0])

    p_z, i_z = _new_slices(i_z, D)
    p_y, i_y = _new_slices(i_y, H)
    p_x, i_x = _new_slices(i_x, W)

    probs_index = (p_c, p_z, p_y, p_x)
    index = (i_c, i_z, i_y, i_x)
    return probs[probs_index], index
    
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
    
    model = model.to(config['device'])
    return model