import numpy as np
import torch
from torch.autograd import Variable
import time

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        torch.nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()

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
        print('.')
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
