####### TEMPLATE FILE for experiments ########
# Search "Template" and replace with specifc name for experiments

import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import random
import pandas as pd
import logging

# io libary
import os
from aicsimageio import AICSImage, omeTifWriter
import glob
import pathlib

# import required Torch library
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from torch import from_numpy

# import utils
from aicsmlsegment.utils import get_samplers, load_single_image, compute_iou, input_normalization
from aicsmlsegment.model_utils import weights_init, model_inference, apply_on_image
from aicsimageprocessing import resize

## initialize logging
#log = logging.getLogger()
#logging.basicConfig(level=logging.INFO,
#                    format='[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s')



def train(args, model):

    model.train()

    # check logger
    if not args.TestMode  and os.path.isfile(args.LoggerName):
        print('logger file exists')
        quit()
    text_file = open(args.LoggerName, 'a')
    print(f'Epoch,Training_Loss,Validation_Loss\n',file=text_file)
    text_file.close()

    # load the correct loss function
    if args.Loss == 'NLL_CM' and args.model == 'unet_2task':
        from aicsmlsegment.custom_loss import MultiTaskElementNLLLoss 
        criterion = MultiTaskElementNLLLoss(args.LossWeight, args.nclass)
        print('use 2 task elementwise NLL loss')
    elif args.Loss == 'NLL_CM' and (args.model == 'unet_ds' or args.model == 'unet_xy' \
        or args.model == 'unet_deeper_xy' or args.model == 'unet_xy_d6' \
        or args.model == 'unet_xy_p3' or args.model == 'unet_xy_p2'):
        from aicsmlsegment.custom_loss import MultiAuxillaryElementNLLLoss
        criterion = MultiAuxillaryElementNLLLoss(3,args.LossWeight, args.nclass)
        print('use unet with deep supervision loss')
    elif args.Loss == 'NLL_CM' and args.model == 'unet_xy_multi_task':
        from aicsmlsegment.custom_loss import MultiTaskElementNLLLoss 
        criterion = MultiTaskElementNLLLoss(args.LossWeight, args.nclass)
        print('use 2 task elementwise NLL loss')
            
    # prepare the training/validattion filenames
    print('prepare the data ... ...')
    filenames = glob.glob(args.DataPath + '/*_GT.ome.tif')
    filenames.sort()
    total_num = len(filenames)
    if len(args.LeaveOut)==1:
        if args.LeaveOut[0]>0 and args.LeaveOut[0]<1:
            num_train = int(np.floor((1-args.LeaveOut[0]) * total_num))
            shuffled_idx = np.arange(total_num)
            random.shuffle(shuffled_idx)
            train_idx = shuffled_idx[:num_train]
            valid_idx = shuffled_idx[num_train:]
        else:
            valid_idx = [int(args.LeaveOut[0])]
            train_idx = list(set(range(total_num)) - set(map(int, args.LeaveOut)))  
    elif args.LeaveOut:
        valid_idx = list(map(int, args.LeaveOut))  
        train_idx = list(set(range(total_num)) - set(valid_idx))
    
    valid_filenames = []
    train_filenames = []
    for fi, fn in enumerate(valid_idx):
        valid_filenames.append(filenames[fn][:-11])
    for fi, fn in enumerate(train_idx):
        train_filenames.append(filenames[fn][:-11])

    # may need a different validation method
    #validation_set_loader = DataLoader(exp_Loader(validation_filenames), num_workers=1, batch_size=1, shuffle=False)

    if args.Augmentation == 'NOAUG_M':
        from aicsmlsegment.DataLoader3D.Universal_Loader import NOAUG_M as train_loader
        print('use no augmentation, with cost map')
    elif args.Augmentation == 'RR_FH_M':
        from aicsmlsegment.DataLoader3D.Universal_Loader import RR_FH_M as train_loader
        print('use flip + rotation augmentation, with cost map')
    elif args.Augmentation == 'RR_FH_M0':
        from aicsmlsegment.DataLoader3D.Universal_Loader import RR_FH_M0 as train_loader
        print('use flip + rotation augmentation, with cost map')
    elif args.Augmentation == 'RR_FH_M0C':
        from aicsmlsegment.DataLoader3D.Universal_Loader import RR_FH_M0C as train_loader
        print('use flip + rotation augmentation, with cost map, and also count valid pixels')

    # softmax for validation
    softmax = nn.Softmax(dim=1)
    softmax.cuda()

    for epoch in range(args.NumEpochs+1):

        if epoch % args.EpochPerBuffer==0:
            print('shuffling training data ... ...')
            random.shuffle(train_filenames)
            train_set_loader = DataLoader(train_loader(train_filenames, args.PatchPerBuffer, args.size_in, args.size_out), num_workers=args.NumWorkers, batch_size=args.BatchSize, shuffle=True)
            print('training data is ready')
        
        # specific optimizer for this epoch
        optimizer = None
        if len(args.lr)==1: # single value
            optimizer = optim.Adam(model.parameters(),lr = args.lr[0], weight_decay = args.WeightDecay)
        elif len(args.lr)>1: # [stage_1, lr_1, stage_2, lr_2, ..., stage_k, lr_k, lr_final]
            assert len(args.lr)%2==1
            num_training_stage = (len(args.lr)-1)//2
            elsecase=True
            for ts in range(num_training_stage):
                if epoch<args.lr[ts*2]:
                    optimizer = optim.Adam(model.parameters(),lr = args.lr[ts*2+1], weight_decay=args.WeightDecay)
                    elsecase=False
                    break
            if elsecase:
                optimizer = optim.Adam(model.parameters(),lr = args.lr[-1], weight_decay=args.WeightDecay)
        assert optimizer is not None, f'optimzer setup fails'

        # re-open the logger file
        text_file = open(args.LoggerName, 'a')

        # Training starts ...
        epoch_loss = []
        model.train()

        for step, current_batch in tqdm(enumerate(train_set_loader)):
                
                inputs = Variable(current_batch[0].cuda())
                targets = current_batch[1]
                #print(inputs.size())
                #print(targets[0].size())
                outputs = model(inputs)
                #print(len(outputs))
                #print(outputs[0].size())

                if len(targets)>1:
                    for zidx in range(len(targets)):
                        targets[zidx] = Variable(targets[zidx].cuda())
                else: 
                    targets = Variable(targets[0].cuda())

                optimizer.zero_grad()
                if len(current_batch)==3: # input + target + cmap
                    cmap = Variable(current_batch[2].cuda())
                    loss = criterion(outputs, targets, cmap)
                else: # input + target
                    loss = criterion(outputs,targets)
                    
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.data.item())
        
        # Validation starts ...
        validation_loss = np.zeros((len(args.OutputCh)//2,))
        model.eval() 

        for img_idx, fn in enumerate(valid_filenames):

            # target 
            label_reader = AICSImage(fn+'_GT.ome.tif')  #CZYX
            label = label_reader.data
            label = np.squeeze(label,axis=0) # 4-D after squeeze

            # when the tif has only 1 channel, the loaded array may have falsely swaped dimensions (ZCYX). we want CZYX
            # (This may also happen in different OS or different package versions)
            # ASSUMPTION: we have more z slices than the number of channels 
            if label.shape[1]<label.shape[0]: 
                label = np.transpose(label,(1,0,2,3))

            # input image
            input_reader = AICSImage(fn+'.ome.tif') #CZYX  #TODO: check size
            input_img = input_reader.data
            input_img = np.squeeze(input_img,axis=0)
            if input_img.shape[1] < input_img.shape[0]:
                input_img = np.transpose(input_img,(1,0,2,3))

            # cmap tensor
            costmap_reader = AICSImage(fn+'_CM.ome.tif') # ZYX
            costmap = costmap_reader.data
            costmap = np.squeeze(costmap,axis=0)
            if costmap.shape[0] == 1:
                costmap = np.squeeze(costmap,axis=0)
            elif costmap.shape[1] == 1:
                costmap = np.squeeze(costmap,axis=1)

            # output 
            outputs = model_inference(model, input_img, softmax, args)
            
            assert len(args.OutputCh)//2 == len(outputs)

            for vi in range(len(outputs)):
                if label.shape[0]==1: # the same label for all output
                    validation_loss[vi] += compute_iou(outputs[vi][0,:,:,:]>0.5, label[0,:,:,:]==args.OutputCh[2*vi+1], costmap)
                else:
                    validation_loss[vi] += compute_iou(outputs[vi][0,:,:,:]>0.5, label[vi,:,:,:]==args.OutputCh[2*vi+1], costmap)

        # print loss 
        average_training_loss = sum(epoch_loss) / len(epoch_loss)
        average_validation_loss = validation_loss / len(valid_filenames)
        print(f'Epoch: {epoch}, Training Loss: {average_training_loss}, Validation loss: {average_validation_loss}')
        print(f'{epoch},{average_training_loss},{average_validation_loss}\n', file = text_file)
        text_file.close()

        # save the model
        if args.SaveEveryKEpoch > 0 and epoch % args.SaveEveryKEpoch == 0:
            filename = f'{args.model}-{epoch:03}-{args.model_tag}.pth'
            torch.save(model.state_dict(), args.ModelDir + os.sep + filename)
            print(f'save at epoch: {epoch})')


def evaluate(args, model):

    model.eval()
    softmax = nn.Softmax(dim=1)
    softmax.cuda()

    # check validity of parameters
    assert args.nchannel == len(args.InputCh), f'number of input channel does not match input channel indices'

    if args.mode == 'eval':

        filenames = glob.glob(args.InputDir + '/*' + args.DataType)
        filenames.sort()

        for fi, fn in enumerate(filenames):
            print(fn)
            # load data
            struct_img = load_single_image(args, fn, time_flag=False)

            print(struct_img.shape)

            # apply the model
            output_img = apply_on_image(model, struct_img, softmax, args)
            #output_img = model_inference(model, struct_img, softmax, args)

            #print(len(output_img))

            for ch_idx in range(len(args.OutputCh)//2):
                write = omeTifWriter.OmeTifWriter(args.OutputDir + pathlib.PurePosixPath(fn).stem + '_seg_'+ str(args.OutputCh[2*ch_idx])+'.ome.tif')
                if args.Threshold<0:
                    write.save(output_img[ch_idx].astype(float))
                else:
                    out = output_img[ch_idx] > args.Threshold
                    out = out.astype(np.uint8)
                    out[out>0]=255
                    write.save(out)
            
            print(f'Image {fn} has been segmented')

    elif args.mode == 'eval_file':

        fn = args.InputFile
        print(fn)
        data_reader = AICSImage(fn)
        img0 = data_reader.data
        if args.timelapse:
            assert data_reader.shape[0]>1

            for tt in range(data_reader.shape[0]):
                # Assume:  TCZYX
                img = img0[tt, args.InputCh,:,:,:].astype(float)
                img = input_normalization(img, args)

                if len(args.ResizeRatio)>0:
                    img = resize(img, (1, args.ResizeRatio[0], args.ResizeRatio[1], args.ResizeRatio[2]), method='cubic')
                    for ch_idx in range(img.shape[0]):
                        struct_img = img[ch_idx,:,:,:] # note that struct_img is only a view of img, so changes made on struct_img also affects img
                        struct_img = (struct_img - struct_img.min())/(struct_img.max() - struct_img.min())
                        img[ch_idx,:,:,:] = struct_img

                # apply the model
                output_img = model_inference(model, img, softmax, args)

                for ch_idx in range(len(args.OutputCh)//2):
                    writer = omeTifWriter.OmeTifWriter(args.OutputDir + pathlib.PurePosixPath(fn).stem + '_T_'+ f'{tt:03}' +'_seg_'+ str(args.OutputCh[2*ch_idx])+'.ome.tif')
                    if args.Threshold<0:
                        out = output_img[ch_idx].astype(float)
                        out = resize(out, (1.0, 1/args.ResizeRatio[0], 1/args.ResizeRatio[1], 1/args.ResizeRatio[2]), method='cubic')
                        writer.save(out)
                    else:
                        out = output_img[ch_idx] > args.Threshold
                        out = resize(out, (1.0, 1/args.ResizeRatio[0], 1/args.ResizeRatio[1], 1/args.ResizeRatio[2]), method='nearest')
                        out = out.astype(np.uint8)
                        out[out>0]=255
                        writer.save(out)
        else:
            img = img0[0,:,:,:].astype(float)
            if img.shape[1] < img.shape[0]:
                img = np.transpose(img,(1,0,2,3))
            img = img[args.InputCh,:,:,:]
            img = input_normalization(img, args)

            if len(args.ResizeRatio)>0:
                img = resize(img, (1, args.ResizeRatio[0], args.ResizeRatio[1], args.ResizeRatio[2]), method='cubic')
                for ch_idx in range(img.shape[0]):
                    struct_img = img[ch_idx,:,:,:] # note that struct_img is only a view of img, so changes made on struct_img also affects img
                    struct_img = (struct_img - struct_img.min())/(struct_img.max() - struct_img.min())
                    img[ch_idx,:,:,:] = struct_img

            # apply the model
            output_img = model_inference(model, img, softmax, args)

            for ch_idx in range(len(args.OutputCh)//2):
                writer = omeTifWriter.OmeTifWriter(args.OutputDir + pathlib.PurePosixPath(fn).stem +'_seg_'+ str(args.OutputCh[2*ch_idx])+'.ome.tif')
                if args.Threshold<0:
                    writer.save(output_img[ch_idx].astype(float))
                else:
                    out = output_img[ch_idx] > args.Threshold
                    out = out.astype(np.uint8)
                    out[out>0]=255
                    writer.save(out)
        
        print(f'Image {fn} has been segmented')

def main(args):

    # set the seed
    torch.manual_seed(args.seed)
    # set up cuda
    torch.cuda.set_device(args.gpu)
    
    # build the model, cooresponding to the selected loss function
    model = None
    if args.model == 'unet':
        from aicsmlsegment.Net3D.uNet_original import UNet3D as DNN
        model = DNN(args.nchannel, args.nclass)
    elif args.model == 'unet_xy':
        from aicsmlsegment.Net3D.unet_xy import UNet3D as DNN
        model = DNN(args.nchannel, args.nclass)
    elif args.model == 'unet_xy_p3':
        from aicsmlsegment.Net3D.unet_xy_enlarge import UNet3D as DNN
        model = DNN(args.nchannel, args.nclass, 3)
    elif args.model == 'unet_xy_p2':
        from aicsmlsegment.Net3D.unet_xy_enlarge import UNet3D as DNN
        model = DNN(args.nchannel, args.nclass, 2)
        
    assert model is not None, f'model {args.model} not available'

    model = model.cuda()
    if args.state: # resume
        try:
            state = torch.load(args.state, map_location=torch.device('cpu'))
            if 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            else:
                model.load_state_dict(state)
            print('the resuming succeeds!')
            model.to(torch.device('cuda'))

        except AssertionError:
            model.load_state_dict(torch.load(args.state,
                map_location=lambda storage, loc: storage))
    else:
        model.apply(weights_init)
        print('finish model initialization and start new traing')

    if args.mode == 'train':
        if not args.TestMode:
            # add current setting to the experiment record
            df=pd.read_csv('//allen/aics/assay-dev/Segmentation/DeepLearning/SavedModel/NucMem/experiment_record.csv',index_col='index')
            new_df = pd.DataFrame([[args.model, args.state, args.DataPath, args.Loss, \
                args.Augmentation, args.EpochPerBuffer, args.PatchPerBuffer, args.BatchSize, \
                args.LeaveOut,args.lr, args.WeightDecay]],columns=df.columns)
            #new_df = pd.DataFrame({'model':args.model, 'state':args.state, 'DataPath':args.DataPath, 'Loss':args.Loss, \
            #    'Augmentation':args.Augmentation, 'EpochPerBuffer':args.EpochPerBuffer, 'PatchPerBuffer':args.PatchPerBuffer, \
            #    'BatchSize':args.BatchSize, 'LeaveOut':args.LeaveOut, 'LearningRate':args.lr, 'WeightDecay':args.WeightDecay})

            df = df.append(new_df,ignore_index=True)
            df.to_csv('//allen/aics/assay-dev/Segmentation/DeepLearning/SavedModel/NucMem/experiment_record.csv', index_label='index')

        train(args, model)
    elif args.mode == 'eval' or args.mode == 'eval_file':
        evaluate(args, model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--model', required=True)
    parser.add_argument('--state')  # for evaluation or resuming previous training
    parser.add_argument('--nchannel', type=int, default=1)
    parser.add_argument('--nclass', nargs='+', type=int, default=[2])
    parser.add_argument('--size_in', nargs='+',type=int, default=[100,140,140])
    parser.add_argument('--size_out', nargs='+', type=int, default=[12,52,52])
    parser.add_argument('--seed', type=int, default=167)
    parser.add_argument('--OutputCh',nargs='+', type=int, default=[0, 1, 1, 1])

    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True

    parser_eval = subparsers.add_parser('eval')
    parser_eval.add_argument('--InputDir', required=True)
    parser_eval.add_argument('--OutputDir', required=True)
    parser_eval.add_argument('--DataType',default='.tif')
    parser_eval.add_argument('--ResizeRatio',nargs='+', type=float, default=[1.0,1.0,1.0])
    parser_eval.add_argument('--InputCh', nargs='+', type=int, default=[0])
    parser_eval.add_argument('--Normalization',type=int, default=-1)
    parser_eval.add_argument('--Threshold',type=float, default=-1)
    parser_eval.add_argument('--RuntimeAug',default='None')

    parser_eval = subparsers.add_parser('eval_file')
    parser_eval.add_argument('--InputFile', required=True)
    parser_eval.add_argument('--OutputDir', required=True)
    parser_eval.add_argument('--ResizeRatio',nargs='+', type=float, default=[1.0,1.0,1.0])
    parser_eval.add_argument('--InputCh', nargs='+', type=int, default=[0])
    parser_eval.add_argument('--Normalization',type=int, default=-1)
    parser_eval.add_argument('--Threshold',type=float, default=-1)
    parser_eval.add_argument('--RuntimeAug',default='None')
    parser_eval.add_argument('--timelapse',action='store_true')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--Loss', required=True)
    parser_train.add_argument('--LossWeight', nargs='+', type=float, default=[1.0])
    parser_train.add_argument('--Augmentation', required=True)
    parser_train.add_argument('--DataPath', required=True)
    parser_train.add_argument('--EpochPerBuffer', type=int, default=1)
    parser_train.add_argument('--PatchPerBuffer', type=int, default=10)
    parser_train.add_argument('--ModelDir', required=True)
    parser_train.add_argument('--NumEpochs', type=int, default=15)
    parser_train.add_argument('--SaveEveryKEpoch', type=int, default=3)
    parser_train.add_argument('--NumWorkers', type=int, default=1)
    parser_train.add_argument('--BatchSize', type=int, default=2)
    parser_train.add_argument('--LeaveOut', nargs='+', type=float, default=[0.0,1.0])
    parser_train.add_argument('--model_tag', default='default')
    parser_train.add_argument('--lr', type=float, nargs='+', default=[1e-5], help='stage1, lr1, stage2, lr2, ...elsecase')
    parser_train.add_argument('--WeightDecay', '--wd', default=0.005, type=float, metavar='W', help='weight decay (default: 0.005)')
    parser_train.add_argument('--LoggerName', required=True)
    parser_train.add_argument('--TestMode', action='store_true')

    main(parser.parse_args())
