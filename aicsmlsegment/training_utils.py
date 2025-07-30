#import torch.nn.functional as F
#from torch import nn as nn
#from torch.autograd import Variable
import logging
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import importlib
import random 
from glob import glob
from tqdm import tqdm

from aicsimageio import AICSImage

from aicsmlsegment.custom_loss import MultiAuxillaryElementNLLLoss
from aicsmlsegment.custom_metrics import DiceCoefficient, MeanIoU, AveragePrecision
from aicsmlsegment.model_utils import load_checkpoint, save_checkpoint, model_inference
from aicsmlsegment.utils import compute_iou, get_logger, load_single_image, input_normalization

SUPPORTED_LOSSES = ['Aux'] 

def build_optimizer(config, model):
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    assert 'loss' in config, 'Could not find loss function configuration'
    loss_config = config['loss']
    name = loss_config['name']
    assert name in SUPPORTED_LOSSES, f'Invalid loss: {name}. Supported losses: {SUPPORTED_LOSSES}'

    #ignore_index = loss_config.get('ignore_index', None)

    #TODO: add more loss functions
    if name == 'Aux':
        return MultiAuxillaryElementNLLLoss(3, loss_config['loss_weight'],  config['nclass'])


def get_train_dataloader(config):
    assert 'loader' in config, 'Could not loader configuration'
    name = config['loader']['name']
    if name == 'default':
        from aicsmlsegment.DataLoader3D.Universal_Loader import RR_FH_M0 as train_loader
        return train_loader
    else:
        print('other loaders are under construction')
        quit()

def shuffle_split_filenames(datafolder, leaveout):
    print('prepare the data ... ...')
    filenames = glob(datafolder + '/*_GT.ome.tif')
    filenames.sort()
    total_num = len(filenames)
    if len(leaveout)==1:
        if leaveout[0]>0 and leaveout[0]<1:
            num_train = int(np.floor((1-leaveout[0]) * total_num))
            shuffled_idx = np.arange(total_num)
            random.shuffle(shuffled_idx)
            train_idx = shuffled_idx[:num_train]
            valid_idx = shuffled_idx[num_train:]
        else:
            valid_idx = [int(leaveout[0])]
            train_idx = list(set(range(total_num)) - set(map(int, leaveout)))  
    elif leaveout:
        valid_idx = list(map(int, leaveout))  
        train_idx = list(set(range(total_num)) - set(valid_idx))
    
    valid_filenames = []
    train_filenames = []
    for _, fn in enumerate(valid_idx):
        valid_filenames.append(filenames[fn][:-11])
    for _, fn in enumerate(train_idx):
        train_filenames.append(filenames[fn][:-11])

    return train_filenames, valid_filenames

class BasicFolderTrainer:
    """basic version of trainer.
    Args:
        model: model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        loss_criterion (callable): loss function
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
    """

    def __init__(self, model, config, logger=None):

        if logger is None:
            self.logger = get_logger('ModelTrainer', level=logging.DEBUG)
        else:
            self.logger = logger

        device = config['device']
        self.logger.info(f"Sending the model to '{device}'")
        self.model = model.to(device)
        self.logger.debug(model)

        #self.optimizer = optimizer
        #self.scheduler = lr_scheduler
        #self.loss_criterion = loss_criterion
        self.device = device
        #self.loaders = loaders
        self.config = config


    def train(self):

        ### load settings ###
        config = self.config #TODO, fix this
        model = self.model

        # define loss 
        #TODO, add more loss
        loss_config = config['loss']
        if loss_config['name']=='Aux':
            criterion = MultiAuxillaryElementNLLLoss(3,loss_config['loss_weight'], config['nclass'])
        else:
            print('do not support other loss yet')
            quit()

        # dataloader
        validation_config = config['validation']
        loader_config = config['loader']
        args_inference=lambda:None
        if validation_config['metric'] is not None:
            print('prepare the data ... ...')
            filenames = glob(loader_config['datafolder'] + '/*_GT.ome.tif')
            filenames.sort()
            total_num = len(filenames)
            LeaveOut = validation_config['leaveout']
            if len(LeaveOut)==1:
                if LeaveOut[0]>0 and LeaveOut[0]<1:
                    num_train = int(np.floor((1-LeaveOut[0]) * total_num))
                    shuffled_idx = np.arange(total_num)
                    random.shuffle(shuffled_idx)
                    train_idx = shuffled_idx[:num_train]
                    valid_idx = shuffled_idx[num_train:]
                else:
                    valid_idx = [int(LeaveOut[0])]
                    train_idx = list(set(range(total_num)) - set(map(int, LeaveOut)))  
            elif LeaveOut:
                valid_idx = list(map(int, LeaveOut))  
                train_idx = list(set(range(total_num)) - set(valid_idx))
            
            valid_filenames = []
            train_filenames = []
            for fi, fn in enumerate(valid_idx):
                valid_filenames.append(filenames[fn][:-11])
            for fi, fn in enumerate(train_idx):
                train_filenames.append(filenames[fn][:-11])

            args_inference.size_in = config['size_in']
            args_inference.size_out = config['size_out']
            args_inference.OutputCh = validation_config['OutputCh']
            args_inference.nclass =  config['nclass'] 

        else:
            #TODO, update here
            print('need validation')
            quit()

        if loader_config['name']=='default':
            from aicsmlsegment.DataLoader3D.Universal_Loader import RR_FH_M0 as train_loader
            train_set_loader = DataLoader(train_loader(train_filenames, loader_config['PatchPerBuffer'], config['size_in'], config['size_out']), num_workers=loader_config['NumWorkers'], batch_size=loader_config['batch_size'], shuffle=True)
        elif loader_config['name']=='focus':
            from aicsmlsegment.DataLoader3D.Universal_Loader import RR_FH_M0C as train_loader
            train_set_loader = DataLoader(train_loader(train_filenames, loader_config['PatchPerBuffer'], config['size_in'], config['size_out']), num_workers=loader_config['NumWorkers'], batch_size=loader_config['batch_size'], shuffle=True)
        else:
            print('other loader not support yet')
            quit()

        num_iterations = 0 
        num_epoch = 0 #TODO: load num_epoch from checkpoint

        start_epoch = num_epoch
        for _ in range(start_epoch, config['epochs']+1):

            # sets the model in training mode
            model.train()

            optimizer = None
            optimizer = optim.Adam(model.parameters(),lr = config['learning_rate'], weight_decay = config['weight_decay'])

            # check if re-load on training data in needed
            if num_epoch>0 and  num_epoch % loader_config['epoch_shuffle'] ==0:
                print('shuffling data')
                train_set_loader = None
                train_set_loader = DataLoader(train_loader(train_filenames, loader_config['PatchPerBuffer'], config['size_in'], config['size_out']), num_workers=loader_config['NumWorkers'], batch_size=loader_config['batch_size'], shuffle=True)

            # Training starts ...
            epoch_loss = []

            for i, current_batch in tqdm(enumerate(train_set_loader)):
                    
                inputs = Variable(current_batch[0].cuda())
                targets = current_batch[1]
                outputs = model(inputs)

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
                num_iterations += 1

            average_training_loss = sum(epoch_loss) / len(epoch_loss)

            # validation
            if num_epoch % validation_config['validate_every_n_epoch'] ==0:
                validation_loss = np.zeros((len(validation_config['OutputCh'])//2,))
                model.eval() 

                for img_idx, fn in enumerate(valid_filenames):

                    # target 
                    label_reader = AICSImage(fn+'_GT.ome.tif')  #CZYX
                    label = label_reader.get_image_data('CZYX', S=0, T=0, C=[0]) # need 4D output

                    # input image
                    input_reader = AICSImage(fn+'.ome.tif')
                    input_img = input_reader.get_image_data('CZYX', S=0, T=0, C=[0])  # need 4D output

                    # cmap tensor
                    costmap_reader = AICSImage(fn+'_CM.ome.tif') # ZYX
                    costmap = costmap_reader.get_image_data('CZYX', S=0, T=0, C=[0])  # need 4D output

                    # output 
                    outputs = model_inference(model, input_img, model.final_activation, args_inference)
                    
                    assert len(validation_config['OutputCh'])//2 == len(outputs)

                    for vi in range(len(outputs)):
                        if label.shape[0]==1: # the same label for all output
                            validation_loss[vi] += compute_iou(outputs[vi][0,:,:,:]>0.5, label[0,:,:,:]==validation_config['OutputCh'][2*vi+1], costmap)
                        else:
                            validation_loss[vi] += compute_iou(outputs[vi][0,:,:,:]>0.5, label[vi,:,:,:]==validation_config['OutputCh'][2*vi+1], costmap)


                
                average_validation_loss = validation_loss / len(valid_filenames)
                print(f'Epoch: {num_epoch}, Training Loss: {average_training_loss}, Validation loss: {average_validation_loss}')
            else:
                print(f'Epoch: {num_epoch}, Training Loss: {average_training_loss}')


            if num_epoch % config['save_every_n_epoch'] == 0:
                save_checkpoint({
                    'epoch': num_epoch,
                    'num_iterations': num_iterations,
                    'model_state_dict': model.state_dict(),
                    #'best_val_score': self.best_val_score,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'device': str(self.device),
                    }, checkpoint_dir=config['checkpoint_dir'], logger=self.logger)
            num_epoch += 1

        # TODO: add validation step

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        self.logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(),
                                      self.num_iterations)
            self.writer.add_histogram(name + '/grad',
                                      value.grad.data.cpu().numpy(),
                                      self.num_iterations)

    def _log_images(self, input, target, prediction):
        sources = {
            'inputs': input.data.cpu().numpy(),
            'targets': target.data.cpu().numpy(),
            'predictions': prediction.data.cpu().numpy()
        }
        for name, batch in sources.items():
            for tag, image in self._images_from_batch(name, batch):
                self.writer.add_image(tag, image, self.num_iterations, dataformats='HW')