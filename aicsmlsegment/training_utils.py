#import torch.nn.functional as F
#from torch import nn as nn
#from torch.autograd import Variable
import logging
import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.optim as optim
import importlib
import random 
from glob import glob

from aicsimageio import AICSImage

from aicsmlsegment.custom_loss import MultiAuxillaryElementNLLLoss
from aicsmlsegment.custom_metrics import DiceCoefficient, MeanIoU, AveragePrecision
from aicsmlsegment.model_utils import load_checkpoint, save_checkpoint, model_inference
from aicsmlsegment.utils import compute_iou, get_logger

SUPPORTED_LOSSES = ['Aux'] #['ce', 'bce', 'wce', 'pce', 'dice', 'gdl']
SUPPORTED_METRICS = ['dice', 'iou', 'ap']

def build_optimizer(config, model):
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def build_lr_scheduler(config, optimizer):
    lr_config = config.get('lr_scheduler', None)
    if lr_config is None:
        # use ReduceLROnPlateau as a default scheduler
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=20, verbose=True)
    else:
        class_name = lr_config.pop('name')
        m = importlib.import_module('torch.optim.lr_scheduler')
        clazz = getattr(m, class_name)
        # add optimizer to the config
        lr_config['optimizer'] = optimizer
        return clazz(**lr_config)


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
    '''
    if weight is not None:
        # convert to cuda tensor if necessary
        weight = torch.tensor(weight).to(config['device'])

    if name == 'bce':
        skip_last_target = loss_config.get('skip_last_target', False)
        if ignore_index is None and not skip_last_target:
            return nn.BCEWithLogitsLoss()
        else:
            return BCELossWrapper(nn.BCEWithLogitsLoss(), ignore_index=ignore_index, skip_last_target=skip_last_target)
    elif name == 'ce':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif name == 'wce':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return WeightedCrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif name == 'pce':
        return PixelWiseCrossEntropyLoss(class_weights=weight, ignore_index=ignore_index)
    elif name == 'gdl':
        return GeneralizedDiceLoss(weight=weight, ignore_index=ignore_index)
    else:
        sigmoid_normalization = loss_config.get('sigmoid_normalization', True)
        skip_last_target = loss_config.get('skip_last_target', False)
        return DiceLoss(weight=weight, ignore_index=ignore_index, sigmoid_normalization=sigmoid_normalization,
                        skip_last_target=skip_last_target)
    '''
    if name == 'Aux':
        return MultiAuxillaryElementNLLLoss(3, loss_config['loss_weight'],  config['nclass'])


def get_validation_metric(config):
    """
    Returns the validaiton metric function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'eval_metric' key
    :return: an instance of the evaluation metric
    """
    assert 'val_metric' in config, 'Could not find validation metric configuration'
    val_config = config['val_metric']
    name = val_config['name']
    assert name in SUPPORTED_METRICS, f'Invalid validation metric: {name}. Supported metrics: {SUPPORTED_METRICS}'

    ignore_index = val_config.get('ignore_index', None)

    if name == 'dice':
        return DiceCoefficient(ignore_index=ignore_index)
    elif name == 'iou':
        skip_channels = val_config.get('skip_channels', ())
        return MeanIoU(skip_channels=skip_channels, ignore_index=ignore_index)
    elif name == 'ap':
        threshold = val_config.get('threshold', 0.5)
        min_instance_size = val_config.get('min_instance_size', None)
        use_last_target = val_config.get('use_last_target', False)
        return AveragePrecision(threshold=threshold, ignore_index=ignore_index, min_instance_size=min_instance_size,
                                use_last_target=use_last_target)


def get_train_dataloader(config):
    assert 'loader' in config, 'Could not loader configuration'
    name = config['loader']['name']
    if name == 'default':
        from aicsmlsegment.DataLoader3D.Universal_Loader import RR_FH_M0 as train_loader
        return train_loader
    else:
        print('other loaders are under construction')
        quit()

class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count

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
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        best_val_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 val_criterion, loaders, config, logger=None):

        if logger is None:
            self.logger = get_logger('ModelTrainer', level=logging.DEBUG)
        else:
            self.logger = logger

        device = config['device']
        self.logger.info(f"Sending the model to '{device}'")
        self.model = model.to(device)
        self.logger.debug(model)

        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.val_criterion = val_criterion
        self.device = device
        self.loaders = loaders

        self.size_in = config['size_in']
        self.size_out = config['size_out']

        # training setting
        self.checkpoint_dir = config['checkpoint_dir']
        self.max_num_epochs = config['epochs']
        self.save_every_n_epoch = config['save_every_n_epoch']

        # validation setting
        self.validation_config = config['validation']
        self.leaveout = self.validation_config['leaveout']

        self.writer = SummaryWriter(
            log_dir=os.path.join(self.checkpoint_dir, 'logs'))

        # dataloader config 
        loader_config = config['loader']
        self.batch_size = loader_config['batch_size']
        self.PatchPerBuffer = loader_config['PatchPerBuffer']
        self.NumWorkers = loader_config['NumWorkers']
 
        self.epoch_shuffle = loader_config['epoch_shuffle']
        self.datafolder = loader_config['datafolder']
        
        #TODO: these parameters could be updated when resuming
        self.num_iterations = 1  
        self.num_epoch = 1
        
        ##### customized loader ######
        if self.validation_config['metric'] is not None:
            # prepare the training/validattion filenames
            train_filenames, valid_filenames = shuffle_split_filenames(self.datafolder, self.validation_config['leaveout'])
            self.valid_filenames = valid_filenames
            self.train_filenames = train_filenames
        else:
            filenames = glob(self.datafolder + '/*_GT.ome.tif')
            filenames.sort()
            self.train_filenames = []
            self.valid_filenames = []
            for fi, fn in enumerate(filenames):
                self.train_filenames.append(fn[:-11])
            

        self.train_loader =  DataLoader(self.loaders(self.train_filenames, self.PatchPerBuffer, self.size_in, self.size_out), num_workers=self.NumWorkers, batch_size=self.batch_size, shuffle=True)
    
    def run_training(self):
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train()

            # check if we should stop #TODO: add early-stopping check
            if should_terminate:
                break

            self.num_epoch += 1

    def train(self):
        """Trains the model for 1 epoch.
        Args:
            train_loader (torch.utils.data.DataLoader): training data loader
        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = RunningAverage()
        #train_val_scores = RunningAverage()

        # sets the model in training mode
        self.model.train()

        # check if re-load on training data in needed
        if self.num_epoch>0 and  self.num_epoch % self.epoch_shuffle ==0:
           self.train_loader =  DataLoader(self.loaders(self.train_filenames, self.PatchPerBuffer, self.size_in, self.size_out), num_workers=self.NumWorkers, batch_size=self.batch_size, shuffle=True)

        for i, current_batch in enumerate(self.train_loader):
            self.logger.info(
                f'Training iteration {self.num_iterations}. Batch {i}. Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')
            

            if len(current_batch) == 2:  # no costmap
                input, target = current_batch
                input, target = input.to(self.device), target.to(self.device)
                weight = None
            else:
                input, target, weight = current_batch
                input, weight = input.to(self.device), weight.to(self.device)

                # target is always a list, containing one or more tensors
                if len(target)>1:
                    for list_idx in range(len(target)):
                        target[list_idx] = target[list_idx].to(self.device)
                else:
                    target = target[0].to(self.device)

            output, loss = self._forward_pass(input, target, weight)
            train_losses.update(loss.data.item(), input.size(0))
            print(f'iteration {self.num_iterations} in Epoch [{self.num_epoch}/{self.max_num_epochs - 1}], loss = {loss.data.item()}')

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.num_iterations += 1

        if self.num_epoch % self.save_every_n_epoch == 0:
            
            save_checkpoint({
                'epoch': self.num_epoch + 1,
                'num_iterations': self.num_iterations,
                'model_state_dict': self.model.state_dict(),
                #'best_val_score': self.best_val_score,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'device': str(self.device),
                }, checkpoint_dir=self.checkpoint_dir, logger=self.logger)


        # TODO: add validation step

        return False

    '''
    def validate(self):
        self.logger.info('Validating...')

        try:
            with torch.no_grad():

                # Validation starts ...
                validation_loss = np.zeros((len(self.OutputCh)//2,))
                self.model.eval() 

                for _, fn in enumerate(self.valid_filenames):

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

                    #input_img, label, costmap = input_img.to(self.device), label.to(self.device), costmap.to(self.device)
                    
                    # output 
                    args=lambda:None
                    args.size_in = self.size_in
                    args.size_out = self.size_out
                    args.OutputCh = self.OutputCh
                    args.nclass = [self.model.numClass, self.model.numClass1, self.model.numClass2] ####HACK#####
                    outputs = model_inference(self.model, input_img, self.model.final_activation, args)
                    assert len(self.OutputCh)//2 == len(outputs)

                    for vi in range(len(outputs)):
                        if label.shape[0]==1: # the same label for all output
                            validation_loss[vi] += compute_iou(outputs[vi][0,:,:,:]>0.5, label[0,:,:,:]==self.OutputCh[2*vi+1], costmap)
                        else:
                            validation_loss[vi] += compute_iou(outputs[vi][0,:,:,:]>0.5, label[vi,:,:,:]==self.OutputCh[2*vi+1], costmap)

                # print loss 
                average_validation_loss = validation_loss[0] / len(self.valid_filenames)
                self.writer.add_scalar('validation_score', average_validation_loss, self.num_iterations)
                self.logger.info(f'Validation finished. Evaluation score: {average_validation_loss}')
                return average_validation_loss

        finally:
            self.model.train()
    '''
    def _forward_pass(self, input, target, weight=None):
        # forward pass
        output = self.model(input)
        # compute the loss
        if weight is None:
            loss = self.loss_criterion(output, target)
        else:
            loss = self.loss_criterion(output, target, weight)

        return output, loss

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