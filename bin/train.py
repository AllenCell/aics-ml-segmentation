import importlib
import sys
import argparse
import logging
import traceback

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import load_config


from datasets.hdf5 import get_train_loaders

from unet3d.losses import get_loss_criterion
from unet3d.metrics import get_evaluation_metric
from unet3d.model import UNet3D
from unet3d.trainer import UNet3DTrainer
from unet3d.utils import get_logger
from unet3d.utils import get_number_of_learnable_parameters

###############################################################################

log = logging.getLogger()
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s')

###############################################################################


def _create_optimizer(config, model):
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def _create_lr_scheduler(config, optimizer):
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


def main():
    try:
        args = Args()
        dbg = args.debug

        # Do your work here - preferably in a class or function,
        # passing in your args. E.g.
        exe = Example(args.first)
        exe.update_value(args.second)
        print("First : {}\nSecond: {}".format(exe.get_value(), exe.get_previous_value()))

    except Exception as e:
        log.error("=============================================")
        if dbg:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


    logger = get_logger('UNet3DTrainer')

    config = load_config()

    logger.info(config)

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)

    # Create model
    model = model_create(config)
    #model = UNet3D(config['in_channels'], config['out_channels'],
    #               final_sigmoid=config['final_sigmoid'],
    #               init_channel_number=config['init_channel_number'],
    #               conv_layer_order=config['layer_order'],
    #               interpolate=config['interpolate'])
    # model = model.to(config['device'])

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)

    # create data loader
    loaders = get_train_loaders(config)

    # Create the optimizer
    optimizer = _create_optimizer(config, model)

    # Create learning rate adjustment strategy
    lr_scheduler = _create_lr_scheduler(config, optimizer)

    # do the training
    if config['resume'] is not None:
        trainer = UNet3DTrainer.from_checkpoint(config['resume'], model,
                                                optimizer, lr_scheduler, loss_criterion,
                                                eval_criterion, loaders,
                                                logger=logger)
    else:
        trainer = UNet3DTrainer(model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                                config['device'], loaders, config['checkpoint_dir'],
                                max_num_epochs=config['epochs'],
                                max_num_iterations=config['iters'],
                                validate_after_iters=config['validate_after_iters'],
                                log_after_iters=config['log_after_iters'],
                                logger=logger)
    trainer.fit()


if __name__ == '__main__':
    main()