import sys
import argparse
import logging
import traceback

from aicsmlsegment.utils import load_config

from aicsmlsegment.training_utils import BasicFolderTrainer, get_validation_metric, get_loss_criterion, build_optimizer, build_lr_scheduler, get_train_dataloader
from aicsmlsegment.utils import get_logger
from aicsmlsegment.model_utils import get_number_of_learnable_parameters, build_model


def main(args):

    # create logger
    logger = get_logger('ModelTrainer')
    config = load_config(args.config)
    logger.info(config)

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)

    # Create model
    model = build_model(config)
    #model = UNet3D(config['in_channels'], config['out_channels'],
    #               final_sigmoid=config['final_sigmoid'],
    #               init_channel_number=config['init_channel_number'],
    #               conv_layer_order=config['layer_order'],
    #               interpolate=config['interpolate'])
    # model = model.to(config['device'])

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create evaluation metric
    val_criterion = get_validation_metric(config)

    # create data loader
    loaders = get_train_dataloader(config)
    loader_config = config['loader']

    # Create the optimizer
    optimizer = build_optimizer(config, model)

    # Create learning rate adjustment strategy
    lr_scheduler = build_lr_scheduler(config, optimizer)

    # do the training
    if config['resume'] is not None:
        trainer = BasicFolderTrainer.from_checkpoint(config['resume'], model, loader_config,
                                                optimizer, lr_scheduler, loss_criterion,
                                                val_criterion, loaders, config['OutputCh'],
                                                logger=logger)
    else:
        print('start a new training')
    trainer = BasicFolderTrainer(model, loader_config, optimizer, lr_scheduler, loss_criterion, val_criterion,
                            config['device'], loaders, config['OutputCh'], config['checkpoint_dir'],
                            max_num_epochs=config['epochs'],
                            validate_every_n_epoch=config['validate_every_n_epoch'],
                            logger=logger)
    trainer.run_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    main(parser.parse_args())