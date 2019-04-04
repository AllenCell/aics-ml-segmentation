import sys
import argparse
import logging
import traceback

from aicsmlsegment.utils import load_config

from aicsmlsegment.training_utils import BasicFolderTrainer, get_validation_metric, get_loss_criterion, build_optimizer, build_lr_scheduler, get_train_dataloader
from aicsmlsegment.utils import get_logger
from aicsmlsegment.model_utils import get_number_of_learnable_parameters, build_model, load_checkpoint


def main(args):

    # create logger
    logger = get_logger('ModelTrainer')
    config = load_config(args.config)
    logger.info(config)

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)

    # Create model
    model = build_model(config)

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create evaluation metric
    val_criterion = []
    if config['validation']['metric'] is not None:
        val_criterion = get_validation_metric(config['validation'])

    # create data loader
    loaders = get_train_dataloader(config)

    # Create the optimizer
    optimizer = build_optimizer(config, model)

    # Create learning rate adjustment strategy
    lr_scheduler = None
    #lr_scheduler = build_lr_scheduler(config, optimizer)

    # check if resuming
    if config['resume'] is not None:
        print(f"Loading checkpoint '{config['resume']}'...")
        load_checkpoint(config['resume'], model)
    else:
        print('start a new training')

    # run the training
    trainer = BasicFolderTrainer(model, optimizer, lr_scheduler, loss_criterion, val_criterion, 
                        loaders, config, logger=logger)
    trainer.run_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    main(parser.parse_args())