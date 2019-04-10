#!/usr/bin/env python

import sys
import argparse
import logging
import traceback

from aicsmlsegment.utils import load_config

from aicsmlsegment.training_utils import BasicFolderTrainer, get_loss_criterion, build_optimizer, get_train_dataloader
from aicsmlsegment.utils import get_logger
from aicsmlsegment.model_utils import get_number_of_learnable_parameters, build_model, load_checkpoint


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    # create logger
    logger = get_logger('ModelTrainer')
    config = load_config(args.config)
    logger.info(config)

    # Create model  
    model = build_model(config)

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # check if resuming
    if config['resume'] is not None:
        print(f"Loading checkpoint '{config['resume']}'...")
        load_checkpoint(config['resume'], model)
    else:
        print('start a new training')

    # run the training
    trainer = BasicFolderTrainer(model, config, logger=logger)
    trainer.train()

if __name__ == '__main__':
    main()