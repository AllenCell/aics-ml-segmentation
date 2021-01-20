#!/usr/bin/env python

import argparse
from aicsmlsegment.utils import (
    load_config,
)
from aicsmlsegment.monai_utils import Monai_BasicUNet, DataModule
import pytorch_lightning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config, model_config = load_config(args.config, train=False)

    # load the trained model instance
    model_path = config["model_path"]
    print(f"Loading model from {model_path}...")
    model = Monai_BasicUNet.load_from_checkpoint(
        model_path, config=config, model_config=model_config, train=False
    )

    gpu_config = config["gpus"]
    if gpu_config < -1:
        print("Number of GPUs must be -1 or > 0")
        quit()

    # ddp is the default unless only one gpu is requested
    accelerator = config["dist_backend"]

    trainer = pytorch_lightning.Trainer(
        gpus=gpu_config,
        num_sanity_val_steps=0,
        distributed_backend=accelerator,
    )
    data_module = DataModule(config, train=False)

    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":

    main()
