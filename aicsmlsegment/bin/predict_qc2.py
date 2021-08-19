#!/usr/bin/env python
import os
import numpy as np
import argparse
from aicsmlsegment.utils import load_config, create_unique_run_directory
from aicsmlsegment.Model_qc2 import Model
from aicsmlsegment.DataUtils.DataMod_qc2 import DataModule_qc
import pytorch_lightning
import torch.autograd.profiler as profiler
from pytorch_lightning.callbacks import Callback

# define custom Callback
class MyCustomCallback(Callback):
    
    def __init__(self, dirpath):
        super().__init__()
        self.dirpath = dirpath

    def on_test_end(self, trainer, pl_module):
        np.save(os.path.join(self.dirpath, 'out_list'), np.array(pl_module.out_list))
        np.save(os.path.join(self.dirpath, 'gt_list'), np.array(pl_module.gt_list))

def main():
    with profiler.profile(profile_memory=True) as prof:

        # load config
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        args = parser.parse_args()
        config, model_config = load_config(args.config, train=False)

        # load the trained model instance
        model_path = config["model_path"]
        print(f"Loading model from {model_path}...")
        try:
            model = Model.load_from_checkpoint(
                model_path, config=config, model_config=model_config, train=False
            )
        except KeyError:  # backwards compatibility for old .pytorch checkpoints
            from aicsmlsegment.model_utils import load_checkpoint

            model = Model(config, model_config, train=False)
            load_checkpoint(model_path, model)

        # set up GPU
        gpu_config = config["gpus"]
        if gpu_config < -1:
            print("Number of GPUs must be -1 or > 0")
            quit()

        # prepare output directory
        output_dir = create_unique_run_directory(config, train=False)
        config["OutputDir"] = output_dir

        print(config)

        callbacks = []
        myCustomCallback = MyCustomCallback(dirpath=output_dir)
        callbacks.append(myCustomCallback)

        # ddp is the default unless only one gpu is requested
        accelerator = config["dist_backend"]
        trainer = pytorch_lightning.Trainer(
            gpus=gpu_config,
            num_sanity_val_steps=0,
            callbacks=callbacks,
            distributed_backend=accelerator,
            precision=config["precision"],
        )
        data_module = DataModule_qc(config, train=False)
        trainer.test(model, datamodule=data_module)

    # # print usage profile
    # print(prof.key_averages().table())


if __name__ == "__main__":
    main()
