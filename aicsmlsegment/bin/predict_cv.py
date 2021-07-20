#!/usr/bin/env python
import os
import argparse
from aicsmlsegment.utils import load_config, create_unique_run_directory
from aicsmlsegment.Model import Model
from aicsmlsegment.DataUtils.DataMod_cv import DataModule_CV
import pytorch_lightning
import torch.autograd.profiler as profiler


def main():
    with profiler.profile(profile_memory=True) as prof:

        # load config
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        # parser.add_argument("--config2", required=True)
        args = parser.parse_args()
        config, model_config = load_config(args.config, train=False)
        # prepare output directory
        # output_dir = create_unique_run_directory(config, train=False)
        # config["OutputDir"] = output_dir

        # load model from cross validation
        config["total_cross_vali_num"] = 5
        for fold_id in range(config["total_cross_vali_num"]):
            config["current_cross_vali_fold"] = fold_id

            # load the trained model instance
            model_path = os.path.join(config["model_path"], 'cross_vali_'+str(fold_id), 'checkpoint_epoch=59.ckpt')
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

            print(config)

            # ddp is the default unless only one gpu is requested
            accelerator = config["dist_backend"]
            trainer = pytorch_lightning.Trainer(
                gpus=gpu_config,
                num_sanity_val_steps=0,
                distributed_backend=accelerator,
                precision=config["precision"],
            )
            data_module = DataModule_CV(config, train=False)
            # print(f'cross_vali_{fold_id}: {data_module.valid_filenames}')
            with profiler.record_function("inference"):
                trainer.test(model, datamodule=data_module)

    # print usage profile
    print(prof.key_averages().table())


if __name__ == "__main__":
    main()
