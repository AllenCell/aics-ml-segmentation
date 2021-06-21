#!/usr/bin/env python

import argparse
from aicsmlsegment.utils import load_config, create_unique_run_directory
from aicsmlsegment.Model import Model
from aicsmlsegment.DataUtils.DataMod import DataModule
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
        # config2, model_config2 = load_config(args.config2, train=False)

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

        # model2 = Model(config2, model_config2, train=False)
        # model_dict = model.state_dict()
        # state_dict = {}
        # for k, v in model_dict.items():
        #     if 'ec' in k or 'dc' in k:
        #         number = int(k[10])
        #         if number > 1:
        #             number = number + 1
        #         corrected_k = k[0:10]+str(number)+k[11:]
        #         state_dict[corrected_k] = v
        #     else:
        #         state_dict[k] = v
        # model2.load_state_dict(state_dict)

        # set up GPU
        gpu_config = config["gpus"]
        if gpu_config < -1:
            print("Number of GPUs must be -1 or > 0")
            quit()

        # prepare output directory
        output_dir = create_unique_run_directory(config, train=False)
        config["OutputDir"] = output_dir

        print(config)

        # ddp is the default unless only one gpu is requested
        accelerator = config["dist_backend"]
        trainer = pytorch_lightning.Trainer(
            gpus=gpu_config,
            num_sanity_val_steps=0,
            distributed_backend=accelerator,
            precision=config["precision"],
        )
        data_module = DataModule(config, train=False)
        with profiler.record_function("inference"):
            trainer.test(model, datamodule=data_module)

        # gpu_config = config2["gpus"]
        # if gpu_config < -1:
        #     print("Number of GPUs must be -1 or > 0")
        #     quit()

        # # prepare output directory
        # output_dir = create_unique_run_directory(config2, train=False)
        # config2["OutputDir"] = output_dir

        # print(config2)

        # # ddp is the default unless only one gpu is requested
        # accelerator = config2["dist_backend"]
        # trainer = pytorch_lightning.Trainer(
        #     gpus=gpu_config,
        #     num_sanity_val_steps=0,
        #     distributed_backend=accelerator,
        #     precision=config2["precision"],
        # )
        # data_module = DataModule(config2, train=False)
        # with profiler.record_function("inference"):
        #     trainer.test(model2, datamodule=data_module)

    # print usage profile
    print(prof.key_averages().table())


if __name__ == "__main__":
    main()
