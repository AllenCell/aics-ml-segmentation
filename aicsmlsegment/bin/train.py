import pytorch_lightning
import argparse
from aicsmlsegment.utils import load_config, get_logger
from aicsmlsegment.model_utils import (
    get_number_of_learnable_parameters,
)
from aicsmlsegment.monai_utils import Monai_BasicUNet, DataModule


SUPPORTED_MONAI_MODELS = [
    "BasicUNet",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # create logger
    logger = get_logger("ModelTrainer")
    config = load_config(args.config)
    logger.info(config)

    # load a specified saved model
    if config["resume"] is not None:
        print(f"Loading checkpoint '{config['resume']}'...")
        model = Monai_BasicUNet.load_from_checkpoint(
            config["resume"], config=config, train=True
        )
    else:
        print("Training new model from scratch")
        model = Monai_BasicUNet(config, train=True)

    # Log the number of learnable parameters

    # logger.info(
    #     f"Number of learnable params {get_number_of_learnable_parameters(model)}"
    # )

    checkpoint_dr = config["checkpoint_dir"]
    # model checkpoint every n epochs as specified in config
    MC = pytorch_lightning.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dr,
        filename="checkpoint_{epoch}",
        period=config["save_every_n_epoch"],
        save_top_k=-1,
    )
    callbacks = [MC]

    assert "callbacks" in config, "callbacks are required in config"
    callbacks_config = config["callbacks"]

    if callbacks_config["name"] == "EarlyStopping":
        es = pytorch_lightning.callbacks.EarlyStopping(
            monitor=callbacks_config["monitor"],
            min_delta=callbacks_config["min_delta"],
            patience=callbacks_config["patience"],
            verbose=callbacks_config["verbose"],
            mode=callbacks_config["verbose"],
        )
        callbacks.append(es)

    gpu_config = config["gpus"]
    if gpu_config is None:
        gpu_config = -1
    if gpu_config < -1:
        print("Number of GPUs must be -1 or > 0")
        quit()

    # ddp is the default unless only one gpu is requested
    accelerator = config["dist_backend"]
    if accelerator is None and gpu_config != 1:
        accelerator = "ddp"

    print("Training on ", gpu_config, "GPUs with backend", accelerator)
    print("Initializing trainer...", end=" ")
    trainer = pytorch_lightning.Trainer(
        gpus=gpu_config,
        max_epochs=config["epochs"],
        check_val_every_n_epoch=config["validation"]["validate_every_n_epoch"],
        num_sanity_val_steps=0,
        callbacks=callbacks,
        reload_dataloaders_every_epoch=False,  # check https://github.com/PyTorchLightning/pytorch-lightning/pull/5043 for updates on pull request
        # reload_dataloaders_every_n_epoch = config['loader']['epoch_shuffle']
        distributed_backend=accelerator,
    )
    print("Done")
    print("Initializing data module...", end=" ")
    data_module = DataModule(config)
    print("Done")
    trainer.fit(model, data_module)
    print(
        "The best performing checkpoint is",
        MC.best_model_path,
    )


if __name__ == "__main__":
    main()
