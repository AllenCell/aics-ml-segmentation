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

    logger.info(
        f"Number of learnable params {get_number_of_learnable_parameters(model)}"
    )

    checkpoint_dr = config["checkpoint_dir"]
    # model checkpoint every n epochs as specified in config
    MC = pytorch_lightning.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dr,
        filename="checkpoint_{epoch}",
        period=config["save_every_n_epoch"],
        verbose=1,
        save_top_k=-1,
    )

    callbacks = [MC]
    print("Initializing trainer...", end=" ")
    trainer = pytorch_lightning.Trainer(
        gpus=-1,
        max_epochs=config["epochs"],
        check_val_every_n_epoch=config["validation"]["validate_every_n_epoch"],
        num_sanity_val_steps=0,
        callbacks=callbacks,
        reload_dataloaders_every_epoch=False,  # check https://github.com/PyTorchLightning/pytorch-lightning/pull/5043 for updates on pull request
        # reload_dataloaders_every_n_epoch = config['loader']['epoch_shuffle']
    )
    print("Done")
    print("Initializing data module...", end=" ")
    data_module = DataModule(config)
    print("Done")
    trainer.fit(model, data_module)
    print(
        "The best performing checkpoint is",
        trainer.best_model_path,
        "with score",
        trainer.best_model_score,
    )


if __name__ == "__main__":
    main()
