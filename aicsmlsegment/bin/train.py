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

    if config["resume"] is not None:
        print(f"Loading checkpoint '{config['resume']}'...")
        # load_checkpoint(config["resume"], model)
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
        verbose=True,
        save_top_k=-1,
    )

    callbacks = [MC]
    trainer = pytorch_lightning.Trainer(
        gpus=[0],
        max_epochs=config["epochs"],
        check_val_every_n_epoch=config["validation"]["validate_every_n_epoch"],
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )
    data_module = DataModule(config)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
