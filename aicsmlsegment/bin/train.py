import pytorch_lightning
import argparse
from aicsmlsegment.utils import load_config, get_logger
from aicsmlsegment.monai_utils import Model, DataModule
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    # torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # create logger
    logger = get_logger("ModelTrainer")
    config, model_config = load_config(args.config, train=True)
    logger.info(config)

    # load a specified saved model
    if config["resume"] is not None:
        print(f"Loading checkpoint '{config['resume']}'...")
        try:
            model = Model.load_from_checkpoint(
                config["resume"], config=config, model_config=model_config, train=True
            )
        except KeyError:  # backwards compatibility
            from aicsmlsegment.model_utils import load_checkpoint

            model = Model(config, model_config, train=True)
            load_checkpoint(config["resume"], model)
    else:
        print("Training new model from scratch")
        model = Model(config, model_config, train=True)

    checkpoint_dr = config["checkpoint_dir"]
    # model checkpoint every n epochs as specified in config
    MC = ModelCheckpoint(
        dirpath=checkpoint_dr,
        filename="checkpoint_{epoch}",
        period=config["save_every_n_epoch"],
        save_top_k=-1,
    )
    # LR = pytorch_lightning.callbacks.LearningRateLogger()
    callbacks = [MC]  # LR]

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

    if config["SWA"] is not None:
        assert (
            config["scheduler"]["name"] != "ReduceLROnPlateau"
        ), "ReduceLROnPlateau scheduler is not currently compatible with SWA"
        swa = pytorch_lightning.callbacks.StochasticWeightAveraging(
            swa_epoch_start=config["SWA"]["swa_start"],
            swa_lrs=config["SWA"]["swa_lr"],
            annealing_epochs=config["SWA"]["annealing_epochs"],
            annealing_strategy=config["SWA"]["annealing_strategy"],
        )
        callbacks.append(swa)

    gpu_config = config["gpus"]
    if gpu_config < -1:
        print("Number of GPUs must be -1 or > 0")
        quit()

    # ddp is the default unless only one gpu is requested
    accelerator = config["dist_backend"]
    if config["tensorboard"]:
        logger = pytorch_lightning.loggers.TensorBoardLogger("../../../lightning_logs/")
    else:
        logger = None

    print("Training on ", gpu_config, "GPUs with backend", accelerator)
    print("Initializing trainer...", end=" ")
    trainer = pytorch_lightning.Trainer(
        gpus=gpu_config,
        max_epochs=config["epochs"],
        check_val_every_n_epoch=config["validation"]["validate_every_n_epoch"],
        num_sanity_val_steps=1,
        callbacks=callbacks,
        reload_dataloaders_every_epoch=False,  # check https://github.com/PyTorchLightning/pytorch-lightning/pull/5043 for updates on pull request
        # reload_dataloaders_every_n_epoch = config['loader']['epoch_shuffle']
        distributed_backend=accelerator,
        logger=logger,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
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
