import pytorch_lightning
import argparse
from aicsmlsegment.utils import load_config, get_logger, create_unique_run_directory
from aicsmlsegment.Model import Model
from aicsmlsegment.DataUtils.DataMod import DataModule
from pytorch_lightning.callbacks import ModelCheckpoint


def main(config=None, model_config=None):

    #########
    # only for debugging
    #########
    if config is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        args = parser.parse_args()
        # create logger
        config, model_config = load_config(args.config, train=True)

    logger = get_logger("ModelTrainer")

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

    checkpoint_dir = create_unique_run_directory(config, train=True)
    config["checkpoint_dir"] = checkpoint_dir
    logger.info(config)

    # save model checkpoint every n epochs
    MC = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="checkpoint_{epoch}",
        period=config["save_every_n_epoch"],
        save_top_k=-1,
    )
    callbacks = [MC]

    callbacks_config = config["callbacks"]

    # it is possible to use early stopping by adding callback config
    # in configuration yaml
    if callbacks_config["name"] == "EarlyStopping":
        es = pytorch_lightning.callbacks.EarlyStopping(
            monitor=callbacks_config["monitor"],
            min_delta=callbacks_config["min_delta"],
            patience=callbacks_config["patience"],
            verbose=callbacks_config["verbose"],
            mode=callbacks_config["verbose"],
        )
        callbacks.append(es)

    # it is possible to use stachastic weight averaging by adding
    # a "SWA" option in configuration yaml
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

    # gpu setting
    gpu_config = config["gpus"]
    if gpu_config < -1:
        print("Number of GPUs must be -1 or > 0")
        quit()

    # ddp is the default unless only one gpu is requested
    accelerator = config["dist_backend"]
    plugins = None
    if accelerator == "ddp":
        from pytorch_lightning.plugins import DDPPlugin

        # reduces multi-gpu model memory, removes unecessary backwards pass
        plugins = ["ddp_sharded", DDPPlugin(find_unused_parameters=False)]

    # it is possible to use tensorboard to track the experiment by adding
    # a "tensorboard" option in the configuration yaml
    if config["tensorboard"] is not None:
        from pytorch_lightning.callbacks import LearningRateMonitor, GPUStatsMonitor
        from pytorch_lightning.loggers import TensorBoardLogger

        logger = TensorBoardLogger(config["tensorboard"])
        GPU = GPUStatsMonitor(intra_step_time=True, inter_step_time=True)
        LR = LearningRateMonitor(logging_interval="epoch")
        callbacks += [GPU, LR]
    else:
        from pytorch_lightning.loggers import CSVLogger

        logger = CSVLogger(save_dir=checkpoint_dir)

    # define the model trainer
    trainer = pytorch_lightning.Trainer(
        gpus=gpu_config,
        max_epochs=config["epochs"],
        check_val_every_n_epoch=config["validation"]["validate_every_n_epoch"],
        num_sanity_val_steps=0,
        callbacks=callbacks,
        reload_dataloaders_every_epoch=False,  # check https://github.com/PyTorchLightning/pytorch-lightning/pull/5043 for updates on pull request  # noqa E501
        # reload_dataloaders_every_n_epoch = config['loader']['epoch_shuffle']
        distributed_backend=accelerator,
        logger=logger,
        precision=config["precision"],
        plugins=plugins,
    )

    # define the data module
    data_module = DataModule(config)

    # starts training
    trainer.fit(model, data_module)

    # after training is done, print the best model
    print(
        "The best performing checkpoint is",
        MC.best_model_path,
    )


if __name__ == "__main__":
    main()
