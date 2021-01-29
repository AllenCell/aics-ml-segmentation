import pytorch_lightning
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch

# from torchcontrib.optim import SWA    https://github.com/PyTorchLightning/pytorch-lightning/pull/5640

from monai.networks.nets import BasicUNet
import monai.losses as MonaiLosses

import aicsmlsegment.custom_loss as CustomLosses
import aicsmlsegment.custom_metrics as CustomMetrics
from aicsmlsegment.model_utils import model_inference, apply_on_image
from aicsmlsegment.DataLoader3D.Universal_Loader import (
    UniversalDataset,
    TestDataset,
    minmax,
    undo_resize,
)

from monai.metrics import DiceMetric
import random
import numpy as np
from glob import glob
from skimage.io import imsave
from skimage.morphology import remove_small_objects
import os
import pathlib

SUPPORTED_LOSSES = [
    "Dice",
    "GeneralizedDice",
    "Dice+CrossEntropy",
    "GeneralizedDice+CrossEntropy",
    "CrossEntropy",
    "Dice+FocalLoss",
    "GeneralizedDice+FocalLoss",
]


SUPPORTED_METRICS = [
    "default",
    "Dice",
]


def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function and whether it accepts a costmap and loss weights
    """
    name = config["loss"]["name"]

    # weight = loss_config["loss_weight"]
    assert (
        name in SUPPORTED_LOSSES
    ), f"Invalid loss: {name}. Supported losses: {SUPPORTED_LOSSES}"

    if name == "Dice":
        return MonaiLosses.DiceLoss(sigmoid=True), False, None
    elif name == "GeneralizedDice":
        return MonaiLosses.GeneralizedDiceLoss(sigmoid=True), False, None
    elif name == "Dice+CrossEntropy":
        return (
            CustomLosses.CombinedLoss(
                MonaiLosses.DiceLoss(sigmoid=True), torch.nn.BCEWithLogitsLoss()
            ),
            False,
            None,
        )
    elif name == "GeneralizedDice+CrossEntropy":
        return (
            CustomLosses.CombinedLoss(
                MonaiLosses.GeneralizedDiceLoss(sigmoid=True),
                torch.nn.BCEWithLogitsLoss(),
            ),
            False,
            None,
        )
    elif name == "CrossEntropy":
        return (torch.nn.BCEWithLogitsLoss(), False, None)
    elif name == "Dice+FocalLoss":
        return (
            CustomLosses.DiceFocalLoss(config["validation"]["OutputCh"]),
            False,
            None,
        )
    elif name == "GeneralizedDice+FocalLoss":
        return (
            CustomLosses.GeneralizedDiceFocalLoss(config["validation"]["OutputCh"]),
            False,
            None,
        )


def get_metric(config):
    validation_config = config["validation"]
    metric = validation_config["metric"]

    assert (
        metric in SUPPORTED_METRICS
    ), f"Invalid metric: {metric}. Supported metrics are: {SUPPORTED_METRICS}"

    if metric == "Dice":
        return DiceMetric
    elif metric == "default" or metric == "IOU":
        return CustomMetrics.MeanIoU()
    elif metric == "AveragePrecision":
        return CustomMetrics.AveragePrecision()


class Monai_BasicUNet(pytorch_lightning.LightningModule):
    def __init__(self, config, model_config, train):
        super().__init__()
        self.model = BasicUNet(**model_config)
        self.norm = config["model"]["norm"]
        self.config = config
        self.args_inference = {}
        if train:
            loader_config = config["loader"]
            self.datapath = loader_config["datafolder"]
            self.nworkers = loader_config["NumWorkers"]
            self.batchsize = loader_config["batch_size"]

            validation_config = config["validation"]
            self.leaveout = validation_config["leaveout"]
            self.validation_period = validation_config["validate_every_n_epoch"]

            self.lr = config["learning_rate"]
            self.weight_decay = config["weight_decay"]

            self.args_inference["inference_batch_size"] = loader_config["batch_size"]
            self.args_inference["OutputCh"] = validation_config["OutputCh"]

            (
                self.loss_function,
                self.accepts_costmap,
                self.loss_weight,
            ) = get_loss_criterion(config)
            self.metric = get_metric(config)

            self.scheduler_params = config["scheduler"]
            # self.swa_config = config["SWA"]

        else:
            if config["RuntimeAug"] <= 0:
                self.args_inference["RuntimeAug"] = False
            else:
                self.args_inference["RuntimeAug"] = True
            self.args_inference["OutputCh"] = config["OutputCh"]
            self.args_inference["inference_batch_size"] = config["batch_size"]
            self.args_inference["mode"] = config["mode"]["name"]
            self.args_inference["Threshold"] = config["Threshold"]

        self.args_inference["size_out"] = config["model"]["patch_size"]

    def forward(self, x):
        """
        returns raw predictions
        """
        return self.model(x)

    def configure_optimizers(self):
        print("Configuring optimizers")
        optims = []
        scheds = []

        scheduler_params = self.scheduler_params

        # basic optimizer
        optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        optims.append(optimizer)

        if scheduler_params["name"] is not None:
            if scheduler_params["name"] == "ExponentialLR":
                from torch.optim.lr_scheduler import ExponentialLR

                assert scheduler_params["gamma"] > 0
                scheduler = ExponentialLR(
                    optims[0],
                    gamma=scheduler_params["gamma"],
                    verbose=scheduler_params["verbose"],
                )

            elif scheduler_params["name"] == "CosineAnnealingLR":
                from torch.optim.lr_scheduler import CosineAnnealingLR as CALR

                assert scheduler_params["T_max"] > 0
                scheduler = CALR(
                    optims[0],
                    T_max=scheduler_params["T_max"],
                    verbose=scheduler_params["verbose"],
                )

            elif scheduler_params["name"] == "StepLR":
                from torch.optim.lr_scheduler import StepLR

                assert scheduler_params["step_size"] > 0
                assert scheduler_params["gamma"] > 0
                scheduler = StepLR(
                    optims[0],
                    step_size=scheduler_params["step_size"],
                    gamma=scheduler_params["gamma"],
                    verbose=scheduler_params["verbose"],
                )
            elif scheduler_params["name"] == "ReduceLROnPlateau":
                from torch.optim.lr_scheduler import ReduceLROnPlateau

                assert 0 < scheduler_params["factor"] < 1
                assert scheduler_params["patience"] > 0
                # if patience is too short, validation metrics won't be available
                if "val" in scheduler_params["monitor"]:
                    assert (
                        scheduler_params["patience"] > self.validation_period
                    ), "Patience must be larger than validation frequency"
                scheduler = ReduceLROnPlateau(
                    optims[0],
                    mode=scheduler_params["mode"],
                    factor=scheduler_params["factor"],
                    patience=scheduler_params["patience"],
                    verbose=scheduler_params["verbose"],
                )
                # monitoring metric must be specified
                return {
                    "optimizer": optims[0],
                    "lr_scheduler": scheduler,
                    "monitor": scheduler_params["monitor"],
                }
            else:
                print(
                    "That scheduler is not yet supported. No scheduler is being used."
                )
                return optims
            print("done")
            scheds.append(scheduler)
            return optims, scheds
        else:
            print("no scheduler is used")
            return optims

    def training_step(self, batch, batch_idx):
        inputs = batch[0]
        targets = batch[1]
        outputs = self.forward(inputs)

        # focal loss requires > 1 channel
        if "Focal" not in self.config["loss"]["name"]:
            # select output channel
            outputs = outputs[:, self.args_inference["OutputCh"], :, :, :]
            outputs = torch.unsqueeze(
                outputs, dim=1
            )  # add back in channel dimension to match targets

        if self.accepts_costmap:
            cmap = batch[2]
            loss = self.loss_function(outputs, targets, cmap)
        else:
            if self.loss_weight is not None:
                loss = self.loss_function(outputs, targets, self.loss_weight)
            else:
                loss = self.loss_function(outputs, targets)
        # metric = self.metric(outputs, targets)
        self.log(
            "epoch_train_loss",
            loss,
            sync_dist=True,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_img = batch[0]
        label = batch[1]

        extract = True
        squeeze = False
        # focal loss needs >1 channel in predictions
        if "Focal" in self.config["loss"]["name"]:
            extract = False
            squeeze = True
        outputs = model_inference(
            self.model,
            input_img,
            self.args_inference,
            squeeze=squeeze,
            extract_output_ch=extract,
        )

        if self.accepts_costmap:
            costmap = batch[2]
            val_loss = self.loss_function(outputs, label, costmap)
        else:
            val_loss = self.loss_function(outputs, label)

        # val_metric = self.metric(outputs, label)

        # sync_dist on_epoch=True ensures that results will be averaged across gpus
        self.log("val_loss", val_loss, sync_dist=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        img = batch["img"]
        fn = batch["fn"][0]
        tt = batch["tt"]
        # default comes through as double tensor
        img = img.float()

        args_inference = self.args_inference
        output_img = apply_on_image(
            self.model, img, args_inference, squeeze=False, to_numpy=True, sigmoid=True
        )

        if args_inference["mode"] != "folder":
            out = minmax(output_img)
            out = undo_resize(out, self.config)
            if args_inference["Threshold"] > 0:
                out = out > args_inference["Threshold"]
                out = out.astype(np.uint8)
                out[out > 0] = 255
        else:
            if args_inference["Threshold"] < 0:
                out = minmax(output_img)
                out = undo_resize(out, self.config)
                out = minmax(out)
            else:
                out = remove_small_objects(
                    output_img > args_inference["Threshold"],
                    min_size=2,
                    connectivity=1,
                )
                out = out.astype(np.uint8)
                out[out > 0] = 255

        if len(tt) == 0:
            imsave(
                self.config["OutputDir"]
                + os.sep
                + pathlib.PurePosixPath(fn).stem
                + "_struct_segmentation.tiff",
                out,
            )
        else:
            imsave(
                self.config["OutputDir"]
                + os.sep
                + pathlib.PurePosixPath(fn).stem
                + "_T_"
                + f"{tt:03}"
                + "_struct_segmentation.tiff",
                out,
            )
        self.log("test_loss", 0, on_step=False, on_epoch=False)


class DataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, config, train=True):
        super().__init__()
        self.config = config

        if train:
            self.loader_config = config["loader"]
            self.model_config = config["model"]

            name = config["loader"]["name"]
            if name != "default":
                print("other loaders are under construction")
                quit()

            self.transforms = []
            if "Transforms" in self.loader_config:
                self.transforms = self.loader_config["Transforms"]

    def prepare_data(self):
        pass

    def setup(self, stage):
        if stage == "fit":  # no setup is required for testing
            # load settings #
            config = self.config

            # get validation and training filenames from input dir from config
            validation_config = config["validation"]
            loader_config = config["loader"]
            if validation_config["metric"] is not None:
                print("Preparing train/validation split...", end=" ")
                filenames = glob(loader_config["datafolder"] + "/*_GT.ome.tif")
                filenames.sort()
                total_num = len(filenames)
                LeaveOut = validation_config["leaveout"]
                if len(LeaveOut) == 1:
                    if LeaveOut[0] > 0 and LeaveOut[0] < 1:
                        num_train = int(np.floor((1 - LeaveOut[0]) * total_num))
                        shuffled_idx = np.arange(total_num)
                        random.shuffle(shuffled_idx)
                        train_idx = shuffled_idx[:num_train]
                        valid_idx = shuffled_idx[num_train:]
                    else:
                        valid_idx = [int(LeaveOut[0])]
                        train_idx = list(
                            set(range(total_num)) - set(map(int, LeaveOut))
                        )
                elif LeaveOut:
                    valid_idx = list(map(int, LeaveOut))
                    train_idx = list(set(range(total_num)) - set(valid_idx))

                valid_filenames = []
                train_filenames = []
                # remove file extensions from filenames
                for fi, fn in enumerate(valid_idx):
                    valid_filenames.append(filenames[fn][:-11])
                for fi, fn in enumerate(train_idx):
                    train_filenames.append(filenames[fn][:-11])

                self.valid_filenames = valid_filenames
                self.train_filenames = train_filenames
                print("Done.")

            else:
                print("need validation in config file")
                quit()

    def train_dataloader(self):
        print("Initializing train dataloader: ", end=" ")
        loader_config = self.loader_config
        model_config = self.model_config
        train_set_loader = DataLoader(
            UniversalDataset(
                self.train_filenames,
                loader_config["PatchPerBuffer"],
                model_config["patch_size"],
                model_config["in_channels"],
                self.transforms,
                patchize=True,
            ),
            batch_size=loader_config["batch_size"],
            shuffle=True,
            num_workers=loader_config["NumWorkers"],
        )
        return train_set_loader

    def val_dataloader(self):
        print("Initializing validation dataloader: ", end=" ")
        loader_config = self.loader_config
        model_config = self.model_config
        val_set_loader = DataLoader(
            UniversalDataset(
                self.valid_filenames,
                loader_config["PatchPerBuffer"],
                model_config["patch_size"],
                model_config["in_channels"],
                [],  # no transforms for validation data
                patchize=False,  # validate on entire image
            ),
            batch_size=loader_config["batch_size"],
            shuffle=False,
            num_workers=loader_config["NumWorkers"],
        )
        return val_set_loader

    def test_dataloader(self):
        test_set_loader = DataLoader(
            TestDataset(self.config),
            batch_size=1,
            shuffle=False,
            num_workers=self.config["NumWorkers"],
        )
        return test_set_loader
