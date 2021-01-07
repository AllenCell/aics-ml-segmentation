import pytorch_lightning
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch

from monai.networks.layers import Norm
from monai.networks.nets import BasicUNet

import aicsmlsegment.custom_loss as CustomLosses
import aicsmlsegment.custom_metrics as CustomMetrics
from aicsmlsegment.model_utils import (
    model_inference,
)
from aicsmlsegment.DataLoader3D.Universal_Loader import UniversalDataset

import random
import numpy as np
from glob import glob


SUPPORTED_LOSSES = [
    "Dice",
    "GeneralizedDice",
]
#     "MultiAuxillaryElementNLL",
#     "MultiTaskElementNLL",
#     "ElementAngularMSE",
#    "ElementNLL",
#     "WeightedCrossEntropy",
#     "PixelwiseCrossEntropy",
# ]

SUPPORTED_METRICS = ["default"]


def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    assert "loss" in config, "Could not find loss function configuration"
    loss_config = config["loss"]
    name = loss_config["name"]
    assert (
        name in SUPPORTED_LOSSES
    ), f"Invalid loss: {name}. Supported losses: {SUPPORTED_LOSSES}"

    # ignore_index = loss_config.get('ignore_index', None)

    if name == "Aux":
        return (
            CustomLosses.MultiAuxillaryElementNLLLoss(
                3, loss_config["loss_weight"], config["nclass"]
            ),
            True,
        )
    elif name == "ElementNLL":
        return CustomLosses.ElementNLLLoss(config["nclass"]), True
    elif name == "MultiAuxiliaryElementNLL":
        print("MultiAuxiliaryElementNLL Not implemented")
        quit()
    elif name == "MultiTaskElementNLL":
        return (
            CustomLosses.MultiTaskElementNLLLoss(
                loss_config["loss_weight"], loss_config["nclass"]
            ),
            True,
        )
    elif name == "ElementAngularMSE":
        return CustomLosses.ElementAngularMSE(), True
    elif name == "Dice":
        return CustomLosses.DiceLoss(), False
    elif name == "GeneralizedDice":
        return CustomLosses.GeneralizedDiceLoss(), False
    elif name == "WeightedCrossEntropy":
        return CustomLosses.WeightedCrossEntropyLoss(), False
    elif name == "PixelwiseCrossEntropy":
        return CustomLosses.PixelWiseCrossEntropyLoss(), True


class Monai_BasicUNet(pytorch_lightning.LightningModule):
    def __init__(self, config, train):
        super().__init__()
        self.model = BasicUNet(
            dimensions=len(config["size_in"]),
            in_channels=config["nchannel"],
            out_channels=config["nclass"],
            norm=Norm.BATCH,
        )
        self.args_inference = lambda: None
        if train:
            assert "loader" in config, "loader required"
            loader_config = config["loader"]
            self.datapath = loader_config["datafolder"]
            self.nworkers = loader_config["NumWorkers"]
            self.batchsize = loader_config["batch_size"]

            assert "validation" in config, "validation required"
            validation_config = config["validation"]
            self.leaveout = validation_config["leaveout"]

            self.lr = config["learning_rate"]
            self.weight_decay = config["weight_decay"]

            self.args_inference.OutputCh = validation_config["OutputCh"]

            self.loss_function, self.accepts_costmap = get_loss_criterion(config)
            self.metric = CustomMetrics.MeanIoU()

        else:
            self.args_inference.OutputCh = config["OutputCh"]

        self.args_inference.size_in = config["size_in"]
        self.args_inference.size_out = config["size_out"]
        self.args_inference.nclass = config["nclass"]

        device = config["device"]
        print(f"Sending the model to '{device}'")
        self.model = self.model.to(device)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        inputs = batch[0].cuda()
        targets = batch[1].cuda()
        outputs = self.forward(inputs)

        # select output channel
        outputs = outputs[:, self.args_inference.OutputCh, :, :, :]
        outputs = torch.unsqueeze(
            outputs, dim=1
        )  # add back in channel dimension to match targets
        if self.accepts_costmap:
            cmap = batch[2].cuda()
            loss = self.loss_function(outputs, targets, cmap)
        else:
            loss = self.loss_function(outputs, targets)

        mean_iou = self.metric(outputs, targets)

        return {"loss": loss, "iou": mean_iou}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_iou = torch.stack([x["iou"] for x in outputs]).mean()
        self.log("train_loss", avg_loss, prog_bar=True)
        self.log("IOU", avg_iou, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        input_img = batch[0].cuda()
        label = batch[1].cuda()

        outputs = model_inference(self.model, input_img, self.args_inference)
        outputs = outputs[:, self.args_inference.OutputCh, :, :, :]

        outputs = torch.unsqueeze(
            outputs, dim=1
        )  # add back in channel dimension to match label

        if self.accepts_costmap:
            costmap = batch[2].cuda()
            val_loss = self.loss_function(outputs, label, costmap)
        else:
            val_loss = self.loss_function(outputs, label)

        mean_iou = self.metric(outputs, label)

        return {"val_loss": val_loss, "val_iou": mean_iou}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_iou = torch.stack([x["val_iou"] for x in outputs]).mean()

        self.log("avg_val_loss", avg_loss, prog_bar=True)
        self.log("VAL_IOU", avg_iou, prog_bar=True)


class DataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        assert "loader" in config, "Could not find loader configuration"
        self.config = config
        self.loader_config = config["loader"]

        name = config["loader"]["name"]
        if name != "default" and name != "focus":
            print("other loaders are under construction")
            quit()

        self.transforms = []
        if "Transforms" in self.loader_config:
            self.transforms = self.loader_config["Transforms"]

    def prepare_data(self):
        pass

    def setup(self, stage):
        # load settings #
        config = self.config  # TODO, fix this

        # get validation and training filenames from input dir from config
        validation_config = config["validation"]
        loader_config = config["loader"]
        if validation_config["metric"] is not None:
            print("prepare the data ... ...")
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
                    train_idx = list(set(range(total_num)) - set(map(int, LeaveOut)))
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

        else:
            # TODO, update here
            print("need validation in config file")
            quit()

    def train_dataloader(self):
        loader_config = self.loader_config
        config = self.config
        train_set_loader = DataLoader(
            UniversalDataset(
                self.train_filenames,
                loader_config["PatchPerBuffer"],
                config["size_in"],
                config["size_out"],
                config["nchannel"],
                self.transforms,
            ),
            batch_size=loader_config["batch_size"],
            shuffle=True,
            num_workers=loader_config["NumWorkers"],
        )
        return train_set_loader

    def val_dataloader(self):
        loader_config = self.loader_config
        config = self.config
        val_set_loader = DataLoader(
            UniversalDataset(
                self.valid_filenames,
                loader_config["PatchPerBuffer"],
                config["size_in"],
                config["size_out"],
                config["nchannel"],
                [],  # no transforms for validation data
            ),
            batch_size=1,
            shuffle=False,
            num_workers=loader_config["NumWorkers"],
        )
        return val_set_loader
