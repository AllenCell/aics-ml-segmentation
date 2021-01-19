import pytorch_lightning
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch

from monai.networks.layers import Norm, Act
from monai.networks.nets import BasicUNet
import monai.losses as MonaiLosses

import aicsmlsegment.custom_loss as CustomLosses
import aicsmlsegment.custom_metrics as CustomMetrics
from aicsmlsegment.model_utils import (
    model_inference,
)
from aicsmlsegment.DataLoader3D.Universal_Loader import UniversalDataset
from monai.metrics import DiceMetric
import random
import numpy as np
from glob import glob


SUPPORTED_LOSSES = [
    "Dice",
    "GeneralizedDice",
]
#     "MultiTaskElementNLL",
# "ElementNLL",
#       "ElementAngularMSE",
#     "WeightedCrossEntropy",
#     "PixelwiseCrossEntropy",
# ]

SUPPORTED_METRICS = [
    "default",
    "Dice",
]
# "IOU", "AveragePrecision"]

MODEL_PARAMETERS = {
    "BasicUNet": {
        "Optional": [
            "features",
            "act",
            "norm",
            "dropout",
        ],
        "Required": ["dimensions", "in_channels", "out_channels"],
    }
}

ACTIVATIONS = {
    "LeakyReLU": Act.LEAKYRELU,
    "PReLU": Act.PRELU,
    "ReLU": Act.RELU,
    "ReLU6": Act.RELU6,
}

NORMALIZATIONS = {
    "batch": Norm.BATCH,
    "instance": Norm.INSTANCE,
}


def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function and whether it accepts a costmap
    """
    assert "loss" in config, "Could not find loss function configuration"
    loss_config = config["loss"]
    name = loss_config["name"]
    weight = loss_config["loss_weight"]
    assert (
        name in SUPPORTED_LOSSES
    ), f"Invalid loss: {name}. Supported losses: {SUPPORTED_LOSSES}"

    if name == "ElementNLL":
        return CustomLosses.ElementNLLLoss(config["nclass"]), False, weight
    elif name == "MultiTaskElementNLL":
        return (
            CustomLosses.MultiTaskElementNLLLoss(
                loss_config["loss_weight"], loss_config["nclass"]
            ),
            True,
            weight,
        )
    elif name == "ElementAngularMSE":
        return CustomLosses.ElementAngularMSELoss(), True, weight
    elif name == "Dice":
        return MonaiLosses.DiceLoss(sigmoid=True), False, None
        # return CustomLosses.DiceLoss(), False
    elif name == "GeneralizedDice":
        # return CustomLosses.GeneralizedDiceLoss(), False
        return MonaiLosses.GeneralizedDiceLoss(sigmoid=True), False, None
    elif name == "WeightedCrossEntropy":
        return CustomLosses.WeightedCrossEntropyLoss(), False, weight
    elif name == "PixelwiseCrossEntropy":
        return CustomLosses.PixelWiseCrossEntropyLoss(), True, weight


def get_metric(config):
    assert "validation" in config, "Could not find validation information"
    validation_config = config["validation"]
    assert "metric" in validation_config, "Could not find validation metric"
    metric = validation_config["metric"]

    assert (
        metric in SUPPORTED_METRICS
    ), f"Invalid metric: {metric}. Supported metrics are: {SUPPORTED_METRICS}"

    if metric == "Dice":
        return DiceMetric
        # return CustomMetrics.DiceCoefficient()
    elif metric == "default" or metric == "IOU":
        return CustomMetrics.MeanIoU()
    elif metric == "AveragePrecision":
        return CustomMetrics.AveragePrecision()


def get_model_configurations(config):
    assert "model" in config, "Model specifications must be included"
    model_config = config["model"]

    model_parameters = {}

    all_parameters = MODEL_PARAMETERS[model_config["name"]]

    # allow users to overwrite specific parameters
    for param in all_parameters["Optional"]:
        # if optional parameters are not specified, skip them to use monai defaults
        if param in model_config and not model_config[param] is None:
            if param == "norm":
                try:
                    model_parameters[param] = NORMALIZATIONS[model_config[param]]
                except KeyError:
                    print(f"{model_config[param]} is not an acceptable normalization.")
                    quit()
            elif param == "act":
                try:
                    model_parameters[param] = ACTIVATIONS[model_config[param]]
                except KeyError:
                    print(f"{model_config[param]} is not an acceptable activation.")
                    quit()
            else:
                model_parameters[param] = model_config[param]
    # find parameters that must be included
    for param in all_parameters["Required"]:
        assert (
            param in model_config
        ), f"{param} is required for model {model_config['name']}"
        model_parameters[param] = model_config[param]

    print(model_parameters)
    return model_parameters


class Monai_BasicUNet(pytorch_lightning.LightningModule):
    def __init__(self, config, train):
        super().__init__()
        model_configuration = get_model_configurations(config)
        self.model = BasicUNet(**model_configuration)

        print(self.model)
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

            (
                self.loss_function,
                self.accepts_costmap,
                self.loss_weight,
            ) = get_loss_criterion(config)
            self.metric = get_metric(config)

        else:
            self.args_inference.OutputCh = config["OutputCh"]

        self.args_inference.size_in = config["model"]["patch_size"]
        self.args_inference.size_out = config["model"]["patch_size"]
        self.args_inference.nclass = config["model"]["out_channels"]

        device = config["device"]
        print(f"Sending the model to '{device}'")
        self.model = self.model.to(device)

    def forward(self, x):
        """
        returns raw predictions
        """
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
            if self.loss_weight is not None:
                loss = self.loss_function(outputs, targets, self.loss_weight)
            else:
                loss = self.loss_function(outputs, targets)

        # metric = self.metric(outputs, targets)

        return {"loss": loss}  # , "metric": metric}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        # avg_metric = torch.stack([x["metric"] for x in outputs]).mean()
        self.log("Epoch_train_loss", avg_loss, prog_bar=True)
        # self.log("Epoch_train_metric", avg_metric, prog_bar=True)

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

        # val_metric = self.metric(outputs, label)

        return {"val_loss": val_loss}  # , "val_metric": val_metric}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        # avg_metric = torch.stack([x["val_metric"] for x in outputs]).mean()

        self.log("Epoch_val_loss", avg_loss, prog_bar=True)
        # self.log("Epoch_val_metric", avg_metric, prog_bar=True)


class DataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        assert "loader" in config, "Could not find loader configuration"
        self.config = config
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
        # load settings #
        config = self.config  # TODO, fix this

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
            print("Done.")

        else:
            # TODO, update here
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
                model_config["patch_size"],
                model_config["in_channels"],
                [],  # no transforms for validation data
                patchize=False,  # validate on entire image
            ),
            batch_size=1,
            shuffle=False,
            num_workers=loader_config["NumWorkers"],
        )
        return val_set_loader
