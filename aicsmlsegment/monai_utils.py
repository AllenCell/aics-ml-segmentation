import pytorch_lightning
from torch.optim import Adam
import torch
import monai.losses as MonaiLosses
from typing import Dict


import aicsmlsegment.custom_loss as CustomLosses
import aicsmlsegment.custom_metrics as CustomMetrics
from aicsmlsegment.model_utils import (
    model_inference,
    apply_on_image,
)
from aicsmlsegment.DataLoader3D.Universal_Loader import (
    minmax,
    undo_resize,
)
from aicsmlsegment.utils import compute_iou

from monai.metrics import DiceMetric
import numpy as np
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
    "Aux",
    "MaskedDiceLoss",
    "PixelWiseCrossEntropyLoss",
    "MaskedDice+MaskedPixelwiseCrossEntropy",
    "ElementAngularMSELoss",
    "MaskedMSELoss",
    "MaskedDice+MaskedMSELoss",
    "MSELoss",
]


SUPPORTED_METRICS = [
    "default",
    "Dice",
]


def get_loss_criterion(config: Dict):
    """
    Returns the loss function based on provided configuration

    Parameters
    ----------
    config: Dict
        a top level configuration object containing the 'loss' key

    Return:
    -------------
    an instance of the loss function and whether it accepts a costmap and loss weights
    """
    name = config["loss"]["name"]

    # validate the name of the selected loss function
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
    elif name == "Aux":
        return (
            CustomLosses.MultiAuxillaryElementNLLLoss(
                len(config["model"]["nclass"]),
                config["loss"]["loss_weight"],
                config["model"]["nclass"],
            ),
            True,
            None,
        )
    elif name == "MaskedDiceLoss":
        return MonaiLosses.MaskedDiceLoss(sigmoid=True), True, None
    elif name == "PixelWiseCrossEntropyLoss":
        return CustomLosses.PixelWiseCrossEntropyLoss(), True, None
    elif name == "MaskedDice+MaskedPixelwiseCrossEntropy":
        return (
            CustomLosses.MaskedDiceCELoss(config["validation"]["OutputCh"]),
            True,
            None,
        )
    elif name == "ElementAngularMSELoss":
        return CustomLosses.ElementAngularMSELoss(), True, None
    elif name == "MaskedMSELoss":
        return CustomLosses.MaskedMSELoss(), True, None
    elif name == "MaskedDice+MaskedMSELoss":
        return (
            (
                CustomLosses.CombinedLoss(
                    CustomLosses.MaskedMSELoss(), CustomLosses.MaskedDiceLoss()
                ),
                True,
                None,
            ),
        )
    elif name == "MSELoss":
        return torch.nn.MSELoss(), False, None


def get_metric(config):
    """
    Returns the metric function based on provided configuration

    Parameters
    ----------
    config: Dict
        a top level configuration object containing the 'validation' key

    Return:
    -------------
    an instance of the validation metric function
    """
    validation_config = config["validation"]
    metric = validation_config["metric"]

    # validate the name of selected metric
    assert (
        metric in SUPPORTED_METRICS
    ), f"Invalid metric: {metric}. Supported metrics are: {SUPPORTED_METRICS}"

    if metric == "Dice":
        return DiceMetric
    elif metric == "default" or metric == "IOU":
        return CustomMetrics.MeanIoU()
    elif metric == "AveragePrecision":
        return CustomMetrics.AveragePrecision()


class Model(pytorch_lightning.LightningModule):
    def __init__(self, config, model_config, train):
        super().__init__()

        self.args_inference = {}

        self.model_name = config["model"]["name"]
        self.model_config = model_config

        if "unet_xy" in self.model_name:
            import importlib
            from aicsmlsegment.model_utils import weights_init as weights_init

            module = importlib.import_module("aicsmlsegment.Net3D." + self.model_name)
            init_args = {
                "in_channel": model_config["nchannel"],
                "n_classes": model_config["nclass"],
                "test_mode": not train,
            }
            if self.model_name == "sdunet":
                init_args["loss"] = config["loss"]["name"]

            if "zoom" in self.model_name:
                init_args["down_ratio"] = model_config.get("zoom_ratio", 3)

            model = getattr(module, "UNet3D")
            self.model = model(**init_args).apply(weights_init)

            self.args_inference["size_in"] = model_config["size_in"]
            self.args_inference["size_out"] = model_config["size_out"]
            self.args_inference["nclass"] = model_config["nclass"]

        else:  # monai model
            if self.model_name == "segresnetvae":
                from monai.networks.nets.segresnet import SegResNetVAE as model

                model_config["input_image_size"] = model_config["patch_size"]
            elif self.model_name == "extended_vnet":
                from aicsmlsegment.Net3D.vnet import VNet as model
            elif self.model_name == "extended_dynunet":
                from aicsmlsegment.Net3D.dynunet import DynUNet as model

            else:
                import importlib

                module = importlib.import_module(
                    "monai.networks.nets." + self.model_name
                )
                # deal with monai name scheme - module name != class name for networks
                net_name = [attr for attr in dir(module) if "Net" in attr][0]
                model = getattr(module, net_name)
            # monai model assumes same size for input and output
            self.args_inference["size_in"] = model_config["patch_size"]
            self.args_inference["size_out"] = model_config["patch_size"]

            del model_config["patch_size"]

            self.model = model(**model_config)

        self.config = config
        self.aggregate_img = None
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

        else:
            if config["RuntimeAug"] <= 0:
                self.args_inference["RuntimeAug"] = False
            else:
                self.args_inference["RuntimeAug"] = True
            self.args_inference["OutputCh"] = config["OutputCh"]
            self.args_inference["inference_batch_size"] = config["batch_size"]
            self.args_inference["mode"] = config["mode"]["name"]
            self.args_inference["Threshold"] = config["Threshold"]
            if config["large_image_resize"] is not None:
                self.aggregate_img = {}
                self.count_map = {}
        self.save_hyperparameters()

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
                    min_lr=0.0000001,
                )
                # monitoring metric must be specified
                return {
                    "optimizer": optims[0],
                    "lr_scheduler": scheduler,
                    "monitor": scheduler_params["monitor"],
                }
            elif scheduler_params["name"] == "1cycle":
                from torch.optim.lr_scheduler import OneCycleLR

                scheduler = OneCycleLR(
                    optims[0],
                    max_lr=scheduler_params["max_lr"],
                    total_steps=scheduler_params["total_steps"],
                    pct_start=scheduler_params["pct_start"],
                    verbose=scheduler_params["verbose"],
                )
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
        outputs = self(inputs)

        if ("unet_xy" in self.model_name and "sdu" not in self.model_name) or (
            "sdu" in self.model_name and self.config["loss"]["name"] == "Aux"
        ):  # old segmenter
            cmap = batch[2]
            loss = self.loss_function(outputs, targets, cmap)
            self.log(
                "epoch_train_loss",
                loss,
                sync_dist=True,
                prog_bar=True,
                on_epoch=True,
                on_step=False,
            )
            return {"loss": loss}
        if self.model_name == "segresnetvae":
            # segresnetvae forward returns an additional vae loss term
            outputs, vae_loss = outputs

        if (
            self.model_name == "extended_dynunet"
            and self.model_config["deep_supervision"]
        ):  # output is a stacked tensor all of same shape instead of a list
            outputs = torch.unbind(outputs, dim=1)

            if self.accepts_costmap:
                cmap = batch[2]
                cmap = torch.unsqueeze(cmap, dim=1)  # add channel dim
            loss = torch.zeros(1, device=self.device)
            for out in outputs:
                out = torch.unsqueeze(
                    out[:, self.args_inference["OutputCh"], :, :, :], dim=1
                )
                if self.accepts_costmap:
                    loss += self.loss_function(out, targets, cmap)
                else:
                    loss += self.loss_function(out, targets)
            self.log(
                "epoch_train_loss",
                loss / len(outputs),
                sync_dist=True,
                prog_bar=True,
                on_epoch=True,
                on_step=False,
            )
            return {"loss": loss}

        # focal loss requires > 1 channel
        if (
            "Focal" not in self.config["loss"]["name"]
            and "Pixel" not in self.config["loss"]["name"]
        ):
            # select output channel
            if isinstance(outputs, list):
                for i in range(len(outputs)):
                    outputs[i] = torch.unsqueeze(
                        outputs[i][:, self.args_inference["OutputCh"], :, :, :], dim=1
                    )
            else:
                outputs = torch.unsqueeze(
                    outputs[:, self.args_inference["OutputCh"], :, :, :], dim=1
                )  # add back in channel dimension to match targets

        if (
            isinstance(outputs, list) and self.model_name == "dynunet"
        ):  # average loss across deep supervision heads for dynunet w/ deep supervision
            if self.accepts_costmap:
                cmap = batch[2]
                cmap = torch.unsqueeze(cmap, dim=1)  # add channel dim
                loss = self.loss_function(outputs[0], targets, cmap)
            else:
                loss = self.loss_function(outputs[0], targets)

            for out in range(1, len(outputs)):
                # resize label
                x = torch.linspace(-1, 1, outputs[out].shape[-1], device=self.device)
                y = torch.linspace(-1, 1, outputs[out].shape[-2], device=self.device)
                z = torch.linspace(-1, 1, outputs[out].shape[-3], device=self.device)
                meshz, meshy, meshx = torch.meshgrid((z, y, x))
                grid = torch.stack((meshx, meshy, meshz), 3)
                grid = torch.stack(
                    [grid] * targets.shape[0]
                )  # one grid for each target in batch
                resize_target = torch.nn.functional.grid_sample(
                    targets, grid, align_corners=True
                )
                if self.accepts_costmap:
                    resize_costmap = torch.nn.functional.grid_sample(
                        cmap, grid, align_corners=True
                    )
                    loss += self.loss_function(
                        outputs[out], resize_target, resize_costmap
                    )
                else:
                    loss += self.loss_function(outputs[out], resize_target)

            self.log(
                "epoch_train_loss",
                loss / len(outputs),
                sync_dist=True,
                prog_bar=True,
                on_epoch=True,
                on_step=False,
            )
            return {"loss": loss}

        if self.accepts_costmap:
            cmap = batch[2]
            cmap = torch.unsqueeze(cmap, dim=1)  # add channel dim
            loss = self.loss_function(outputs, targets, cmap)
        else:
            if self.loss_weight is not None:
                loss = self.loss_function(outputs, targets, self.loss_weight)
            else:
                loss = self.loss_function(outputs, targets)
        # metric = self.metric(outputs, targets)

        if self.model_name == "segresnetvae":
            loss += 0.1 * vae_loss  # from https://arxiv.org/pdf/1810.11654.pdf

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
        if (
            "Focal" in self.config["loss"]["name"]
            or "Pixel" in self.config["loss"]["name"]
        ):
            extract = False
            squeeze = True

        outputs, vae_loss = model_inference(
            self.model,
            input_img,
            self.args_inference,
            squeeze=squeeze,
            extract_output_ch=extract,
            sigmoid=False,  # all loss functions accept logits
            model_name=self.model_name,
        )

        if "unet_xy" in self.model_name:
            costmap = batch[2]
            costmap = torch.unsqueeze(costmap, dim=1)
            # costmap = torch.ones(label.shape)
            val_loss = compute_iou(outputs > 0.5, label, costmap)
            self.log("val_iou", val_loss, sync_dist=True, on_epoch=True, prog_bar=True)
            return

        if self.accepts_costmap:
            costmap = batch[2]
            if self.config["loss"]["name"] != "ElementAngularMSELoss":
                costmap = torch.unsqueeze(costmap, dim=1)  # add channel
            val_loss = self.loss_function(outputs, label, costmap)
        else:
            val_loss = self.loss_function(outputs, label)

        self.log(
            "val_iou",
            compute_iou(outputs > 0.5, label, torch.ones(outputs.shape)),
            sync_dist=True,
            on_epoch=True,
            prog_bar=True,
        )
        # sync_dist on_epoch=True ensures that results will be averaged across gpus
        self.log(
            "val_loss",
            val_loss + 0.1 * vae_loss,  # from https://arxiv.org/pdf/1810.11654.pdf
            sync_dist=True,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        img = batch["img"]
        fn = batch["fn"][0]
        tt = batch["tt"][0]
        save_n_batches = batch["save_n_batches"].cpu().detach().numpy()[0]

        args_inference = self.args_inference

        sigmoid = True
        if "unet_xy" in self.model_name:
            sigmoid = False  # softmax is applied to outputs during apply_on_image

        to_numpy = True
        if self.aggregate_img is not None:
            to_numpy = False  # prevent excess gpu->cpu data transfer

        output_img, _ = apply_on_image(
            self.model,
            img,
            args_inference,
            squeeze=False,
            to_numpy=to_numpy,
            sigmoid=sigmoid,
            model_name=self.model_name,
            extract_output_ch=True,
        )

        if self.aggregate_img is not None:
            # initialize the aggregate img
            i, j, k = (
                batch["ijk"][0],
                batch["ijk"][1],
                batch["ijk"][2],
            )
            if fn not in self.aggregate_img:
                self.aggregate_img[fn] = torch.zeros(
                    batch["im_shape"], dtype=torch.float32, device=self.device
                )
                self.count_map[fn] = torch.zeros(
                    batch["im_shape"], dtype=torch.uint8, device=self.device
                )
            self.aggregate_img[fn][
                :,  # preserve all channels
                i : i + output_img.shape[2],
                j : j + output_img.shape[3],
                k : k + output_img.shape[4],
            ] += torch.squeeze(output_img, dim=0)

            self.count_map[fn][
                :,  # preserve all channels
                i : i + output_img.shape[2],
                j : j + output_img.shape[3],
                k : k + output_img.shape[4],
            ] += 1

        # only want to perform post-processing and saving once the aggregated image
        # is completeor we're not aggregating an image
        if (batch_idx + 1) % save_n_batches == 0:
            # prepare aggregate img for output
            if self.aggregate_img is not None:
                output_img = self.aggregate_img[fn] / self.count_map[fn]
                output_img = output_img.cpu().detach().numpy()
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
            if tt == -1:
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
            # self.log("", 0, on_step=False, on_epoch=False)
