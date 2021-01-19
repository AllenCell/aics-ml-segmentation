#!/usr/bin/env python

import argparse
import os
import pathlib
import numpy as np

from skimage.morphology import remove_small_objects
from skimage.io import imsave
from aicsimageio import AICSImage
from scipy.ndimage import zoom

from aicsmlsegment.utils import (
    load_config,
    image_normalization,
)
from aicsmlsegment.model_utils import (
    apply_on_image,
)

from aicsmlsegment.monai_utils import Monai_BasicUNet, DataModule
import pytorch_lightning


def minmax(img):
    return (img - img.min()) / (img.max() - img.min())


def resize(img, config, min_max=False):
    if len(config["ResizeRatio"]) > 0 and config["ResizeRatio"] != [
        1.0,
        1.0,
        1.0,
    ]:
        # don't resize if resize ratio is all 1s
        # note that struct_img is only a view of img, so changes made on struct_img also affects img
        img = zoom(
            img,
            (
                1,
                config["ResizeRatio"][0],
                config["ResizeRatio"][1],
                config["ResizeRatio"][2],
            ),
            order=2,
            mode="reflect",
        )
        if min_max:
            for ch_idx in range(img.shape[0]):
                struct_img = img[ch_idx, :, :, :]
                img[ch_idx, :, :, :] = minmax(struct_img)
    return img


def undo_resize(img, config):
    if len(config["ResizeRatio"]) > 0 and config["ResizeRatio"] != [1.0, 1.0, 1.0]:
        img = zoom(
            img,
            (
                1.0,
                1 / config["ResizeRatio"][0],
                1 / config["ResizeRatio"][1],
                1 / config["ResizeRatio"][2],
            ),
            order=2,
            mode="reflect",
        )
    return img.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    # load the trained model instance
    model_path = config["model_path"]
    print(f"Loading model from {model_path}...")
    model = Monai_BasicUNet.load_from_checkpoint(model_path, config=config, train=False)

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

    trainer = pytorch_lightning.Trainer(
        gpus=gpu_config,
        num_sanity_val_steps=0,
        distributed_backend=accelerator,
    )
    data_module = DataModule(config, train=False)

    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":

    main()
