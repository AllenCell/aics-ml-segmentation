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

from aicsmlsegment.monai_utils import Monai_BasicUNet


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
    model.to("cuda")
    print("model loaded")
    # extract the parameters for running the model inference
    args_inference = model.args_inference
    if config["RuntimeAug"] <= 0:
        args_inference["RuntimeAug"] = False
    else:
        args_inference["RuntimeAug"] = True

    # run
    inf_config = config["mode"]
    if inf_config["name"] == "file":
        fn = inf_config["InputFile"]
        data_reader = AICSImage(fn)

        if inf_config["timelapse"]:
            assert data_reader.shape[1] > 1, "not a timelapse, check your data"

            for tt in range(data_reader.shape[1]):
                # Assume:  dimensions = TCZYX
                img = data_reader.get_image_data(
                    "CZYX", S=0, T=tt, C=config["InputCh"]
                ).astype(float)
                img = image_normalization(img, config["Normalization"])
                img = resize(img, config)

                # apply the model
                output_img = apply_on_image(
                    model, img.cuda(), args_inference, squeeze=False, to_numpy=True
                )

                # extract the result and write the output
                out = output_img[:, args_inference["OutputCh"], :, :, :]
                out = minmax(out)
                out = undo_resize(out, config)

                if config["Threshold"] > 0:
                    out = out > config["Threshold"]
                    out = out.astype(np.uint8)
                    out[out > 0] = 255
                imsave(
                    config["OutputDir"]
                    + os.sep
                    + pathlib.PurePosixPath(fn).stem
                    + "_T_"
                    + f"{tt:03}"
                    + "_struct_segmentation.tiff",
                    out,
                )

        else:
            img = data_reader.get_image_data(
                "CZYX", S=0, T=0, C=config["InputCh"]
            ).astype(float)
            print("Image read")
            img = image_normalization(img, config["Normalization"])
            print("Image normalized")

            img = resize(img, config)

            print("Image resized")
            print("applying model", end=" ")
            # apply the model
            output_img = apply_on_image(
                model, img, args_inference, squeeze=False, to_numpy=True
            )
            print("done")
            # extract the result and write the output
            out = output_img[:, args_inference["OutputCh"], :, :, :]
            out = minmax(out)
            out = undo_resize(out, config)
            print("output resized")
            if config["Threshold"] > 0:
                out = out > config["Threshold"]
                out = out.astype(np.uint8)
                out[out > 0] = 255
            print("thresholded")
            imsave(
                config["OutputDir"]
                + os.sep
                + pathlib.PurePosixPath(fn).stem
                + "_struct_segmentation.tiff",
                out,
            )

            print(f"Image {fn} has been segmented")

    elif inf_config["name"] == "folder":
        from glob import glob

        filenames = glob(inf_config["InputDir"] + "/*" + inf_config["DataType"])
        filenames.sort()  # (reverse=True)
        print("files to be processed:")
        print(filenames)

        for _, fn in enumerate(filenames):

            # load data
            data_reader = AICSImage(fn)
            img = data_reader.get_image_data(
                "CZYX", S=0, T=0, C=config["InputCh"]
            ).astype(float)

            img = resize(img, config, min_max=False)

            img = image_normalization(img, config["Normalization"])

            # apply the model
            output_img = apply_on_image(
                model, img, args_inference, squeeze=False, to_numpy=True
            )
            # extract the result and write the output
            if config["Threshold"] < 0:
                out = output_img[:, args_inference["OutputCh"], :, :, :]
                out = minmax(out)

                out = undo_resize(out, config)
                out = minmax(out)
            else:
                out = remove_small_objects(
                    output_img[:, args_inference["OutputCh"], :, :, :]
                    > config["Threshold"],
                    min_size=2,
                    connectivity=1,
                )
                out = out.astype(np.uint8)
                out[out > 0] = 255
            imsave(
                config["OutputDir"]
                + os.sep
                + pathlib.PurePosixPath(fn).stem
                + "_struct_segmentation.tiff",
                out,
            )

            print(f"Image {fn} has been segmented")


if __name__ == "__main__":

    main()
