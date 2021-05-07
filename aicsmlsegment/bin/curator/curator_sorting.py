#!/usr/bin/env python

import os
import sys
import logging
import argparse
import traceback
import importlib
import pathlib
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from glob import glob
from random import shuffle
from scipy import stats
from skimage.io import imsave
from skimage.draw import line, polygon
from scipy import ndimage as ndi

from aicssegmentation.core.utils import histogram_otsu
from aicsimageio import AICSImage, imread
from aicsimageio.writers import OmeTiffWriter
from aicsmlsegment.utils import input_normalization

matplotlib.use("TkAgg")

####################################################################################################
# global settings
button = 0
flag_done = False
pts = []
draw_img = None
draw_mask = None
draw_ax = None


log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(name)s - %(lineno)3d][%(levelname)s] %(message)s",
)
#
# Set the default log level for other modules used by this script
# logging.getLogger("labkey").setLevel(logging.ERROR)
# logging.getLogger("requests").setLevel(logging.WARNING)
# logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.INFO)
####################################################################################################


def quit_curation(event):
    if event.key == "q":
        exit()


def gt_sorting_callback(event):
    global button
    while 1:
        button = event.button
        if button == 3:
            print("You selected this image as GOOD")
            break
        elif button == 1:
            print("You selected this image as BAD")
            break
    plt.close()


def draw_polygons(event):
    global pts, draw_img, draw_ax, draw_mask
    if event.button == 1:
        if not (event.ydata == None or event.xdata == None):
            pts.append([event.xdata, event.ydata])
            if len(pts) > 1:
                rr, cc = line(
                    int(round(pts[-1][0])),
                    int(round(pts[-1][1])),
                    int(round(pts[-2][0])),
                    int(round(pts[-2][1])),
                )
                draw_img[cc, rr, :1] = 255
                draw_ax.set_data(draw_img)
                plt.draw()
    elif event.button == 3:
        if len(pts) > 2:
            # draw polygon
            pts_array = np.asarray(pts)
            rr, cc = polygon(pts_array[:, 0], pts_array[:, 1])
            draw_img[cc, rr, :1] = 255
            draw_ax.set_data(draw_img)
            draw_mask[cc, rr] = 1
            pts.clear()
            plt.draw()
        else:
            print("need at least three clicks before finishing annotation")


def quit_mask_drawing(event):
    if event.key == "d":
        plt.close()
    elif event.key == "q":
        exit()


def gt_sorting(raw_img, seg):
    global button

    bw = seg > 0
    z_profile = np.zeros((bw.shape[0],), dtype=int)
    for zz in range(bw.shape[0]):
        z_profile[zz] = np.count_nonzero(bw[zz, :, :])
    mid_frame = histogram_otsu(z_profile) * bw.shape[0]
    print("trying to find the best Z to display ...")
    print(f"the raw image has z profile {z_profile}")
    print(f"find best Z = {mid_frame}")
    mid_frame = int(round(mid_frame))

    # create 2x4 mosaic
    out = np.zeros((2 * raw_img.shape[1], 4 * raw_img.shape[2], 3), dtype=np.uint8)
    row_index = 0
    im = raw_img

    for cc in range(3):
        out[
            row_index * raw_img.shape[1] : (row_index + 1) * raw_img.shape[1],
            0 * raw_img.shape[2] : 1 * raw_img.shape[2],
            cc,
        ] = im[mid_frame - 4, :, :]
        out[
            row_index * raw_img.shape[1] : (row_index + 1) * raw_img.shape[1],
            1 * raw_img.shape[2] : 2 * raw_img.shape[2],
            cc,
        ] = im[mid_frame, :, :]
        out[
            row_index * raw_img.shape[1] : (row_index + 1) * raw_img.shape[1],
            2 * raw_img.shape[2] : 3 * raw_img.shape[2],
            cc,
        ] = im[mid_frame + 4, :, :]
        out[
            row_index * raw_img.shape[1] : (row_index + 1) * raw_img.shape[1],
            3 * raw_img.shape[2] : 4 * raw_img.shape[2],
            cc,
        ] = np.amax(im, axis=0)

    row_index = 1
    offset = 20
    im = seg + offset  # make it brighter
    im[im == offset] = 0
    im = im.astype(float) * (255 / im.max())
    im = np.round(im)
    im = im.astype(np.uint8)
    for cc in range(3):
        out[
            row_index * raw_img.shape[1] : (row_index + 1) * raw_img.shape[1],
            0 * raw_img.shape[2] : 1 * raw_img.shape[2],
            cc,
        ] = im[mid_frame - 4, :, :]
        out[
            row_index * raw_img.shape[1] : (row_index + 1) * raw_img.shape[1],
            1 * raw_img.shape[2] : 2 * raw_img.shape[2],
            cc,
        ] = im[mid_frame, :, :]
        out[
            row_index * raw_img.shape[1] : (row_index + 1) * raw_img.shape[1],
            2 * raw_img.shape[2] : 3 * raw_img.shape[2],
            cc,
        ] = im[mid_frame + 4, :, :]
        out[
            row_index * raw_img.shape[1] : (row_index + 1) * raw_img.shape[1],
            3 * raw_img.shape[2] : 4 * raw_img.shape[2],
            cc,
        ] = np.amax(im, axis=0)

    # display the image for good/bad inspection
    fig = plt.figure()
    figManager = plt.get_current_fig_manager()
    figManager.full_screen_toggle()
    ax = fig.add_subplot(111)
    ax.imshow(out)
    ax.set_title(
        "Interface for Sorting. Left click = BAD. Right click = GOOD \n"
        + "Press Q to quit the current curation (can be resumed later)\n"
        + "Columns left to right: 4 z slice below middle z slice, middle z slice, \n"
        + "4 z slice above middle z slice, max z projection \n"
        + "Top row: raw image; bottom row: segmentation. \n "
    )
    # plt.tight_layout()
    cid = fig.canvas.mpl_connect("button_press_event", gt_sorting_callback)
    cid2 = fig.canvas.mpl_connect("key_press_event", quit_curation)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    fig.canvas.mpl_disconnect(cid2)

    score = 0
    if button == 3:
        score = 1

    button = 0
    return score


def create_mask(raw_img, seg):
    global pts, draw_img, draw_mask, draw_ax
    bw = seg > 0
    z_profile = np.zeros((bw.shape[0],), dtype=int)
    for zz in range(bw.shape[0]):
        z_profile[zz] = np.count_nonzero(bw[zz, :, :])

    mid_frame = histogram_otsu(z_profile) * bw.shape[0]
    print("trying to find the best Z to display ...")
    print(f"the raw image has z profile {z_profile}")
    print(f"find best Z = {mid_frame}")
    mid_frame = int(round(mid_frame))

    offset = 20
    seg_label = seg + offset  # make it brighter
    seg_label[seg_label == offset] = 0
    seg_label = seg_label.astype(float) * (255 / seg_label.max())
    seg_label = np.round(seg_label)
    seg_label = seg_label.astype(np.uint8)

    img = np.zeros((raw_img.shape[1], 3 * raw_img.shape[2], 3), dtype=np.uint8)

    for cc in range(3):
        img[:, : raw_img.shape[2], cc] = raw_img[mid_frame, :, :]
        img[:, raw_img.shape[2] : 2 * raw_img.shape[2], cc] = seg_label[mid_frame, :, :]
        img[:, 2 * raw_img.shape[2] :, cc] = np.amax(seg_label, axis=0)

    draw_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    draw_img = img.copy()
    # display the image for good/bad inspection
    fig = plt.figure()
    figManager = plt.get_current_fig_manager()
    figManager.full_screen_toggle()
    ax = fig.add_subplot(111)
    ax.set_title(
        "Interface for annotating excluding mask. \n"
        + "Left: Middle z slice of raw. Middle: Middle z slice of segmentation. Right: Max z projection of segmentation \n"
        + "Please draw in the left panel \n"
        + "Left click to add a vertex; Right click to close the current polygon \n"
        + "Press D to finish annotating mask, Press Q to quit curation (can resume later)"
    )
    draw_ax = ax.imshow(img)
    cid = fig.canvas.mpl_connect("button_press_event", draw_polygons)
    cid2 = fig.canvas.mpl_connect("key_press_event", quit_mask_drawing)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    fig.canvas.mpl_disconnect(cid2)


class Args(object):
    """
    Use this to define command line arguments and use them later.

    For each argument do the following
    1. Create a member in __init__ before the self.__parse call.
    2. Provide a default value here.
    3. Then in p.add_argument, set the dest parameter to that variable name.

    See the debug parameter as an example.
    """

    def __init__(self, log_cmdline=True):
        self.debug = False
        self.output_dir = "." + os.sep
        self.struct_ch = 0
        self.xy = 0.108

        #
        self.__parse()
        #
        if self.debug:
            log.setLevel(logging.DEBUG)
            log.debug("-" * 80)
            self.show_info()
            log.debug("-" * 80)

    @staticmethod
    def __no_args_print_help(parser):
        """
        This is used to print out the help if no arguments are provided.
        Note:
        - You need to remove it's usage if your script truly doesn't want arguments.
        - It exits with 1 because it's an error if this is used in a script with no args.
          That's a non-interactive use scenario - typically you don't want help there.
        """
        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(1)

    def __parse(self):
        p = argparse.ArgumentParser()
        # Add arguments
        p.add_argument(
            "--d",
            "--debug",
            action="store_true",
            dest="debug",
            help="If set debug log output is enabled",
        )
        p.add_argument("--raw_path", required=True, help="path to raw images")
        p.add_argument("--data_type", required=True, help="the type of raw images")
        p.add_argument("--input_channel", default=0, type=int)
        p.add_argument("--seg_path", required=True, help="path to segmentation results")
        p.add_argument(
            "--train_path", required=True, help="path to output training data"
        )
        p.add_argument("--mask_path", help="[optional] the output directory for masks")
        p.add_argument(
            "--csv_name", required=True, help="the csv file to save the sorting results"
        )
        p.add_argument(
            "--Normalization",
            required=True,
            type=int,
            help="the normalization recipe to use",
        )

        self.__no_args_print_help(p)
        p.parse_args(namespace=self)

    def show_info(self):
        log.debug("Working Dir:")
        log.debug("\t{}".format(os.getcwd()))
        log.debug("Command Line:")
        log.debug("\t{}".format(" ".join(sys.argv)))
        log.debug("Args:")
        for (k, v) in self.__dict__.items():
            log.debug("\t{}: {}".format(k, v))


###############################################################################


class Executor(object):
    def __init__(self, args):

        if os.path.exists(args.csv_name):
            print(
                "the csv file for saving sorting results exists, sorting will be resumed"
            )
        else:
            print("no existing csv found, start a new sorting ")
            if not args.data_type.startswith("."):
                args.data_type = "." + args.data_type

            filenames = glob(args.raw_path + os.sep + "*" + args.data_type)
            filenames.sort()
            with open(args.csv_name, "w") as csvfile:
                filewriter = csv.writer(
                    csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
                )
                filewriter.writerow(["raw", "seg", "score", "mask"])
                for _, fn in enumerate(filenames):
                    seg_fn = (
                        args.seg_path
                        + os.sep
                        + os.path.basename(fn)[: -1 * len(args.data_type)]
                        + "_struct_segmentation.tiff"
                    )
                    assert os.path.exists(seg_fn)
                    filewriter.writerow([fn, seg_fn, None, None])

    def execute(self, args):

        global draw_mask
        # part 1: do sorting
        df = pd.read_csv(args.csv_name, index_col=False)

        for index, row in df.iterrows():

            if not np.isnan(row["score"]) and (row["score"] == 1 or row["score"] == 0):
                continue

            reader = AICSImage(row["raw"])
            struct_img = reader.get_image_data("ZYX", S=0, T=0, C=args.input_channel)
            struct_img[struct_img > 5000] = struct_img.min()  # adjust contrast
            raw_img = (struct_img - struct_img.min() + 1e-8) / (
                struct_img.max() - struct_img.min() + 1e-8
            )
            raw_img = 255 * raw_img
            raw_img = raw_img.astype(np.uint8)

            seg = np.squeeze(imread(row["seg"]))

            score = gt_sorting(raw_img, seg)
            if score == 1:
                df["score"].iloc[index] = 1
                need_mask = input(
                    "Do you need to add a mask for this image, enter y or n:  "
                )
                if need_mask == "y":
                    create_mask(raw_img, seg.astype(np.uint8))
                    mask_fn = (
                        args.mask_path
                        + os.sep
                        + os.path.basename(row["raw"])[:-5]
                        + "_mask.tiff"
                    )
                    crop_mask = np.zeros(seg.shape, dtype=np.uint8)
                    for zz in range(crop_mask.shape[0]):
                        crop_mask[zz, :, :] = draw_mask[
                            : crop_mask.shape[1], : crop_mask.shape[2]
                        ]

                    crop_mask = crop_mask.astype(np.uint8)
                    crop_mask[crop_mask > 0] = 255
                    with OmeTiffWriter(mask_fn) as writer:
                        writer.save(crop_mask)
                    df["mask"].iloc[index] = mask_fn
            else:
                df["score"].iloc[index] = 0

            df.to_csv(args.csv_name, index=False)

        #########################################
        # generate training data:
        #  (we want to do this step after "sorting"
        #  (is mainly because we want to get the sorting
        #  step as smooth as possible, even though
        #  this may waster i/o time on reloading images)
        # #######################################
        print("finish merging, start building the training data ...")

        existing_files = glob(args.train_path + os.sep + "img_*.ome.tif")
        print(len(existing_files))

        training_data_count = len(existing_files) // 3

        for index, row in df.iterrows():
            if row["score"] == 1:
                training_data_count += 1

                # load raw image
                reader = AICSImage(row["raw"])
                img = reader.get_image_data(
                    "CZYX", S=0, T=0, C=[args.input_channel]
                ).astype(np.float32)
                struct_img = input_normalization(img, args)
                struct_img = struct_img[0, :, :, :]

                # load segmentation gt
                seg = np.squeeze(imread(row["seg"])) > 0.01
                seg = seg.astype(np.uint8)
                seg[seg > 0] = 1

                cmap = np.ones(seg.shape, dtype=np.float32)
                if os.path.isfile(str(row["mask"])):
                    # load segmentation gt
                    mask = np.squeeze(imread(row["mask"]))
                    cmap[mask > 0] = 0

                with OmeTiffWriter(
                    args.train_path
                    + os.sep
                    + "img_"
                    + f"{training_data_count:03}"
                    + ".ome.tif"
                ) as writer:
                    writer.save(struct_img)

                with OmeTiffWriter(
                    args.train_path
                    + os.sep
                    + "img_"
                    + f"{training_data_count:03}"
                    + "_GT.ome.tif"
                ) as writer:
                    writer.save(seg)

                with OmeTiffWriter(
                    args.train_path
                    + os.sep
                    + "img_"
                    + f"{training_data_count:03}"
                    + "_CM.ome.tif"
                ) as writer:
                    writer.save(cmap)

        print("training data is ready")


def main():
    dbg = False
    try:
        args = Args()
        dbg = args.debug

        # Do your work here - preferably in a class or function,
        # passing in your args. E.g.
        exe = Executor(args)
        exe.execute(args)

    except Exception as e:
        log.error("=============================================")
        if dbg:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


if __name__ == "__main__":
    main()
