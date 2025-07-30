#!/usr/bin/env python

import os
import sys
import logging
import argparse
import traceback
import numpy as np
from glob import glob

from aicsimageio import AICSImage, imread
from aicsimageio.writers import OmeTiffWriter

from aicsmlsegment.utils import input_normalization

######################################################################################
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
######################################################################################


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
        - It exits with 1 because it's an error if this is used in a script with no args
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
            "--Normalization", default=0, help="the normalization method to use"
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
        pass

    def execute(self, args):

        if not args.data_type.startswith("."):
            args.data_type = "." + args.data_type

        filenames = glob(args.raw_path + os.sep + "*" + args.data_type)
        filenames.sort()

        existing_files = glob(args.train_path + os.sep + "img_*.ome.tif")
        print(len(existing_files))

        training_data_count = len(existing_files) // 3
        for _, fn in enumerate(filenames):

            training_data_count += 1

            # load raw
            reader = AICSImage(fn)
            struct_img = reader.get_image_data(
                "CZYX", S=0, T=0, C=[args.input_channel]
            ).astype(np.float32)
            struct_img = input_normalization(struct_img, args)

            # load seg
            seg_fn = (
                args.seg_path
                + os.sep
                + os.path.basename(fn)[: -1 * len(args.data_type)]
                + "_struct_segmentation.tiff"
            )
            seg = np.squeeze(imread(seg_fn)) > 0.01
            seg = seg.astype(np.uint8)
            seg[seg > 0] = 1

            # excluding mask
            cmap = np.ones(seg.shape, dtype=np.float32)
            mask_fn = (
                args.mask_path
                + os.sep
                + os.path.basename(fn)[: -1 * len(args.data_type)]
                + "_mask.tiff"
            )
            if os.path.isfile(mask_fn):
                mask = np.squeeze(imread(mask_fn))
                cmap[mask == 0] = 0

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
