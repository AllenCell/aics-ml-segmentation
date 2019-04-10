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
from glob import glob
from random import shuffle
from scipy import stats
from skimage.io import imsave
from skimage.draw import line, polygon
from scipy import ndimage as ndi

from aicssegmentation.core.utils import histogram_otsu
from aicsimageio import AICSImage, omeTifWriter

from aicsmlsegment.utils import input_normalization

####################################################################################################
# global settings
button = 0
flag_done = False
pts = []
draw_img = None
draw_mask = None
draw_ax = None


log = logging.getLogger()
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(name)s - %(lineno)3d][%(levelname)s] %(message)s')
#
# Set the default log level for other modules used by this script
# logging.getLogger("labkey").setLevel(logging.ERROR)
# logging.getLogger("requests").setLevel(logging.WARNING)
# logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.INFO)
####################################################################################################

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
        self.output_dir = './'
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
        p.add_argument('--d', '--debug', action='store_true', dest='debug',
                       help='If set debug log output is enabled')
        p.add_argument('--raw_path', required=True, help='path to raw images')
        p.add_argument('--input_channel', default=0, type=int)
        p.add_argument('--seg_path', required=True, help='path to segmentation results')
        p.add_argument('--train_path', required=True, help='path to output training data')
        p.add_argument('--mask_path', help='[optional] the output directory for masks')
        p.add_argument('--Normalization', default=0, help='the normalization method to use')

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


'''
parent_path = '/allen/aics/assay-dev/Analysis/labelfree_predictions/'
sample_csv = '/allen/aics/assay-dev/Analysis/labelfree_predictions/dna_samples.csv'
df_sample = pd.read_csv(sample_csv)
map_csv = parent_path + 'labelfree_results_original_resized.csv'
df = pd.read_csv(map_csv)
'''
###############################################################################

class Executor(object):

    def __init__(self, args):
        pass

    def execute(self, args):

        df = pd.read_csv(args.csv_name)
        training_data_count = 0
        for _, row in df.iterrows():
            if row['score']==1:
                training_data_count += 1
                input_channel = row['raw_input_ch']
                # load raw image
                reader = AICSImage(row['raw'])
                img = reader.data.astype(np.float32)
                struct_img = img[0,[input_channel],:,:,:].copy()

                struct_img = input_normalization(struct_img, args)

                #img_smooth = ndi.gaussian_filter(img_raw, sigma=50, mode='nearest', truncate=3.0)
                #img_smooth_sub = img_raw - img_smooth
                #struct_img = (img_smooth_sub - img_smooth_sub.min())/(img_smooth_sub.max()-img_smooth_sub.min())

                # load segmentation gt
                reader = AICSImage(row['seg'])
                img = reader.data
                assert img.shape[0]==1 and img.shape[1]==1
                seg = img[0,0,:,:,:]>0
                seg = seg.astype(np.uint8)
                seg[seg>0]=1

                cmap = np.ones(seg.shape, dtype=np.float32)
                if os.path.isfile(str(row['mask'])):
                    # load segmentation gt
                    reader = AICSImage(row['mask'])
                    img = reader.data
                    assert img.shape[0]==1 and img.shape[1]==1
                    mask = img[0,0,:,:,:]
                    cmap[mask>0]=0

                writer = omeTifWriter.OmeTifWriter(args.train_path + os.sep + 'img_' + f'{training_data_count:03}' + '.ome.tif')
                writer.save(struct_img)

                writer = omeTifWriter.OmeTifWriter(args.train_path + os.sep + 'img_' + f'{training_data_count:03}' + '_GT.ome.tif')
                writer.save(seg)

                writer = omeTifWriter.OmeTifWriter(args.train_path + os.sep + 'img_' + f'{training_data_count:03}' + '_CM.ome.tif')
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

'''
sample_fn = '/allen/aics/assay-dev/Segmentation/DeepLearning/for_april_2019_release/LMNB1_classic_workflow_segmentation_iter_1/image_040_struct_segmentation.tiff'
mpx, mpy = -1, -1



# Create a black image, a window and bind the function to window
reader = AICSImage(sample_fn)
data = reader.data 
img = np.amax(data[0,0,:,:,:]>0, axis=0)
img = img.astype(np.uint8)
img[img>0]=255
img = np.stack([img,img,img],axis=2)
mask = np.zeros_like(img)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_polygons)

while(1):
    cv2.imshow('image',img)
    if len(pts)>0:
        cv2.polylines(img, [np.asarray(pts).astype(np.int32)], False, (0,255,255))
    k = cv2.waitKey(50)
    if k == 27:
        break
    elif k == ord('d'):
        pts.append(pts[0])
        cv2.fillPoly(img, [np.asarray(pts).astype(np.int32)], (0,255,255))
        cv2.fillPoly(mask, [np.asarray(pts).astype(np.int32)], (255,0,0))
        pts=[]


print(mask.shape)
mask = mask[:,:,0]
imsave('test.tiff',mask)
    
cv2.destroyAllWindows()
'''