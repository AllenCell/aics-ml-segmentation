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
import cv2

from aicssegmentation.core.utils import histogram_otsu
from aicsimageio import AICSImage, omeTifWriter
from aicsmlsegment.utils import input_normalization

matplotlib.use('TkAgg')

####################################################################################################
# global settings
ignore_img = False
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

def gt_sorting_callback(event):
    global button
    while(1):
        button = event.button
        if button == 3:
            print('You selected this image as GOOD')
            break
        elif button == 1:
            print('You selected this image as BAD')
            break
    plt.close()

def draw_polygons(event):
    global pts, draw_img, draw_ax, draw_mask
    if event.button == 1:
        if not (event.ydata == None or event.xdata == None):
            pts.append([event.xdata,event.ydata])
            if len(pts)>1:
                rr, cc = line(int(round(pts[-1][0])), int(round(pts[-1][1])), int(round(pts[-2][0])), int(round(pts[-2][1])) )
                draw_img[cc,rr,:1]=255
            draw_ax.imshow(draw_img)
            plt.show()
    elif event.button == 3:
        if len(pts)>2:
            # draw polygon
            pts_array = np.asarray(pts)
            rr, cc = polygon(pts_array[:,0], pts_array[:,1])
            draw_img[cc,rr,:1]=255
            draw_ax.imshow(draw_img)
            draw_mask[cc,rr]=1
            pts.clear()
            plt.show()
        else:
            print('need at least three clicks before finishing annotation')

def quit_mask_drawing(event):
    global ignore_img
    if event.key == 'd':
        plt.close()
    elif event.key == 'b':
        ignore_img = True
        plt.close()


def create_merge_mask(raw_img, seg1, seg2):
    global pts, draw_img, draw_mask, draw_ax
   
    offset = 20
    seg1_label = seg1 + offset # make it brighter
    seg1_label[seg1_label==offset]=0
    seg1_label = seg1_label.astype(float) * (255/seg1_label.max())
    seg1_label = np.round(seg1_label)
    seg1_label = seg1_label.astype(np.uint8)

    offset = 25
    seg2_label = seg2 + offset # make it brighter
    seg2_label[seg2_label==offset]=0
    seg2_label = seg2_label.astype(float) * (255/seg2_label.max())
    seg2_label = np.round(seg2_label)
    seg2_label = seg2_label.astype(np.uint8)


    bw = seg1>0
    z_profile = np.zeros((bw.shape[0],),dtype=int)
    for zz in range(bw.shape[0]):
        z_profile[zz] = np.count_nonzero(bw[zz,:,:])
    mid_frame = round(histogram_otsu(z_profile)*bw.shape[0]).astype(int)

    img = np.zeros((2*raw_img.shape[1], 3*raw_img.shape[2], 3),dtype=np.uint8)

    row_index = 0

    for cc in range(3):
        img[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], :raw_img.shape[2], cc]=np.amax(raw_img, axis=0)
        img[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], raw_img.shape[2]:2*raw_img.shape[2], cc]=np.amax(seg1_label, axis=0)
        img[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 2*raw_img.shape[2]:, cc]=np.amax(seg2_label, axis=0)
    
    row_index = 1
    for cc in range(3):
        img[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], :raw_img.shape[2], cc]=raw_img[mid_frame,:,:]
        img[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], raw_img.shape[2]:2*raw_img.shape[2], cc]=seg1_label[mid_frame,:,:]
        img[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 2*raw_img.shape[2]:, cc]=seg2_label[mid_frame,:,:]

    draw_mask = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    draw_img = img.copy()
    # display the image for good/bad inspection
    fig = plt.figure()
    figManager = plt.get_current_fig_manager() 
    figManager.full_screen_toggle() 
    ax = fig.add_subplot(111)
    ax.imshow(img)
    draw_ax = ax
    cid = fig.canvas.mpl_connect('button_press_event', draw_polygons)
    cid2 = fig.canvas.mpl_connect('key_press_event', quit_mask_drawing)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

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
        p.add_argument('--data_type', required=True, help='the type of raw images')
        p.add_argument('--input_channel', default=0, type=int)
        p.add_argument('--seg1_path', required=True, help='path to segmentation results v1')
        p.add_argument('--seg2_path', required=True, help='path to segmentation results v2')
        p.add_argument('--train_path', required=True, help='path to output training data')
        p.add_argument('--mask_path', help='[optional] the output directory for merging masks')
        p.add_argument('--ex_mask_path', help='[optional] the output directory for excluding masks')
        p.add_argument('--csv_name', required=True, help='the csv file to save the sorting results')
        p.add_argument('--Normalization', required=True, type=int, help='the normalization recipe to use')

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
            print('the csv file for saving sorting results exists, sorting will be resumed')
        else:
            print('no existing csv found, start a new sorting ')
            if not args.data_type.startswith('.'):
                args.data_type = '.' + args.data_type

            filenames = glob(args.raw_path + '/*' + args.data_type)
            filenames.sort()
            with open(args.csv_name, 'w') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow(['raw','seg1','seg2','score','merging_mask','excluding_mask'])
                for _, fn in enumerate(filenames):
                    seg1_fn = args.seg1_path + os.sep + os.path.basename(fn)[:-1*len(args.data_type)] + '_struct_segmentation.tiff'
                    seg2_fn = args.seg2_path + os.sep + os.path.basename(fn)[:-1*len(args.data_type)] + '_struct_segmentation.tiff'
                    assert os.path.exists(seg1_fn)
                    assert os.path.exists(seg2_fn)
                    filewriter.writerow([fn, seg1_fn , seg2_fn , None, None, None])

    def execute(self, args):

        global draw_mask, ignore_img
        # part 1: do sorting
        df = pd.read_csv(args.csv_name)

        for index, row in df.iterrows():

            if not np.isnan(row['score']) and (row['score']==1 or row['score']==0):
                continue

            reader = AICSImage(row['raw'])
            im_full = reader.data
            struct_img = im_full[0,args.input_channel,:,:,:]
            raw_img = (struct_img- struct_img.min() + 1e-8)/(struct_img.max() - struct_img.min() + 1e-8)
            raw_img = 255 * raw_img
            raw_img = raw_img.astype(np.uint8)

            reader_seg1 = AICSImage(row['seg1'])
            im_seg1_full = reader_seg1.data
            assert im_seg1_full.shape[0]==1
            assert im_seg1_full.shape[1]==1 or im_seg1_full.shape[2]==1
            if im_seg1_full.shape[1]==1:
                seg1 = im_seg1_full[0,0,:,:,:]>0.1
            else:
                seg1 = im_seg1_full[0,:,0,:,:]>0.1

            reader_seg2 = AICSImage(row['seg2'])
            im_seg2_full = reader_seg2.data
            assert im_seg2_full.shape[0]==1 
            assert im_seg2_full.shape[1]==1 or im_seg2_full.shape[2]==1
            if im_seg2_full.shape[1]==1:
                seg2 = im_seg2_full[0,0,:,:,:]>0
            else:
                seg2 = im_seg2_full[0,:,0,:,:]>0
            
            create_merge_mask(raw_img, seg1.astype(np.uint8), seg2.astype(np.uint8))

            if ignore_img:
                df['score'].iloc[index]=0
            else:
                df['score'].iloc[index]=1

                mask_fn = args.mask_path + os.sep + os.path.basename(row['raw'])[:-5] + '_mask.tiff'
                crop_mask = np.zeros(seg1.shape, dtype=np.uint8)
                for zz in range(crop_mask.shape[0]):
                    crop_mask[zz,:,:] = draw_mask[:crop_mask.shape[1],:crop_mask.shape[2]]

                crop_mask = crop_mask.astype(np.uint8)
                crop_mask[crop_mask>0]=255
                writer = omeTifWriter.OmeTifWriter(mask_fn)
                writer.save(crop_mask)
                df['merging_mask'].iloc[index]=mask_fn

                need_mask = input('Do you need to add an excluding mask for this image, enter y or n:  ')
                if need_mask == 'y':
                    create_merge_mask(raw_img, seg1.astype(np.uint8), seg2.astype(np.uint8))

                    mask_fn = args.ex_mask_path + os.sep + os.path.basename(row['raw'])[:-5] + '_mask.tiff'
                    crop_mask = np.zeros(seg1.shape, dtype=np.uint8)
                    for zz in range(crop_mask.shape[0]):
                        crop_mask[zz,:,:] = draw_mask[:crop_mask.shape[1],:crop_mask.shape[2]]

                    crop_mask = crop_mask.astype(np.uint8)
                    crop_mask[crop_mask>0]=255
                    writer = omeTifWriter.OmeTifWriter(mask_fn)
                    writer.save(crop_mask)
                    df['excluding_mask'].iloc[index]=mask_fn


            df.to_csv(args.csv_name)
            

        #########################################
        # generate training data:
        #  (we want to do this step after "sorting"
        #  (is mainly because we want to get the sorting 
        #  step as smooth as possible, even though
        #  this may waster i/o time on reloading images)
        # #######################################
        print('finish merging, start building the training data ...')
        training_data_count = 0
        for index, row in df.iterrows():
            if row['score']==1:
                training_data_count += 1

                # load raw image
                reader = AICSImage(row['raw'])
                img = reader.data.astype(np.float32)
                struct_img = input_normalization(img[0,[args.input_channel],:,:,:], args)
                struct_img= struct_img[0,:,:,:]

                reader_seg1 = AICSImage(row['seg1'])
                im_seg1_full = reader_seg1.data
                assert im_seg1_full.shape[0]==1
                assert im_seg1_full.shape[1]==1 or im_seg1_full.shape[2]==1
                if im_seg1_full.shape[1]==1:
                    seg1 = im_seg1_full[0,0,:,:,:]>0.1
                else:
                    seg1 = im_seg1_full[0,:,0,:,:]>0.1

                reader_seg2 = AICSImage(row['seg2'])
                im_seg2_full = reader_seg2.data
                assert im_seg2_full.shape[0]==1 
                assert im_seg2_full.shape[1]==1 or im_seg2_full.shape[2]==1
                if im_seg2_full.shape[1]==1:
                    seg2 = im_seg2_full[0,0,:,:,:]>0
                else:
                    seg2 = im_seg2_full[0,:,0,:,:]>0

                if os.path.isfile(str(row['merging_mask'])):
                    reader = AICSImage(row['merging_mask'])
                    img = reader.data
                    assert img.shape[0]==1 and img.shape[1]==1
                    mask = img[0,0,:,:,:]>0
                    seg1[mask>0]=0
                    seg2[mask==0]=0
                    seg1 = np.logical_or(seg1,seg2)
                
                cmap = np.ones(seg1.shape, dtype=np.float32)
                if os.path.isfile(str(row['excluding_mask'])):
                    reader = AICSImage(row['excluding_mask'])
                    img = reader.data
                    assert img.shape[0]==1 and img.shape[1]==1
                    ex_mask = img[0,0,:,:,:]>0
                    cmap[ex_mask>0]=0
 
                
                writer = omeTifWriter.OmeTifWriter(args.train_path + os.sep + 'img_' + f'{training_data_count:03}' + '.ome.tif')
                writer.save(struct_img)

                seg1 = seg1.astype(np.uint8)
                seg1[seg1>0]=255
                writer = omeTifWriter.OmeTifWriter(args.train_path + os.sep + 'img_' + f'{training_data_count:03}' + '_GT.ome.tif')
                writer.save(seg1)

                writer = omeTifWriter.OmeTifWriter(args.train_path + os.sep + 'img_' + f'{training_data_count:03}' + '_CM.ome.tif')
                writer.save(cmap)
        print('training data is ready')


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

