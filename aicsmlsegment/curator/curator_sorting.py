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
    if event.key == 'd':
        plt.close()

def gt_sorting(raw_img, seg):
    global button

    bw = seg>0
    z_profile = np.zeros((bw.shape[0],),dtype=int)
    for zz in range(bw.shape[0]):
        z_profile[zz] = np.count_nonzero(bw[zz,:,:])
    mid_frame = round(histogram_otsu(z_profile)*bw.shape[0]).astype(int)

    #create 2x5 mosaic
    out = np.zeros((2*raw_img.shape[1], 4*raw_img.shape[2], 3),dtype=np.uint8)
    row_index=0
    im = raw_img
    
    for cc in range(3):
        out[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 0*raw_img.shape[2]:1*raw_img.shape[2], cc]=im[mid_frame-4,:,:]
        out[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 1*raw_img.shape[2]:2*raw_img.shape[2], cc]=im[mid_frame,:,:]
        out[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 2*raw_img.shape[2]:3*raw_img.shape[2], cc]=im[mid_frame+4,:,:]
        out[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 3*raw_img.shape[2]:4*raw_img.shape[2], cc]=np.amax(im, axis=0)
        '''
        out[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 0*raw_img.shape[2]:1*raw_img.shape[2], cc]=im[mid_frame-8,:,:]
        out[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 1*raw_img.shape[2]:2*raw_img.shape[2], cc]=im[mid_frame-4,:,:]
        out[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 2*raw_img.shape[2]:3*raw_img.shape[2], cc]=im[mid_frame,:,:]
        out[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 3*raw_img.shape[2]:4*raw_img.shape[2], cc]=im[mid_frame+4,:,:]
        out[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 4*raw_img.shape[2]:5*raw_img.shape[2], cc]=im[mid_frame+8,:,:]
        out[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 5*raw_img.shape[2]:6*raw_img.shape[2], cc]=np.amax(im, axis=0)
        '''
    row_index=1
    offset = 20
    im = seg + offset # make it brighter
    im[im==offset]=0
    im = im.astype(float) * (255/im.max())
    im = np.round(im)
    im = im.astype(np.uint8)
    for cc in range(3):
        out[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 0*raw_img.shape[2]:1*raw_img.shape[2], cc]=im[mid_frame-4,:,:]
        out[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 1*raw_img.shape[2]:2*raw_img.shape[2], cc]=im[mid_frame,:,:]
        out[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 2*raw_img.shape[2]:3*raw_img.shape[2], cc]=im[mid_frame+4,:,:]
        out[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 3*raw_img.shape[2]:4*raw_img.shape[2], cc]=np.amax(im, axis=0)
        '''
        out[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 0*raw_img.shape[2]:1*raw_img.shape[2], cc]=im[mid_frame-8,:,:]
        out[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 1*raw_img.shape[2]:2*raw_img.shape[2], cc]=im[mid_frame-4,:,:]
        out[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 2*raw_img.shape[2]:3*raw_img.shape[2], cc]=im[mid_frame,:,:]
        out[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 3*raw_img.shape[2]:4*raw_img.shape[2], cc]=im[mid_frame+4,:,:]
        out[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 4*raw_img.shape[2]:5*raw_img.shape[2], cc]=im[mid_frame+8,:,:]
        out[row_index*raw_img.shape[1]:(row_index+1)*raw_img.shape[1], 5*raw_img.shape[2]:6*raw_img.shape[2], cc]=np.amax(im, axis=0)
        '''

    # display the image for good/bad inspection
    fig = plt.figure()
    figManager = plt.get_current_fig_manager() 
    figManager.full_screen_toggle() 
    ax = fig.add_subplot(111)
    ax.imshow(out)
    #plt.tight_layout()
    cid = fig.canvas.mpl_connect('button_press_event', gt_sorting_callback)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    score = 0
    if button == 3:
        score = 1

    button = 0
    return score

def create_mask(raw_img, seg):
    global pts, draw_img, draw_mask, draw_ax
    bw = seg>0
    z_profile = np.zeros((bw.shape[0],),dtype=int)
    for zz in range(bw.shape[0]):
        z_profile[zz] = np.count_nonzero(bw[zz,:,:])
    mid_frame = round(histogram_otsu(z_profile)*bw.shape[0]).astype(int)

    offset = 20
    seg_label = seg + offset # make it brighter
    seg_label[seg_label==offset]=0
    seg_label = seg_label.astype(float) * (255/seg_label.max())
    seg_label = np.round(seg_label)
    seg_label = seg_label.astype(np.uint8)

    img = np.zeros((raw_img.shape[1], 3*raw_img.shape[2], 3),dtype=np.uint8)

    for cc in range(3):
        img[:, :raw_img.shape[2], cc]=raw_img[mid_frame,:,:]
        img[:, raw_img.shape[2]:2*raw_img.shape[2], cc]=seg_label[mid_frame,:,:]
        img[:, 2*raw_img.shape[2]:, cc]=np.amax(seg_label, axis=0)

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
        p.add_argument('--input_channel', default=0, type=int)
        p.add_argument('--seg_path', required=True, help='path to segmentation results')
        p.add_argument('--train_path', required=True, help='path to output training data')
        p.add_argument('--mask_path', help='[optional] the output directory for masks')
        p.add_argument('--csv_name', required=True, help='the csv file to save the sorting results')

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

        if os.path.exists(args.csv_name):
            print('the csv file for saving sorting results exists, sorting will be resumed')
        else:
            print('no existing csv found, start a new sorting ')
            filenames = glob(args.raw_path + '/*.tiff')
            filenames.sort()
            with open(args.csv_name, 'w') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow(['raw','seg','score','mask'])
                for _, fn in enumerate(filenames):
                    seg_fn = args.seg_path + os.sep + os.path.basename(fn)[:-5] + '_struct_segmentation.tiff'
                    assert os.path.exists(seg_fn)
                    filewriter.writerow([fn, seg_fn , None , None])

    def execute(self, args):

        global draw_mask
        # part 1: do sorting
        df = pd.read_csv(args.csv_name)

        for index, row in df.iterrows():
            reader = AICSImage(row['raw'])
            im_full = reader.data
            struct_img = im_full[0,args.input_channel,:,:,:]
            raw_img = (struct_img- struct_img.min() + 1e-8)/(struct_img.max() - struct_img.min() + 1e-8)
            raw_img = 255 * raw_img
            raw_img = raw_img.astype(np.uint8)

            reader_seg = AICSImage(row['seg'])
            im_seg_full = reader_seg.data
            assert im_seg_full.shape[0]==1 and im_seg_full.shape[1]==1
            seg = im_seg_full[0,0,:,:,:]

            if not np.isnan(row['score']) and (row['score']==1 or row['score']==0):
                continue

            score = gt_sorting(raw_img, seg)
            if score == 1:
                df['score'].iloc[index]=1
                need_mask = input('Do you need to add a mask for this image, enter y or n:  ')
                if need_mask == 'y':
                    create_mask(raw_img, seg.astype(np.uint8))
                    mask_fn = args.mask_path + os.sep + os.path.basename(row['raw'])[:-5] + '_mask.tiff'
                    crop_mask = np.zeros(seg.shape, dtype=np.uint8)
                    for zz in range(crop_mask.shape[0]):
                        crop_mask[zz,:,:] = draw_mask[:crop_mask.shape[1],:crop_mask.shape[2]]

                    crop_mask = crop_mask.astype(np.uint8)
                    crop_mask[crop_mask>0]=255
                    writer = omeTifWriter.OmeTifWriter(mask_fn)
                    writer.save(crop_mask)
                    df['mask'].iloc[index]=mask_fn
            else:
                df['score'].iloc[index]=0

            df.to_csv(args.csv_name)

        #########################################
        # generate training data:
        #  (we want to do this step after "sorting"
        #  (is mainly because we want to get the sorting 
        #  step as smooth as possible, even though
        #  this may waster i/o time on reloading images)
        # #######################################
        training_data_count = 0
        for index, row in df.iterrows():
            if row['score']==1:
                training_data_count += 1

                # load raw image
                reader = AICSImage(row['raw'])
                img = reader.data.astype(np.float32)
                struct_img = img[0,args.input_channel,:,:,:].copy()

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