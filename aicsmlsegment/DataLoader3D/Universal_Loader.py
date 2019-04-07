import numpy as np
import os
from tifffile import imread, imsave
from PIL import Image
import random
from tqdm import tqdm 

from torch import from_numpy
from aicsimageio import AICSImage

import time
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset


# CODE for generic loader
#   No augmentation = NOAUG,simply load data and convert to tensor
#   Augmentation code:
#       RR = Rotate by a random degree from 1 to 180
#       R4 = Rotate by 0, 90, 180, 270
#       FH = Flip Horizantally
#       FV = Flip Vertically
#       FD = Flip Depth (i.e., along z dim)
#       SS = Size Scaling by a ratio between -0.1 to 0.1 (TODO)
#       IJ = Intensity Jittering (TODO)
#       DD = Dense Deformation (TODO)


class RR_FH_M0(Dataset):

    def __init__(self, filenames, num_patch, size_in, size_out):

        self.img = []
        self.gt = []
        self.cmap = []

        padding = [(x-y)//2 for x,y in zip(size_in, size_out)]
        total_in_count = size_in[0] * size_in[1] * size_in[2]
        total_out_count = size_out[0] * size_out[1] * size_out[2]

        num_data = len(filenames)
        num_patch_per_img = np.zeros((num_data,), dtype=int)
        if num_data >= num_patch:
            print('suggest to use more patch in each buffer')
            num_patch_per_img[:num_patch]=1
        else: 
            basic_num = num_patch // num_data
            # assign each image the same number of patches to extract
            num_patch_per_img[:] = basic_num

            # assign one more patch to the first few images to achieve the total patch number
            num_patch_per_img[:(num_patch-basic_num*num_data)] = num_patch_per_img[:(num_patch-basic_num*num_data)] + 1

        for img_idx, fn in tqdm(enumerate(filenames)):

            if len(self.img)==num_patch:
                break

            label_reader = AICSImage(fn+'_GT.ome.tif')  #CZYX
            label = label_reader.data
            label = np.squeeze(label,axis=0) # 4-D after squeeze

            # when the tif has only 1 channel, the loaded array may have falsely swaped dimensions (ZCYX). we want CZYX
            # (This may also happen in different OS or different package versions)
            # ASSUMPTION: we have more z slices than the number of channels 
            if label.shape[1]<label.shape[0]: 
                label = np.transpose(label,(1,0,2,3))

            input_reader = AICSImage(fn+'.ome.tif') #CZYX  #TODO: check size
            input_img = input_reader.data
            input_img = np.squeeze(input_img,axis=0)
            if input_img.shape[1] < input_img.shape[0]:
                input_img = np.transpose(input_img,(1,0,2,3))

            costmap_reader = AICSImage(fn+'_CM.ome.tif') # ZYX
            costmap = costmap_reader.data
            costmap = np.squeeze(costmap,axis=0)
            if costmap.shape[0] == 1:
                costmap = np.squeeze(costmap,axis=0)
            elif costmap.shape[1] == 1:
                costmap = np.squeeze(costmap,axis=1)

            img_pad0 = np.pad(input_img, ((0,0),(0,0),(padding[1],padding[1]),(padding[2],padding[2])), 'constant')
            raw = np.pad(img_pad0, ((0,0),(padding[0],padding[0]),(0,0),(0,0)), 'constant')

            raw_p0 = np.pad(input_img, ((0,0),(padding[0],padding[0]),(padding[1],padding[1]),(padding[2],padding[2])), 'constant')

            cost_scale = costmap.max()
            if cost_scale<1: ## this should not happen, but just in case
                cost_scale = 1

            deg = random.randrange(1,180)
            flip_flag = random.random()

            for zz in range(label.shape[1]):

                for ci in range(label.shape[0]):
                    labi = label[ci,zz,:,:]
                    labi_pil = Image.fromarray(np.uint8(labi))
                    new_labi_pil = labi_pil.rotate(deg,resample=Image.NEAREST)
                    if flip_flag<0.5:
                        new_labi_pil = new_labi_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    new_labi = np.array(new_labi_pil.convert('L'))
                    label[ci,zz,:,:] = new_labi.astype(int)

                cmap = costmap[zz,:,:]
                cmap_pil = Image.fromarray(np.uint8(255*(cmap/cost_scale)))
                new_cmap_pil = cmap_pil.rotate(deg,resample=Image.NEAREST)
                if flip_flag<0.5:
                    new_cmap_pil = new_cmap_pil.transpose(Image.FLIP_LEFT_RIGHT)
                new_cmap = np.array(new_cmap_pil.convert('L'))
                costmap[zz,:,:] = cost_scale*(new_cmap/255.0)

            for zz in range(raw.shape[1]):
                for ci in range(raw.shape[0]):
                    str_im = raw[ci,zz,:,:]
                    str_im_pil = Image.fromarray(np.uint8(str_im*255))
                    new_str_im_pil = str_im_pil.rotate(deg,resample=Image.BICUBIC)
                    if flip_flag<0.5:
                        new_str_im_pil = new_str_im_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    new_str_image = np.array(new_str_im_pil.convert('L'))
                    raw[ci,zz,:,:] = (new_str_image.astype(float))/255.0 

                    str_im = raw_p0[ci,zz,:,:]
                    str_im_pil = Image.fromarray(np.uint8(str_im*255))
                    new_str_im_pil = str_im_pil.rotate(deg,resample=Image.BICUBIC)
                    if flip_flag<0.5:
                        new_str_im_pil = new_str_im_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    new_str_image = np.array(new_str_im_pil.convert('L'))
                    raw_p0[ci,zz,:,:] = (new_str_image.astype(float))/255.0 

            new_patch_num = 0
            
            while new_patch_num < num_patch_per_img[img_idx]:
                
                pz = random.randint(0, label.shape[1] - size_out[0])
                py = random.randint(0, label.shape[2] - size_out[1])
                px = random.randint(0, label.shape[3] - size_out[2])

                
                # check if this is a good crop
                #ref_patch_raw = raw_p0[0,pz:pz+size_in[0],py:py+size_in[1],px:px+size_in[2]] 
                ref_patch_cmap = costmap[pz:pz+size_out[0],py:py+size_out[1],px:px+size_out[2]]
                #if np.count_nonzero(ref_patch_raw<1e-5) > 0.5*raw_p0.shape[0]*total_in_count: #not too large padding 
                #    continue
                #if np.count_nonzero(ref_patch_cmap<1e-5) > 0.5*total_out_count: #not too many white space
                #    continue
                

                # confirmed good crop
                (self.img).append(raw[:,pz:pz+size_in[0],py:py+size_in[1],px:px+size_in[2]] )
                (self.gt).append(label[:,pz:pz+size_out[0],py:py+size_out[1],px:px+size_out[2]])
                (self.cmap).append(ref_patch_cmap)

                new_patch_num += 1

    def __getitem__(self, index):

        image_tensor = from_numpy(self.img[index].astype(float))
        cmap_tensor = from_numpy(self.cmap[index].astype(float))

        label_tensor = []
        if self.gt[index].shape[0]>0:
            for zz in range(self.gt[index].shape[0]):
                label_tensor.append(from_numpy(self.gt[index][zz,:,:,:].astype(float)).float())
        else: 
            label_tensor.append(from_numpy(self.gt[index].astype(float)).float())

        return image_tensor.float(), label_tensor, cmap_tensor.float()

    def __len__(self):
        return len(self.img)

class RR_FH_M0C(Dataset):

    def __init__(self, filenames, num_patch, size_in, size_out):

        self.img = []
        self.gt = []
        self.cmap = []

        padding = [(x-y)//2 for x,y in zip(size_in, size_out)]
        total_in_count = size_in[0] * size_in[1] * size_in[2]
        total_out_count = size_out[0] * size_out[1] * size_out[2]

        num_data = len(filenames)
        num_patch_per_img = np.zeros((num_data,), dtype=int)
        if num_data >= num_patch:
            print('suggest to use more patch in each buffer')
            num_patch_per_img[:num_patch]=1
        else: 
            basic_num = num_patch // num_data
            # assign each image the same number of patches to extract
            num_patch_per_img[:] = basic_num

            # assign one more patch to the first few images to achieve the total patch number
            num_patch_per_img[:(num_patch-basic_num*num_data)] = num_patch_per_img[:(num_patch-basic_num*num_data)] + 1

        for img_idx, fn in tqdm(enumerate(filenames)):

            label_reader = AICSImage(fn+'_GT.ome.tif')  #CZYX
            label = label_reader.data
            label = np.squeeze(label,axis=0) # 4-D after squeeze

            # when the tif has only 1 channel, the loaded array may have falsely swaped dimensions (ZCYX). we want CZYX
            # (This may also happen in different OS or different package versions)
            # ASSUMPTION: we have more z slices than the number of channels 
            if label.shape[1]<label.shape[0]: 
                label = np.transpose(label,(1,0,2,3))

            input_reader = AICSImage(fn+'.ome.tif') #CZYX  #TODO: check size
            input_img = input_reader.data
            input_img = np.squeeze(input_img,axis=0)
            if input_img.shape[1] < input_img.shape[0]:
                input_img = np.transpose(input_img,(1,0,2,3))

            costmap_reader = AICSImage(fn+'_CM.ome.tif') # ZYX
            costmap = costmap_reader.data
            costmap = np.squeeze(costmap,axis=0)
            if costmap.shape[0] == 1:
                costmap = np.squeeze(costmap,axis=0)
            elif costmap.shape[1] == 1:
                costmap = np.squeeze(costmap,axis=1)

            img_pad0 = np.pad(input_img, ((0,0),(0,0),(padding[1],padding[1]),(padding[2],padding[2])), 'constant')
            raw = np.pad(img_pad0, ((0,0),(padding[0],padding[0]),(0,0),(0,0)), 'constant')

            raw_p0 = np.pad(input_img, ((0,0),(padding[0],padding[0]),(padding[1],padding[1]),(padding[2],padding[2])), 'constant')

            cost_scale = costmap.max()
            if cost_scale<1: ## this should not happen, but just in case
                cost_scale = 1

            deg = random.randrange(1,180)
            flip_flag = random.random()

            for zz in range(label.shape[1]):

                for ci in range(label.shape[0]):
                    labi = label[ci,zz,:,:]
                    labi_pil = Image.fromarray(np.uint8(labi))
                    new_labi_pil = labi_pil.rotate(deg,resample=Image.NEAREST)
                    if flip_flag<0.5:
                        new_labi_pil = new_labi_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    new_labi = np.array(new_labi_pil.convert('L'))
                    label[ci,zz,:,:] = new_labi.astype(int)

                cmap = costmap[zz,:,:]
                cmap_pil = Image.fromarray(np.uint8(255*(cmap/cost_scale)))
                new_cmap_pil = cmap_pil.rotate(deg,resample=Image.NEAREST)
                if flip_flag<0.5:
                    new_cmap_pil = new_cmap_pil.transpose(Image.FLIP_LEFT_RIGHT)
                new_cmap = np.array(new_cmap_pil.convert('L'))
                costmap[zz,:,:] = cost_scale*(new_cmap/255.0)

            for zz in range(raw.shape[1]):
                for ci in range(raw.shape[0]):
                    str_im = raw[ci,zz,:,:]
                    str_im_pil = Image.fromarray(np.uint8(str_im*255))
                    new_str_im_pil = str_im_pil.rotate(deg,resample=Image.BICUBIC)
                    if flip_flag<0.5:
                        new_str_im_pil = new_str_im_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    new_str_image = np.array(new_str_im_pil.convert('L'))
                    raw[ci,zz,:,:] = (new_str_image.astype(float))/255.0 

                    str_im = raw_p0[ci,zz,:,:]
                    str_im_pil = Image.fromarray(np.uint8(str_im*255))
                    new_str_im_pil = str_im_pil.rotate(deg,resample=Image.BICUBIC)
                    if flip_flag<0.5:
                        new_str_im_pil = new_str_im_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    new_str_image = np.array(new_str_im_pil.convert('L'))
                    raw_p0[ci,zz,:,:] = (new_str_image.astype(float))/255.0 

            new_patch_num = 0
            
            while new_patch_num < num_patch_per_img[img_idx]:
                
                pz = random.randint(0, label.shape[1] - size_out[0])
                py = random.randint(0, label.shape[2] - size_out[1])
                px = random.randint(0, label.shape[3] - size_out[2])

                
                # check if this is a good crop
                #ref_patch_raw = raw_p0[0,pz:pz+size_in[0],py:py+size_in[1],px:px+size_in[2]] 
                ref_patch_cmap = costmap[pz:pz+size_out[0],py:py+size_out[1],px:px+size_out[2]]
                if np.count_nonzero(ref_patch_cmap>1e-5) < 1000: #not too large padding 
                    continue
                

                # confirmed good crop
                (self.img).append(raw[:,pz:pz+size_in[0],py:py+size_in[1],px:px+size_in[2]] )
                (self.gt).append(label[:,pz:pz+size_out[0],py:py+size_out[1],px:px+size_out[2]])
                (self.cmap).append(ref_patch_cmap)

                new_patch_num += 1

    def __getitem__(self, index):

        image_tensor = from_numpy(self.img[index].astype(float))
        cmap_tensor = from_numpy(self.cmap[index].astype(float))

        if self.gt[index].shape[0]>0:
            label_tensor = []
            for zz in range(self.gt[index].shape[0]):
                tmp_tensor = from_numpy(self.gt[index][zz,:,:,:].astype(float))
                label_tensor.append(tmp_tensor.float())
        else: 
            label_tensor = from_numpy(self.gt[index].astype(float))
            label_tensor = label_tensor.float()

        return image_tensor.float(), label_tensor, cmap_tensor.float()

    def __len__(self):
        return len(self.img)

class NOAUG_M(Dataset):

    def __init__(self, filenames, num_patch, size_in, size_out):

        self.img = []
        self.gt = []
        self.cmap = []

        padding = [(x-y)//2 for x,y in zip(size_in, size_out)]
        total_in_count = size_in[0] * size_in[1] * size_in[2]
        total_out_count = size_out[0] * size_out[1] * size_out[2]

        num_data = len(filenames)
        num_patch_per_img = np.zeros((num_data,), dtype=int)
        if num_data >= num_patch:
            print('suggest to use more patch in each buffer')
            num_patch_per_img[:num_patch]=1
        else: 
            basic_num = num_patch // num_data
            # assign each image the same number of patches to extract
            num_patch_per_img[:] = basic_num

            # assign one more patch to the first few images to achieve the total patch number
            num_patch_per_img[:(num_patch-basic_num*num_data)] = num_patch_per_img[:(num_patch-basic_num*num_data)] + 1


        for img_idx, fn in tqdm(enumerate(filenames)):

            label_reader = AICSImage(fn+'_GT.ome.tif')  #CZYX
            label = label_reader.data
            label = np.squeeze(label,axis=0) # 4-D after squeeze

            # when the tif has only 1 channel, the loaded array may have falsely swaped dimensions (ZCYX). we want CZYX
            # (This may also happen in different OS or different package versions)
            # ASSUMPTION: we have more z slices than the number of channels 
            if label.shape[1]<label.shape[0]: 
                label = np.transpose(label,(1,0,2,3))

            input_reader = AICSImage(fn+'.ome.tif') #CZYX  #TODO: check size
            input_img = input_reader.data
            input_img = np.squeeze(input_img,axis=0)
            if input_img.shape[1] < input_img.shape[0]:
                input_img = np.transpose(input_img,(1,0,2,3))

            costmap_reader = AICSImage(fn+'_CM.ome.tif') # ZYX
            costmap = costmap_reader.data
            costmap = np.squeeze(costmap,axis=0)
            if costmap.shape[0] == 1:
                costmap = np.squeeze(costmap,axis=0)
            elif costmap.shape[1] == 1:
                costmap = np.squeeze(costmap,axis=1)

            img_pad0 = np.pad(input_img, ((0,0),(0,0),(padding[1],padding[1]),(padding[2],padding[2])), 'symmetric')
            raw = np.pad(img_pad0, ((0,0),(padding[0],padding[0]),(0,0),(0,0)), 'constant')

            raw_p0 = np.pad(input_img, ((0,0),(padding[0],padding[0]),(padding[1],padding[1]),(padding[2],padding[2])), 'constant')

            new_patch_num = 0
            
            while new_patch_num < num_patch_per_img[img_idx]:
                
                pz = random.randint(0, label.shape[1] - size_out[0])
                py = random.randint(0, label.shape[2] - size_out[1])
                px = random.randint(0, label.shape[3] - size_out[2])

                
                ## check if this is a good crop
                #ref_patch_raw = raw_p0[0,pz:pz+size_in[0],py:py+size_in[1],px:px+size_in[2]] 
                ref_patch_cmap = costmap[pz:pz+size_out[0],py:py+size_out[1],px:px+size_out[2]]
                #if np.count_nonzero(ref_patch_raw<1e-5) > 0.5*raw_p0.shape[0]*total_in_count: #not too large padding 
                #    continue
                #if np.count_nonzero(ref_patch_cmap<1e-5) > 0.5*total_out_count: #not too many white space
                #    continue
                

                # confirmed good crop
                (self.img).append(raw[:,pz:pz+size_in[0],py:py+size_in[1],px:px+size_in[2]] )
                (self.gt).append(label[:,pz:pz+size_out[0],py:py+size_out[1],px:px+size_out[2]])
                (self.cmap).append(ref_patch_cmap)

                new_patch_num += 1
                
    def __getitem__(self, index):

        image_tensor = from_numpy(self.img[index].astype(float))
        cmap_tensor = from_numpy(self.cmap[index].astype(float))

        #if self.gt[index].shape[0]>1:
        label_tensor = []
        for zz in range(self.gt[index].shape[0]):
            tmp_tensor = from_numpy(self.gt[index][zz,:,:,:].astype(float))
            label_tensor.append(tmp_tensor.float())
        #else: 
        #    label_tensor = from_numpy(self.gt[index].astype(float))
        #    label_tensor = label_tensor.float()
            
        return image_tensor.float(), label_tensor, cmap_tensor.float()

    def __len__(self):
        return len(self.img)