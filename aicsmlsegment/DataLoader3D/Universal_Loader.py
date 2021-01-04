import numpy as np
from PIL import Image
import random
from tqdm import tqdm
from torch import from_numpy, Tensor, unsqueeze, cat
from aicsimageio import AICSImage
from random import shuffle
from torch.utils.data import Dataset

from torchio.transforms import RandomAffine


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

        padding = [(x - y) // 2 for x, y in zip(size_in, size_out)]

        num_data = len(filenames)
        shuffle(filenames)
        num_patch_per_img = np.zeros((num_data,), dtype=int)
        if num_data >= num_patch:
            # all one
            num_patch_per_img[:num_patch] = 1
        else:
            basic_num = num_patch // num_data
            # assign each image the same number of patches to extract
            num_patch_per_img[:] = basic_num

            # assign one more patch to the first few images to achieve the total patch number
            num_patch_per_img[: (num_patch - basic_num * num_data)] = (
                num_patch_per_img[: (num_patch - basic_num * num_data)] + 1
            )

        for img_idx, fn in enumerate(filenames):

            if len(self.img) == num_patch:
                break

            label_reader = AICSImage(fn + "_GT.ome.tif")
            label = label_reader.get_image_data("CZYX", S=0, T=0)

            input_reader = AICSImage(fn + ".ome.tif")
            input_img = input_reader.get_image_data("CZYX", S=0, T=0)

            costmap_reader = AICSImage(fn + "_CM.ome.tif")
            costmap = costmap_reader.get_image_data("ZYX", S=0, T=0, C=0)

            img_pad0 = np.pad(
                input_img,
                ((0, 0), (0, 0), (padding[1], padding[1]), (padding[2], padding[2])),
                "constant",
            )
            raw = np.pad(
                img_pad0, ((0, 0), (padding[0], padding[0]), (0, 0), (0, 0)), "constant"
            )

            cost_scale = costmap.max()
            if cost_scale < 1:  ## this should not happen, but just in case
                cost_scale = 1

            deg = random.randrange(1, 180)
            flip_flag = random.random()

            for zz in range(label.shape[1]):

                for ci in range(label.shape[0]):
                    labi = label[ci, zz, :, :]
                    labi_pil = Image.fromarray(np.uint8(labi))
                    new_labi_pil = labi_pil.rotate(deg, resample=Image.NEAREST)
                    if flip_flag < 0.5:
                        new_labi_pil = new_labi_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    new_labi = np.array(new_labi_pil.convert("L"))
                    label[ci, zz, :, :] = new_labi.astype(int)

                cmap = costmap[zz, :, :]
                cmap_pil = Image.fromarray(np.uint8(255 * (cmap / cost_scale)))
                new_cmap_pil = cmap_pil.rotate(deg, resample=Image.NEAREST)
                if flip_flag < 0.5:
                    new_cmap_pil = new_cmap_pil.transpose(Image.FLIP_LEFT_RIGHT)
                new_cmap = np.array(new_cmap_pil.convert("L"))
                costmap[zz, :, :] = cost_scale * (new_cmap / 255.0)

            for zz in range(raw.shape[1]):
                for ci in range(raw.shape[0]):
                    str_im = raw[ci, zz, :, :]
                    str_im_pil = Image.fromarray(np.uint8(str_im * 255))
                    new_str_im_pil = str_im_pil.rotate(deg, resample=Image.BICUBIC)
                    if flip_flag < 0.5:
                        new_str_im_pil = new_str_im_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    new_str_image = np.array(new_str_im_pil.convert("L"))
                    raw[ci, zz, :, :] = (new_str_image.astype(float)) / 255.0
            new_patch_num = 0

            while new_patch_num < num_patch_per_img[img_idx]:

                pz = random.randint(0, label.shape[1] - size_out[0])
                py = random.randint(0, label.shape[2] - size_out[1])
                px = random.randint(0, label.shape[3] - size_out[2])

                # check if this is a good crop
                ref_patch_cmap = costmap[
                    pz : pz + size_out[0], py : py + size_out[1], px : px + size_out[2]
                ]

                # confirmed good crop
                (self.img).append(
                    raw[
                        :,
                        pz : pz + size_in[0],
                        py : py + size_in[1],
                        px : px + size_in[2],
                    ]
                )
                (self.gt).append(
                    label[
                        :,
                        pz : pz + size_out[0],
                        py : py + size_out[1],
                        px : px + size_out[2],
                    ]
                )
                (self.cmap).append(ref_patch_cmap)

                new_patch_num += 1

    def __getitem__(self, index):

        image_tensor = from_numpy(self.img[index].astype(float))
        cmap_tensor = from_numpy(self.cmap[index].astype(float))

        label_tensor = []
        if self.gt[index].shape[0] > 0:
            for zz in range(self.gt[index].shape[0]):
                label_tensor.append(
                    from_numpy(self.gt[index][zz, :, :, :].astype(float)).float()
                )
        else:
            label_tensor.append(from_numpy(self.gt[index].astype(float)).float())

        label_tensor_2 = Tensor(self.gt[index].shape)
        label_tensor = unsqueeze(cat(label_tensor, out=label_tensor_2), 0)

        return image_tensor.float(), label_tensor, cmap_tensor.float()

    def __len__(self):
        return len(self.img)


class RR_FH_M0C(Dataset):
    def __init__(self, filenames, num_patch, size_in, size_out):

        self.img = []
        self.gt = []
        self.cmap = []

        padding = [(x - y) // 2 for x, y in zip(size_in, size_out)]

        num_data = len(filenames)
        shuffle(filenames)

        num_trial_round = 0
        while len(self.img) < num_patch:

            # to avoid dead loop
            num_trial_round = num_trial_round + 1
            if num_trial_round > 2:
                break

            num_patch_to_obtain = num_patch - len(self.img)
            num_patch_per_img = np.zeros((num_data,), dtype=int)
            if num_data >= num_patch_to_obtain:
                # all one
                num_patch_per_img[:num_patch_to_obtain] = 1
            else:
                basic_num = num_patch_to_obtain // num_data
                # assign each image the same number of patches to extract
                num_patch_per_img[:] = basic_num

                # assign one more patch to the first few images to achieve the total patch number
                num_patch_per_img[: (num_patch_to_obtain - basic_num * num_data)] = (
                    num_patch_per_img[: (num_patch_to_obtain - basic_num * num_data)]
                    + 1
                )

            for img_idx, fn in tqdm(enumerate(filenames)):
                if len(self.img) == num_patch:
                    break

                label_reader = AICSImage(fn + "_GT.ome.tif")
                label = label_reader.get_image_data("CZYX", S=0, T=0)
                # HACK
                label = np.squeeze(label)
                label = np.expand_dims(label, axis=0)

                input_reader = AICSImage(fn + ".ome.tif")
                input_img = input_reader.get_image_data("CZYX", S=0, T=0)
                # HACK
                input_img = np.squeeze(input_img)
                input_img = np.expand_dims(input_img, axis=0)

                costmap_reader = AICSImage(fn + "_CM.ome.tif")
                costmap = costmap_reader.get_image_data("CZYX", S=0, T=0)
                # HACK
                costmap = np.squeeze(costmap)

                img_pad0 = np.pad(
                    input_img,
                    (
                        (0, 0),
                        (0, 0),
                        (padding[1], padding[1]),
                        (padding[2], padding[2]),
                    ),
                    "constant",
                )
                raw = np.pad(
                    img_pad0,
                    ((0, 0), (padding[0], padding[0]), (0, 0), (0, 0)),
                    "constant",
                )

                cost_scale = costmap.max()
                if cost_scale < 1:  # this should not happen, but just in case
                    cost_scale = 1

                # random flip
                flip_flag = random.random()
                if flip_flag < 0.5:
                    raw = np.flip(raw, axis=-1)
                    costmap = np.flip(costmap, axis=-1)
                    label = np.flip(label, axis=-1)

                # random rotation
                deg = random.randrange(1, 180)
                trans = RandomAffine(
                    scales=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                    degrees=(0, 0, 0, 0, deg, deg),
                    default_pad_value=0,
                    image_interpolation="bspline",
                    center="image",
                )

                # rotate the raw image
                out_img = trans(np.transpose(raw, (0, 3, 2, 1)))
                raw = np.transpose(out_img, (0, 3, 2, 1))

                trans_label = RandomAffine(
                    scales=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                    degrees=(0, 0, 0, 0, deg, deg),
                    default_pad_value=0,
                    image_interpolation="nearest",
                    center="image",
                )
                # rotate label and costmap
                out_label = trans_label(np.transpose(label, (0, 3, 2, 1)))
                label = np.transpose(out_label, (0, 3, 2, 1))

                out_map = trans_label(
                    np.transpose(np.expand_dims(costmap, axis=0), (0, 3, 2, 1))
                )
                costmap = np.transpose(out_map[0, :, :, :], (2, 1, 0))

                new_patch_num = 0
                num_fail = 0
                while new_patch_num < num_patch_per_img[img_idx]:
                    pz = random.randint(0, label.shape[1] - size_out[0])
                    py = random.randint(0, label.shape[2] - size_out[1])
                    px = random.randint(0, label.shape[3] - size_out[2])

                    # check if this is a good crop
                    ref_patch_cmap = costmap[
                        pz : pz + size_out[0],
                        py : py + size_out[1],
                        px : px + size_out[2],
                    ]
                    if (
                        np.count_nonzero(ref_patch_cmap > 1e-5) < 1000
                    ):  # enough valida samples
                        num_fail = num_fail + 1
                        if num_fail > 50:
                            break
                        continue

                    # confirmed good crop
                    (self.img).append(
                        raw[
                            :,
                            pz : pz + size_in[0],
                            py : py + size_in[1],
                            px : px + size_in[2],
                        ]
                    )
                    (self.gt).append(
                        label[
                            :,
                            pz : pz + size_out[0],
                            py : py + size_out[1],
                            px : px + size_out[2],
                        ]
                    )
                    (self.cmap).append(ref_patch_cmap)

                    new_patch_num += 1

    def __getitem__(self, index):

        image_tensor = from_numpy(self.img[index].astype(float))
        cmap_tensor = from_numpy(self.cmap[index].astype(float))

        label_tensor = []
        if self.gt[index].shape[0] > 0:
            for zz in range(self.gt[index].shape[0]):
                label_tensor.append(
                    from_numpy(self.gt[index][zz, :, :, :].astype(float)).float()
                )
        else:
            label_tensor.append(from_numpy(self.gt[index].astype(float)).float())

        label_tensor_2 = Tensor(self.gt[index].shape)
        label_tensor = unsqueeze(cat(label_tensor, out=label_tensor_2), 0)

        return image_tensor.float(), label_tensor, cmap_tensor.float()

    def __len__(self):
        return len(self.img)


class NOAUG_M(Dataset):
    def __init__(self, filenames, num_patch, size_in, size_out):

        self.img = []
        self.gt = []
        self.cmap = []

        padding = [(x - y) // 2 for x, y in zip(size_in, size_out)]

        num_data = len(filenames)
        shuffle(filenames)
        num_patch_per_img = np.zeros((num_data,), dtype=int)
        if num_data >= num_patch:
            # all one
            num_patch_per_img[:num_patch] = 1
        else:
            basic_num = num_patch // num_data
            # assign each image the same number of patches to extract
            num_patch_per_img[:] = basic_num

            # assign one more patch to the first few images to achieve the total patch number
            num_patch_per_img[: (num_patch - basic_num * num_data)] = (
                num_patch_per_img[: (num_patch - basic_num * num_data)] + 1
            )

        for img_idx, fn in enumerate(filenames):

            label_reader = AICSImage(fn + "_GT.ome.tif")
            label = label_reader.get_image_data("CZYX", S=0, T=0)

            input_reader = AICSImage(fn + ".ome.tif")
            input_img = input_reader.get_image_data("CZYX", S=0, T=0)

            costmap_reader = AICSImage(fn + "_CM.ome.tif")
            costmap = costmap_reader.get_image_data("ZYX", S=0, T=0, C=0)

            img_pad0 = np.pad(
                input_img,
                ((0, 0), (0, 0), (padding[1], padding[1]), (padding[2], padding[2])),
                "symmetric",
            )
            raw = np.pad(
                img_pad0, ((0, 0), (padding[0], padding[0]), (0, 0), (0, 0)), "constant"
            )

            new_patch_num = 0

            while new_patch_num < num_patch_per_img[img_idx]:
                pz = random.randint(0, label.shape[1] - size_out[0])
                py = random.randint(0, label.shape[2] - size_out[1])
                px = random.randint(0, label.shape[3] - size_out[2])

                # check if this is a good crop
                ref_patch_cmap = costmap[
                    pz : pz + size_out[0], py : py + size_out[1], px : px + size_out[2]
                ]

                # confirmed good crop
                (self.img).append(
                    raw[
                        :,
                        pz : pz + size_in[0],
                        py : py + size_in[1],
                        px : px + size_in[2],
                    ]
                )
                (self.gt).append(
                    label[
                        :,
                        pz : pz + size_out[0],
                        py : py + size_out[1],
                        px : px + size_out[2],
                    ]
                )
                (self.cmap).append(ref_patch_cmap)

                new_patch_num += 1

    def __getitem__(self, index):

        image_tensor = from_numpy(self.img[index].astype(float))
        cmap_tensor = from_numpy(self.cmap[index].astype(float))

        label_tensor = []
        for zz in range(self.gt[index].shape[0]):
            tmp_tensor = from_numpy(self.gt[index][zz, :, :, :].astype(float))
            label_tensor.append(tmp_tensor.float())
        label_tensor_2 = Tensor(self.gt[index].shape)
        label_tensor = unsqueeze(cat(label_tensor, out=label_tensor_2), 0)
        return image_tensor.float(), label_tensor, cmap_tensor.float()

    def __len__(self):
        return len(self.img)