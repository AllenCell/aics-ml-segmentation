import numpy as np
import random
from torch import from_numpy, Tensor, unsqueeze, cat
from aicsimageio import AICSImage
from aicsmlsegment.utils import (
    image_normalization,
)
from random import shuffle
from torch.utils.data import Dataset
from scipy.ndimage import zoom


from torchio.transforms import RandomAffine


# CODE for generic loader
#   No augmentation = NOAUG,simply load data and convert to tensor
#   Augmentation code:
#       RR = Rotate by a random degree from 1 to 180
#       R4 = Rotate by 0, 90, 180, 270
#       RF = Random flip
#       FH = Flip Horizontally
#       FV = Flip Vertically
#       FD = Flip Depth (i.e., along z dim)
#       SS = Size Scaling by a ratio between -0.1 to 0.1 (TODO)
#       IJ = Intensity Jittering (TODO)
#       DD = Dense Deformation (TODO)


def minmax(img):
    return (img - img.min()) / (img.max() - img.min())


def resize(img, config, min_max=False):
    if len(config["ResizeRatio"]) > 0 and config["ResizeRatio"] != [
        1.0,
        1.0,
        1.0,
    ]:
        # don't resize if resize ratio is all 1s
        # note that struct_img is only a view of img, so changes made on
        # struct_img also affects img
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


def load_img(filename, img_type, n_channel):
    """
    General function to load and rearrange the dimensions of 3D images
    input:
        filename: name of image to be loaded
        img_type: one of "label", "input", or  "costmap" determining the file extension
        n_channel: number of channels expected by model
    output:
        img: numpy array containing desired image in CZYX order or ZYX if it's a costmap
    """
    extension_dict = {
        "label": "_GT.ome.tif",
        "input": ".ome.tif",
        "costmap": "_CM.ome.tif",
    }
    reader = AICSImage(filename + extension_dict[img_type])
    if img_type == "label" or img_type == "input":
        img = reader.get_image_data("CZYX", S=0, T=0)

        # Legacy aicsimageio fix - image stored as zcyx instead of czyx
        if img.shape[0] != n_channel and img.shape[1] == n_channel:
            print(img.shape)
            img = np.swapaxes(img, 0, 1)
            print(filename, "RESHAPE TO:", img.shape)
    elif img_type == "costmap":
        img = reader.get_image_data("ZYX", S=0, T=0, C=0)

    return img


class UniversalDataset(Dataset):
    """
    Multipurpose dataset for training and validation. Randomly crops images, labels,
    and costmaps into user-specified number of patches. Users can specify which
    augmentations to apply.
    """

    def __init__(
        self,
        filenames,
        num_patch,
        size_out,
        n_channel,
        transforms=[],
        patchize=True,
    ):
        """
        input:
            filenames: path to images
            num_patch: number of random crops to be produced
            size_out: size of output from model
            n_channel: number of iput channels expected by model
            transforms: list of strings specifying transforms
            patchize: whether to divide image into patches
        """
        print("Generating samples...", end=" ")
        self.img = []
        self.gt = []
        self.cmap = []
        self.transforms = transforms
        num_data = len(filenames)
        shuffle(filenames)
        num_patch_per_img = np.zeros((num_data,), dtype=int)
        if num_data >= num_patch:
            # take one patch from each image
            num_patch_per_img[:num_patch] = 1
        else:  # assign how many patches to take form each img
            basic_num = num_patch // num_data
            # assign each image the same number of patches to extract
            num_patch_per_img[:] = basic_num

            # assign one more patch to the first few images to achieve the total patch number
            num_patch_per_img[: (num_patch - basic_num * num_data)] = (
                num_patch_per_img[: (num_patch - basic_num * num_data)] + 1
            )

        # extract patches from images until num_patch reached
        for img_idx, fn in enumerate(filenames):

            if patchize and len(self.img) == num_patch:
                break

            label = load_img(fn, img_type="label", n_channel=n_channel)
            raw = load_img(fn, img_type="input", n_channel=n_channel)
            costmap = load_img(fn, img_type="costmap", n_channel=n_channel)

            cost_scale = costmap.max()
            if cost_scale < 1:  # this should not happen, but just in case
                cost_scale = 1

            if "RF" in transforms:
                # random flip
                flip_flag = random.random()
                if flip_flag < 0.5:
                    raw = np.flip(
                        raw, axis=-1
                    ).copy()  # avoid negative stride error when converting to tensor
                    costmap = np.flip(costmap, axis=-1).copy()
                    label = np.flip(label, axis=-1).copy()

            if "RR" in transforms:
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

            if patchize:
                # take specified number of patches from current image
                new_patch_num = 0
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

                    # confirmed good crop
                    (self.img).append(
                        raw[
                            :,
                            pz : pz + size_out[0],
                            py : py + size_out[1],
                            px : px + size_out[2],
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
            else:
                (self.img).append(raw)
                (self.gt).append(label)
                (self.cmap).append(costmap)
        print("Done.")

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

        # convert to tensor
        label_tensor_2 = Tensor(self.gt[index].shape)
        label_tensor = unsqueeze(cat(label_tensor, out=label_tensor_2), 0)

        return image_tensor.float(), label_tensor, cmap_tensor.float()

    def __len__(self):
        return len(self.img)


class TestDataset(Dataset):
    def __init__(self, config):
        self.imgs = []
        self.tts = []
        inf_config = config["mode"]
        self.mode = inf_config

        if inf_config["name"] == "file":
            fn = inf_config["InputFile"]
            data_reader = AICSImage(fn)
            self.filenames = [fn]
            if inf_config["timelapse"]:
                assert data_reader.shape[1] > 1, "not a timelapse, check your data"

                for tt in range(data_reader.shape[1]):
                    # Assume:  dimensions = TCZYX
                    img = data_reader.get_image_data(
                        "CZYX", S=0, T=tt, C=config["InputCh"]
                    ).astype(float)
                    img = image_normalization(img, config["Normalization"])
                    img = resize(img, config)
                    self.imgs.append(img)
                    self.tts.append(tt)
            else:
                img = data_reader.get_image_data(
                    "CZYX", S=0, T=0, C=config["InputCh"]
                ).astype(float)

                img = image_normalization(img, config["Normalization"])
                img = resize(img, config)
                self.imgs.append(img)

        elif inf_config["name"] == "folder":
            from glob import glob

            filenames = glob(inf_config["InputDir"] + "/*" + inf_config["DataType"])
            filenames.sort()
            print("files to be processed:")
            print(filenames)
            self.filenames = filenames

            for _, fn in enumerate(filenames):
                # load data
                data_reader = AICSImage(fn)
                img = data_reader.get_image_data(
                    "CZYX", S=0, T=0, C=config["InputCh"]
                ).astype(float)

                img = resize(img, config, min_max=False)
                img = image_normalization(img, config["Normalization"])

                self.imgs.append(img)

    def __getitem__(self, index):
        """
        Returns:
            image, filename, and timepoint
        """
        if len(self.filenames) > 0:
            fn = self.filenames[index]
        else:
            fn = self.filenames[0]

        if len(self.tts) > 0:
            tt = self.tts[index]
        else:
            tt = []

        return {"img": self.imgs[index], "fn": fn, "tt": tt}

    def __len__(self):
        return len(self.imgs)