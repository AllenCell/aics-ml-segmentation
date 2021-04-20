import numpy as np
import random
from torch import from_numpy
import torch
from aicsimageio import AICSImage
from aicsmlsegment.utils import (
    image_normalization,
)
from random import shuffle
from torch.utils.data import Dataset, IterableDataset
from scipy.ndimage import zoom
from torchio.transforms import RandomAffine, RandomBiasField, RandomNoise
from monai.transforms import RandShiftIntensity


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
                1.0,
                1 / config["ResizeRatio"][0],
                1 / config["ResizeRatio"][1],
                1 / config["ResizeRatio"][2],
            ),
            order=2,
            mode="reflect",
        )
    return img.astype(np.float32)


def swap(l: list, index1: int, index2: int):
    """
    Swap index1 and index2 of list l
    """
    temp = l[index1]
    l[index1] = l[index2]
    l[index2] = temp
    return l


def validate_shape(img_shape: tuple, n_channel: int = 1, timelapse: bool = False):
    """
    General function to load and rearrange the dimensions of 3D images
    input:
        img_shape: STCZYX shape of image
        n_channel: number of channels expted in image
        timelapse: whether image is a timelapse

    output:
        load_dict: dictionary to be passed to AICSImage.get_image_data containing out_orientation and specific channel indices
        correct_shape: tuple rearranged img_shape
    """
    img_shape = list(img_shape)
    load_order = ["S", "T", "C", "Z", "Y", "X"]
    expected_channel_idx = 2
    # all dimensions that could be channel dimension
    real_channel_idx = [i for i, elem in enumerate(img_shape) if elem == n_channel]
    keep_channels = ["C"]
    if expected_channel_idx not in real_channel_idx:
        assert (
            len(real_channel_idx) > 0
        ), f"The specified channel dimension is incorrect and no other dimensions have size {n_channel}"
        # if nchannels is 1, doesn't matter which other size-1 dim we swap it with
        assert (
            n_channel == 1 or len(real_channel_idx) == 1
        ), f"Index of channel dimension is incorrect and there are multiple candidate channel dimensions. Please check your image metadata. {img_shape}"

        # change load order and image shape to reflect new index of  channel dimension
        real_channel_idx = real_channel_idx[-1]
        keep_channels.append(load_order[real_channel_idx])
        swap(load_order, real_channel_idx, expected_channel_idx)
        swap(img_shape, real_channel_idx, expected_channel_idx)

    load_dict = {"out_orientation": ""}
    correct_shape = []
    for s, load in zip(img_shape, load_order):
        if s == 1 and load not in keep_channels:
            # specify e.g. S=0 for aicsimagio
            load_dict[load] = 0
        else:
            load_dict["out_orientation"] += load
            correct_shape.append(s)
    if timelapse:
        assert (
            correct_shape[1] > 1
        ), "Image is not a timelapse, please check your image metadata"

    return load_dict, tuple(correct_shape)


def load_img(filename, img_type, n_channel=1, input_ch=None, shape_only=False):
    """
    General function to load and rearrange the dimensions of 3D images
    input:
        filename: name of image to be loaded
        img_type: one of "label", "input", or  "costmap" determining the file extension
        n_channel: number of channels expected by model
        input_ch: channel to extract from image during loading for testing
        shape_only: whether to only return validated shape of an image
    output:
        img: list of np.ndarray(s) containing image data.
    """
    extension_dict = {
        "label": "_GT.ome.tif",
        "input": ".ome.tif",
        "costmap": "_CM.ome.tif",
        "test": "",
        "timelapse": "",
    }
    reader = AICSImage(filename + extension_dict[img_type])
    args_dict, correct_shape = validate_shape(
        reader.shape, n_channel, img_type == "timelapse"
    )
    if shape_only:
        return correct_shape
    if img_type != "timelapse":
        img = reader.get_image_data(**args_dict)
        if img_type == "costmap":
            img = np.squeeze(img, 0)  # remove channel dimension
        elif img_type == "test":
            # return as list so we can iterate through it in test dataloader
            img = img.astype(float)
            img = [img[input_ch, :, :, :]]
    else:
        img = [
            reader.get_image_data(**args_dict, T=tt) for tt in range(correct_shape[1])
        ]
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
        size_in,
        size_out,
        n_channel,
        use_costmap=True,
        transforms=[],
        patchize: bool = True,
        check_crop: bool = False,
        init_only: bool = False,
    ):
        """
        input:
            filenames: path to images
            num_patch: number of random crops to be produced
            size_in: size of input to model
            size_out: size of output from model
            n_channel: number of iput channels expected by model
            transforms: list of strings specifying transforms
            patchize: whether to divide image into patches
            check_crop: whether to check
        """
        self.patchize = patchize
        self.img = []
        self.gt = []
        self.cmap = []
        self.transforms = transforms
        self.parameters = {
            "filenames": filenames,
            "num_patch": num_patch,
            "size_in": size_in,
            "size_out": size_out,
            "n_channel": n_channel,
            "use_costmap": use_costmap,
            "transforms": transforms,
            "patchize": patchize,
            "check_crop": check_crop,
        }
        self.num_patch = num_patch
        self.init_only = init_only
        if init_only:
            num_patch = 1
        if not patchize:
            print("Validating on", filenames)
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

        padding = [(x - y) // 2 for x, y in zip(size_in, size_out)]

        # extract patches from images until num_patch reached
        for img_idx, fn in enumerate(filenames):
            # if we're not dividing into patches, don't break before transforming imgs
            if patchize and len(self.img) == num_patch:
                break

            label = load_img(fn, img_type="label", n_channel=n_channel)
            input_img = load_img(fn, img_type="input", n_channel=n_channel)
            if use_costmap:
                costmap = load_img(fn, img_type="costmap", n_channel=n_channel)
            else:
                costmap = np.zeros((1))

            img_pad0 = np.pad(
                input_img,
                ((0, 0), (0, 0), (padding[1], padding[1]), (padding[2], padding[2])),
                "symmetric",
            )
            raw = np.pad(
                img_pad0, ((0, 0), (padding[0], padding[0]), (0, 0), (0, 0)), "constant"
            )

            if "RF" in transforms:
                # random flip
                flip_flag = random.random()
                if flip_flag < 0.5:
                    raw = np.flip(
                        raw, axis=-1
                    ).copy()  # avoid negative stride error when converting to tensor
                    if use_costmap:
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
                if use_costmap:
                    out_map = trans_label(
                        np.transpose(np.expand_dims(costmap, axis=0), (0, 3, 2, 1))
                    )
                    costmap = np.transpose(out_map[0, :, :, :], (2, 1, 0))
            if "RBF" in transforms:
                random_bias_field = RandomBiasField()
                raw = random_bias_field(raw)
            if "RN" in transforms:
                random_noise = RandomNoise()
                raw = random_noise(raw)

            if "RI" in transforms:
                random_intensity = RandShiftIntensity(offsets=0.15, prob=0.2)
                raw = random_intensity(raw)

            if patchize:
                # take specified number of patches from current image
                new_patch_num = 0
                num_fail = 0
                while new_patch_num < num_patch_per_img[img_idx]:
                    pz = random.randint(0, label.shape[1] - size_out[0])
                    py = random.randint(0, label.shape[2] - size_out[1])
                    px = random.randint(0, label.shape[3] - size_out[2])

                    if use_costmap:
                        # check if this is a good crop
                        ref_patch_cmap = costmap[
                            pz : pz + size_out[0],
                            py : py + size_out[1],
                            px : px + size_out[2],
                        ]
                        if check_crop:
                            if np.count_nonzero(ref_patch_cmap > 1e-5) < 1000:
                                num_fail += 1
                                if num_fail > 50:
                                    print("Failed to generate valid crops")
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
                    if use_costmap:
                        (self.cmap).append(ref_patch_cmap)
                    else:
                        self.cmap.append(costmap)

                    new_patch_num += 1
            else:
                (self.img).append(raw)
                (self.gt).append(label)
                (self.cmap).append(costmap)

    def __getitem__(self, index):
        if self.init_only:
            return torch.zeros(0)
        img_tensor = from_numpy(self.img[index].astype(float)).float()
        gt_tensor = from_numpy(self.gt[index].astype(float)).float()
        cmap_tensor = from_numpy(self.cmap[index].astype(float)).float()

        return (
            img_tensor,
            gt_tensor,
            cmap_tensor,
        )

    def __len__(self):
        if self.init_only:
            return self.num_patch
        return len(self.img)

    def get_params(self):
        return self.parameters


def patchize(img, pr, patch_size):
    ijk = []
    imgs = []

    x_max = img.shape[-1]
    y_max = img.shape[-2]
    z_max = img.shape[-3]

    x_patch_sz = x_max // pr[2]
    y_patch_sz = y_max // pr[1]
    z_patch_sz = z_max // pr[0]

    assert (
        x_patch_sz >= patch_size[2]
        and y_patch_sz >= patch_size[1]
        and z_patch_sz >= patch_size[0]
    ), "Large image resize patches must be larger than model patch size"

    maxs = [z_max, y_max, x_max]
    patch_szs = [z_patch_sz, y_patch_sz, x_patch_sz]

    all_coords = []
    for i in range(3):
        # remainder is the number of pixels per axis not evenly divided into patches
        remainder = maxs[i] % pr[i]
        coords = [
            # for the first *remainder* patches, we want to expand the
            # patch_size by one pixel so that after *remainder* iterations,
            # all pixels are included in exactly one patch.
            j * patch_szs[i] + j if j < remainder
            # once *remainder* pixels have been added we don't have to
            # add an extra pixel to each patch's size, but we do
            # have to shift the starts of the remaining patches
            # by the *remainder* pixels we've already added
            else j * patch_szs[i] + remainder
            for j in range(pr[i] + 1)
        ]
        all_coords.append(coords)

    for i in range(pr[0]):  # z
        for j in range(pr[1]):  # y
            for k in range(pr[2]):  # x
                i_start = max(0, all_coords[0][i] - 5)
                i_end = min(z_max, all_coords[0][i + 1] + 5)

                j_start = max(0, all_coords[1][j] - 30)
                j_end = min(y_max, all_coords[1][j + 1] + 30)

                k_start = max(0, all_coords[2][k] - 30)
                k_end = min(x_max, all_coords[2][k + 1] + 30)
                temp = np.array(
                    img[
                        :,
                        i_start:i_end,
                        j_start:j_end,
                        k_start:k_end,
                    ]
                )
                ijk.append([i_start, j_start, k_start])
                imgs.append(temp)
    return ijk, imgs


def pad_image(image, size_in, size_out):
    padding = [(x - y) // 2 for x, y in zip(size_in, size_out)]
    image = np.pad(
        image,
        ((0, 0), (0, 0), (padding[1], padding[1]), (padding[2], padding[2])),
        "symmetric",
    )
    image = np.pad(
        image,
        ((0, 0), (padding[0], padding[0]), (0, 0), (0, 0)),
        "constant",
    )
    image = from_numpy(image.astype(float)).float()
    return image


# TODO deal with timelapse images
class TestDataset_iterable(IterableDataset):
    def __init__(self, config):
        self.config = config
        self.inf_config = config["mode"]
        self.model_config = config["model"]
        self.patchize_ratio = config["large_image_resize"]
        self.patches_per_image = np.prod(self.patchize_ratio)
        self.load_type = "test"
        self.timelapse = False

        try:  # monai
            self.size_in = self.model_config["patch_size"]
            self.size_out = self.model_config["patch_size"]
            self.nchannel = self.model_config["in_channels"]
        except KeyError:  # unet_xy_zoom
            self.size_in = self.model_config["size_in"]
            self.size_out = self.model_config["size_out"]
            self.nchannel = self.model_config["nchannel"]

        if self.inf_config["name"] == "file":
            filenames = [self.inf_config["InputFile"]]
            if "timelapse" in self.inf_config and self.inf_config["timelapse"]:
                self.load_type = "timelapse"
                self.timelapse = True
        else:
            from glob import glob

            filenames = glob(
                self.inf_config["InputDir"] + "/*" + self.inf_config["DataType"]
            )
            filenames.sort()
        print("Files to be processed:", filenames)

        self.filenames = filenames
        self.start = None
        self.end = None
        self.all_img_info = []

    def patchize_wrapper_iterable(self, pr, fn, img, patch_size, tt, timelapse):
        if pr == [1, 1, 1]:
            return_dicts = [
                {
                    "fn": fn,
                    "img": pad_image(img, self.size_in, self.size_out),
                    "im_shape": img.shape,
                    "ijk": -1,
                    "save_n_batches": 1,
                    "tt": tt if timelapse else -1,
                }
            ]
        else:
            save_n_batches = np.prod(
                pr
            )  # how many patches until aggregated image saved
            ijk, imgs = patchize(img, pr, patch_size)
            return_dicts = []
            for index, patch in zip(ijk, imgs):
                return_dict = {
                    "fn": fn,
                    "img": pad_image(patch, self.size_in, self.size_out),
                    "im_shape": img.shape,
                    "ijk": index,
                    "save_n_batches": save_n_batches,
                    "tt": tt if timelapse else -1,
                }
                return_dicts.append(return_dict)
        return return_dicts

    def __iter__(self):
        self.current_index = self.start
        return self

    def __next__(self):
        if self.current_index > self.end and len(self.all_img_info) == 0:
            raise StopIteration
        if len(self.all_img_info) == 0:  # load new image
            fn = self.filenames[self.current_index]
            imgs = load_img(fn, self.load_type, self.nchannel, self.config["InputCh"])
            # only one image unless timelapse
            for tt, img in enumerate(imgs):
                img = resize(img, self.config)
                img = image_normalization(img, self.config["Normalization"])
                # generate patch info
                self.all_img_info += self.patchize_wrapper_iterable(
                    self.patchize_ratio,
                    fn,
                    img,
                    self.size_in,
                    tt,
                    self.timelapse,
                )
            self.current_index += 1  # next iteration load the next file
        return self.all_img_info.pop()  # pop patch/tp
