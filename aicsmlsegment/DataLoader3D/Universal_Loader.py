import numpy as np
import random
from torch import from_numpy
import torch
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


def validate_shape(img, n_channel, filename):
    # Legacy aicsimageio fix - image stored as zcyx instead of czyx
    if img.shape[0] != n_channel and img.shape[1] == n_channel:
        print(img.shape)
        img = np.swapaxes(img, 0, 1)
        print(filename, "RESHAPE TO:", img.shape)
    return img


def load_img(filename, img_type, n_channel, input_ch=None):
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
        "test": "",
        "timelapse": "",
    }

    reader = AICSImage(filename + extension_dict[img_type])
    if img_type in ["label", "input"]:
        img = reader.get_image_data("CZYX", S=0, T=0)
        img = validate_shape(img, n_channel, filename)
    elif img_type == "costmap":
        img = reader.get_image_data("ZYX", S=0, T=0, C=0)
    elif img_type == "test":
        img = reader.get_image_data("CZYX", S=0, T=0, C=input_ch).astype(float)
        img = validate_shape(img, n_channel, filename)
        return [img]  # return as list so we can iterate through it in test dataloader
    elif img_type == "timelapse":
        assert reader.shape[1] > 1, "not a timelapse, check your data"
        imgs = []
        for tt in range(reader.shape[1]):
            # Assume:  dimensions = TCZYX
            img = reader.get_image_data("CZYX", S=0, T=tt, C=input_ch).astype(float)
            img = validate_shape(img, n_channel, filename)
            imgs.append(img)
        return imgs

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

        padding = [(x - y) // 2 for x, y in zip(size_in, size_out)]

        # extract patches from images until num_patch reached
        for img_idx, fn in enumerate(filenames):
            # print("training on", fn)

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
        print("Done.")

    def __getitem__(self, index):
        # converting to tensor in get item prevents conversion to double tensor
        # and saves gpu memory
        img_tensor = from_numpy(self.img[index].astype(float)).float()
        gt_tensor = from_numpy(self.gt[index].astype(float)).float()
        cmap_tensor = from_numpy(self.cmap[index].astype(float)).float()
        return (
            img_tensor,
            gt_tensor,
            cmap_tensor,
        )

    def __len__(self):
        return len(self.img)


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


class TestDataset(Dataset):
    def __init__(self, config):
        inf_config = config["mode"]
        self.mode = inf_config
        try:  # monai
            self.patch_size = config["model"]["patch_size"]
        except KeyError:  # unet_xy_zoom
            self.patch_size = config["model"]["size_in"]
        self.save_n_batches = 1

        self.filenames = []
        self.ijk = []
        self.im_shape = []
        self.imgs = []
        self.tts = []

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
                    self.imgs.append(img)
                    self.tts.append(tt)
                    self.filenames.append(fn)
                    self.im_shape.append(img.shape)

            else:
                img = data_reader.get_image_data(
                    "CZYX", S=0, T=0, C=config["InputCh"]
                ).astype(float)

                img = image_normalization(img, config["Normalization"])
                img = resize(img, config)
                pr = config["large_image_resize"]

                if pr is None or pr == [1, 1, 1]:
                    self.filenames.append(fn)
                    self.imgs.append(img)
                    self.im_shape.append(img.shape)
                else:
                    # how many patches until aggregated image saved
                    self.save_n_batches = np.prod(pr)
                    # make sure that filename aligns with img in get_item
                    self.filenames += [fn] * self.save_n_batches
                    ijk, imgs = patchize(img, pr, self.patch_size)
                    self.ijk += ijk
                    self.imgs += imgs
                    self.im_shape += [img.shape] * len(imgs)

        elif inf_config["name"] == "folder":
            from glob import glob

            filenames = glob(inf_config["InputDir"] + "/*" + inf_config["DataType"])
            filenames.sort()
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

                pr = config["large_image_resize"]
                if pr is None or pr == [1, 1, 1]:
                    self.filenames.append(fn)
                    self.imgs.append(img)
                    self.im_shape.append(img.shape)
                else:
                    # how many patches until aggregated image saved
                    self.save_n_batches = np.prod(pr)
                    # make sure that filename aligns with img in get_item
                    self.filenames += [fn] * self.save_n_batches
                    ijk, imgs = patchize(img, pr, self.patch_size)
                    self.ijk += ijk
                    self.imgs += imgs
                    self.im_shape += [img.shape] * len(imgs)

        if config["model"]["size_in"] != config["model"]["size_out"]:
            padding = [
                (x - y) // 2
                for x, y in zip(config["model"]["size_in"], config["model"]["size_out"])
            ]

            for i in range(len(self.imgs)):
                xy_padded = np.pad(
                    self.imgs[i],
                    (
                        (0, 0),
                        (0, 0),
                        (padding[1], padding[1]),
                        (padding[2], padding[2]),
                    ),
                    "symmetric",
                )
                self.imgs[i] = np.pad(
                    xy_padded,
                    ((0, 0), (padding[0], padding[0]), (0, 0), (0, 0)),
                    "constant",
                )

    def __getitem__(self, index):
        """
        Returns:
            image, filename, timepoint, coordinates of patch corner,
            number of batches to save, and shape of input image
        """
        if len(self.filenames) > 0:
            fn = self.filenames[index]
        else:
            fn = self.filenames[0]

        if len(self.tts) > 0:
            tt = self.tts[index]
        else:
            tt = []

        if len(self.ijk) > 0:
            ijk = self.ijk[index]
        else:
            ijk = []

        if len(self.im_shape) > 0:
            im_shape = self.im_shape[index]
        else:
            im_shape = self.im_shape[0]

        return {
            "img": from_numpy(self.imgs[index].astype(float)).float(),
            "fn": fn,
            "tt": tt,
            "ijk": ijk,
            "save_n_batches": self.save_n_batches,
            "img_shape": im_shape,
        }

    def __len__(self):
        return len(self.imgs)


class TestDataset_load_at_runtime(Dataset):
    def __init__(self, config):
        inf_config = config["mode"]
        model_config = config["model"]
        try:  # monai
            patch_size = model_config["patch_size"]
            nchannel = model_config["in_channels"]
        except KeyError:  # unet_xy_zoom
            patch_size = model_config["size_in"]
            nchannel = model_config["nchannel"]

        self.save_n_batches = 1
        self.filenames = []  # list of filenames, 1 per image patch or image
        self.ijk = []  # list of ijk indices,
        self.im_shape = []
        self.imgs = []
        self.tts = []

        if inf_config["name"] == "file":
            filenames = [inf_config["InputFile"]]
        else:
            from glob import glob

            filenames = glob(inf_config["InputDir"] + "/*" + inf_config["DataType"])
            filenames.sort()
        print("Files to be processed:", filenames)

        load_type = "test"
        if inf_config["timelapse"]:
            load_type = "timelapse"

        for fn in filenames:  # only one filename unless in_config name is folder
            imgs = load_img(fn, load_type, nchannel, config["InputCh"])
            for tt, img in enumerate(imgs):  # only one image unless timelapse
                img = image_normalization(img, config["Normalization"])
                img = resize(img, config)

                img_info = patchize_wrapper(
                    config["large_image_resize"], fn, img, patch_size, tt
                )
                self.filenames += img_info["filenames"]
                self.imgs += img_info["imgs"]
                self.im_shape += img_info["im_shape"]
                self.ijk += img_info["ijk"]
                self.save_n_batches = img_info["save_n_batches"]
                self.tts += img_info["tt"]

        if model_config["size_in"] != model_config["size_out"]:
            padding = [
                (x - y) // 2
                for x, y in zip(model_config["size_in"], model_config["size_out"])
            ]

            for i in range(len(self.imgs)):
                xy_padded = np.pad(
                    self.imgs[i],
                    (
                        (0, 0),
                        (0, 0),
                        (padding[1], padding[1]),
                        (padding[2], padding[2]),
                    ),
                    "symmetric",
                )
                self.imgs[i] = np.pad(
                    xy_padded,
                    ((0, 0), (padding[0], padding[0]), (0, 0), (0, 0)),
                    "constant",
                )

    def __getitem__(self, index):
        """
        Returns:
            image, filename, timepoint, coordinates of patch corner,
            number of batches to save, and shape of input image pre-patchize
        """
        return {
            "img": from_numpy(self.imgs[index].astype(float)).float(),
            "fn": self.filenames[index],
            "tt": self.tts[index],
            "ijk": self.ijk[index],
            "save_n_batches": self.save_n_batches,
            "img_shape": self.im_shape[index],
        }

    def __len__(self):
        return len(self.imgs)


def patchize_wrapper(pr, fn, img, patch_size, tt, timelapse):
    if pr == [1, 1, 1]:
        print("No patches for", fn)
        return_dict = {
            "fn": [fn] if timelapse else fn,
            "img": [img] if timelapse else img,
            "im_shape": [img.shape] if timelapse else img.shape,
            "ijk": [-1] if timelapse else -1,  # avoid reducing list size when inserted
            "save_n_batches": 1,
            "tt": [tt] if timelapse else -1,
        }

    else:
        save_n_batches = np.prod(pr)  # how many patches until aggregated image saved
        ijk, imgs = patchize(
            img, pr, patch_size
        )  # make sure that filename aligns with img in get_item
        return_dict = {
            "fn": [fn] * save_n_batches,
            "img": imgs,
            "im_shape": [img.shape] * len(imgs),
            "ijk": ijk,
            "save_n_batches": save_n_batches,
            "tt": [tt] * save_n_batches,
        }
        print("split", fn, "into", save_n_batches, "batches")
    return return_dict


def pad_image(image, size_in, size_out, precision):
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


class RNDTestLoad(Dataset):
    def __init__(self, config):
        self.config = config
        self.inf_config = config["mode"]
        self.model_config = config["model"]
        self.precision = config["precision"]
        self.patchize_ratio = config["large_image_resize"]
        self.load_type = "test"
        self.timelapse = False
        if self.patchize_ratio is None:
            self.patchize_ratio = [1, 1, 1]

        try:  # monai
            self.patch_size = self.model_config["patch_size"]
            self.nchannel = self.model_config["in_channels"]
        except KeyError:  # unet_xy_zoom
            self.patch_size = self.model_config["size_in"]
            self.nchannel = self.model_config["nchannel"]

        if self.inf_config["name"] == "file":
            filenames = [self.inf_config["InputFile"]]
            if "timelapse" in self.inf_config and self.inf_config["timelapse"] == True:
                self.load_type = "timelapse"
                self.timelapse = True
        else:
            from glob import glob

            filenames = glob(
                self.inf_config["InputDir"] + "/*" + self.inf_config["DataType"]
            )
            filenames.sort()
        print("Files to be processed:", filenames)

        self.image_info = {
            "fn": filenames,
            "ijk": [None] * len(filenames),
            "im_shape": [None] * len(filenames),
            "img": [None] * len(filenames),
            "tt": [None] * len(filenames),
        }
        self.save_n_batches = None

    def __getitem__(self, index):
        fn = self.image_info["fn"][index]
        if self.image_info["img"][index] is None:  # image hasn't been loaded yet
            imgs = load_img(fn, self.load_type, self.nchannel, self.config["InputCh"])
            # only one image unless timelapse
            for tt, img in enumerate(imgs):
                img = image_normalization(img, self.config["Normalization"])
                img = resize(img, self.config)
                # generate patch info
                img_info = patchize_wrapper(
                    self.patchize_ratio,
                    fn,
                    img,
                    self.patch_size,
                    tt,
                    self.timelapse,
                )
                self.save_n_batches = img_info["save_n_batches"]
                #                    timelapse     or patchize
                children_generated = len(imgs) > 1 or self.save_n_batches > 1
                if children_generated:
                    # add children to queue of images, remove parent from queue
                    return_dict = {}
                    for key in self.image_info:
                        if tt == 0:
                            del self.image_info[key][index]
                        # return first child image
                        return_dict[key] = img_info[key][0]
                        # add child images next in list
                        self.image_info[key][index + tt : index + tt] = img_info[key]
                else:
                    return_dict = img_info
                return_dict["save_n_batches"] = self.save_n_batches
                return_dict["img"] = pad_image(
                    return_dict["img"],
                    self.model_config["size_in"],
                    self.model_config["size_out"],
                    self.precision,
                )
                return return_dict

        else:
            return {
                "img": pad_image(
                    self.image_info["img"][index],
                    self.model_config["size_in"],
                    self.model_config["size_out"],
                    self.precision,
                ),
                "fn": self.image_info["fn"][index],
                "tt": self.image_info["tt"][index],
                "ijk": self.image_info["ijk"][index],
                "save_n_batches": self.save_n_batches,
                "im_shape": self.image_info["im_shape"][index],
            }

    def __len__(self):
        return len(self.image_info["fn"]) * np.prod(self.patchize_ratio)
