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


def swap(l, index1, index2):
    temp = l[index1]
    l[index1] = l[index2]
    l[index2] = temp
    return l


def validate_shape(img_shape, n_channel, timelapse):
    img_shape = list(img_shape)
    load_order = ["S", "T", "C", "Z", "Y", "X"]
    expected_channel_idx = 2
    # all dimensions that could be channel dimension
    real_channel_idx = [i for i, elem in enumerate(img_shape) if elem == n_channel]

    keep_channels = ["C"]
    if expected_channel_idx not in real_channel_idx:
        # if nchannels is 1, doesn't matter which other size-1 dim we swap it with
        assert (
            len(real_channel_idx) > 0 and n_channel > 1
        ), "Index of channel dimension is incorrect and there are multiple candidate channel dimensions. Please check your image metadata."

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
    args_dict, correct_shape = validate_shape(
        reader.shape, n_channel, img_type == "timelapse"
    )
    if shape_only:
        return correct_shape
    img = reader.get_image_data(**args_dict)
    if img_type == "costmap":
        img = np.squeeze(img, 0)  # remove channel dimension
    elif img_type == "test":
        # img = img.astype(float)
        img = img[input_ch, :, :, :]
        return [img]  # return as list so we can iterate through it in test dataloader
    elif img_type == "timelapse":
        imgs = []
        for tt in range(correct_shape[1]):
            # Assume:  dimensions = TCZYX
            img = reader.get_image_data("CZYX", S=0, T=tt, C=input_ch).astype(float)
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
        # if init_only:
        #     num_patch = 1
        # print("init only?", init_only)
        # print("GONNA GENERATE", patchize * num_patch, "PATCHES")
        # if not patchize:
        #     print("Val data")
        #     print(filenames)
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

    def __getitem__(self, index):
        # converting to tensor in get item prevents conversion to double tensor
        # and saves gpu memory
        # if index % 7 == 0 and self.patchize:
        #     return (
        #         torch.rand(self.img[index].shape, dtype=torch.float) / 10,
        #         torch.zeros(self.img[index].shape, dtype=torch.float),
        #         torch.ones(self.img[index].shape, dtype=torch.float),
        #     )
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


def patchize_wrapper(pr, fn, img, patch_size, tt, timelapse):
    if pr == [1, 1, 1]:
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
            "tt": [tt] * save_n_batches if timelapse else [-1] * save_n_batches,
        }
    return return_dict


def pad_image(image, size_in, size_out):
    padding = [(x - y) // 2 for x, y in zip(size_in, size_out)]
    print(image.shape)
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


def get_timepoints(filenames):
    timepoints = [load_img(fn, "test", shape_only=True)[1] for fn in filenames]
    return timepoints


class TestDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.inf_config = config["mode"]
        self.model_config = config["model"]
        self.precision = config["precision"]
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

        tp_per_image = [1] * len(filenames)  # get_timepoints(filenames)
        children_per_image = np.array(tp_per_image) * self.patches_per_image
        parent_indices = list(np.cumsum(children_per_image))
        self.total_n_images = parent_indices.pop(-1)
        parent_indices = [0] + parent_indices

        self.image_info = {
            "fn": [None] * self.total_n_images,
            "ijk": [None] * self.total_n_images,
            "im_shape": [None] * self.total_n_images,
            "img": [None] * self.total_n_images,
            "tt": [None] * self.total_n_images,
            "is_parent": [False] * self.total_n_images,
        }

        for idx, fn in zip(parent_indices, filenames):
            self.image_info["is_parent"][idx] = True
            self.image_info["fn"][idx] = fn

        # print(self.image_info["is_parent"])
        self.save_n_batches = None

    def clear_info(self, index):
        # don't have to keep large image info after image has been returned
        for key in self.image_info:
            if key not in ["fn", "is_parent"]:
                self.image_info[key][index] = None

    def __getitem__(self, index):
        # print([fn[-20:] if fn is not None else fn for fn in self.image_info["fn"]])
        fn = self.image_info["fn"][index]
        if self.image_info["img"][index] is None:  # image hasn't been loaded yet
            # # wait for another worker to produce child info
            # if not self.image_info["is_parent"][index]:
            #     timeout = 60 * 1  # 20  minutes for large image normalization
            #     print(index, end=" ")
            #     end_time = time.time() + timeout
            #     print("waiting....")
            #     while self.image_info["img"][index] is None and time.time() < end_time:
            #         time.sleep(1)
            #     if time.time() > end_time:
            #         print(index, "TIMED OUT")
            #     else:
            #         print(index, "OTHER WORKER LOADED IMG")
            #         print(self.image_info["fn"])

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
                    self.size_in,
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
                        if key == "is_parent":
                            continue
                        # return first child image
                        return_dict[key] = img_info[key][0]
                        # add child images next in list
                        start_index = index + tt * self.patches_per_image
                        self.image_info[key][
                            start_index : start_index + len(img_info[key])
                        ] = img_info[key]
                else:
                    return_dict = img_info
                return_dict["save_n_batches"] = self.save_n_batches
                return_dict["img"] = pad_image(
                    return_dict["img"], self.size_in, self.size_out
                )
                # print("Returning parent at", index)
                self.clear_info(index)
                return return_dict

        else:
            # print("Returning child at", index)
            return_dict = {
                "img": pad_image(
                    self.image_info["img"][index], self.size_in, self.size_out
                ),
                "fn": self.image_info["fn"][index],
                "tt": self.image_info["tt"][index],
                "ijk": self.image_info["ijk"][index],
                "save_n_batches": self.save_n_batches,
                "im_shape": self.image_info["im_shape"][index],
            }
            self.clear_info(index)
            return return_dict

    def __len__(self):
        return self.total_n_images
