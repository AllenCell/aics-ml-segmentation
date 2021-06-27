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
from typing import Dict, List, Sequence, Tuple


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


def minmax(img: np.ndarray) -> np.ndarray:
    """
    Performs minmax normalization on an image

    Parameters
    ----------
    img: numpy array

    Return: minmaxed numpy array
    """
    return (img - img.min()) / (img.max() - img.min())


def resize(img: np.ndarray, config: Dict, min_max: bool = False) -> np.ndarray:
    """
    Resize an image based on the provided config.

    Parameters
    ----------
    img: 4d CZYX order numpy array
    config: user-provided configuration file with "ResizeRatio" provided
    min_max: whether to conduct minmax normalization on each channel independently

    Return: resized + minmaxed img if specified
    """
    if len(config["ResizeRatio"]) > 0 and config["ResizeRatio"] != [
        1.0,
        1.0,
        1.0,
    ]:
        # don't resize if resize ratio is all 1s
        # note that struct_img is only a view of img, so changes made on
        # struct_img also affects img
        assert len(img.shape) == 4, f"Expected 4D image, got {len(img.shape)}-D array"
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


def undo_resize(img: np.ndarray, config: Dict):
    """
    Undo Resizing an image based on the provided config.

    Parameters
    ----------
    img: 5d NCZYX order numpy array
    config: user-provided configuration file with "ResizeRatio" provided

    Return: float 32 numpy array img resized to its original dimensions
    """
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


def swap(ll: List, index1: int, index2: int) -> List:
    """
    Swap index1 and index2 of list L

    Parameters
    ----------
    l: list
    index1, index2: integer indices to swap

    Return: List l with index1 and index2 swapped
    """
    temp = ll[index1]
    ll[index1] = ll[index2]
    ll[index2] = temp
    return ll


def validate_shape(
    img_shape: Tuple[int],
    n_channel: int = 1,
    timelapse: bool = False,
) -> Tuple[Dict, Tuple[Sequence[int]]]:
    """
    General function to load and rearrange the dimensions of 3D images
    input:
        img_shape: STCZYX shape of image
        n_channel: number of channels expted in image
        timelapse: whether image is a timelapse

    output:
        load_dict: dictionary to be passed to AICSImage.get_image_data
                   containing out_orientation and specific channel indices
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
        ), f"The specified channel dim is wrong, no other dims have size {n_channel}"
        # if nchannels is 1, doesn't matter which other size-1 dim we swap it with
        assert n_channel == 1 or len(real_channel_idx) == 1, (
            "Index of channel dimension is incorrect and there are multiple candidate "
            f"channel dimensions. Please check your image metadata. {img_shape}"
        )

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


def load_img(
    filename: str,
    img_type: str,
    n_channel: int = 1,
    input_ch: int = None,
    shape_only: bool = False,
) -> List[np.ndarray]:
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
        if img_type == "test":
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
        filenames: Sequence[str],
        num_patch: int,
        size_in: Sequence[int],
        size_out: Sequence[int],
        n_channel: int,
        use_costmap: bool = True,
        transforms: Sequence[str] = [],
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
        num_data = len(filenames)
        shuffle(filenames)
        self.filenames = None
        if not patchize:
            print("Validating on", filenames)
            self.filenames = filenames
        num_patch_per_img = np.zeros((num_data,), dtype=int)
        if num_data >= num_patch:
            # take one patch from each image
            num_patch_per_img[:num_patch] = 1
        else:  # assign how many patches to take form each img
            basic_num = num_patch // num_data
            # assign each image the same number of patches to extract
            num_patch_per_img[:] = basic_num

            # assign 1 more patch to the first few images to get the total patch number
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
        if self.filenames is not None:
            fn = self.filenames[index]
        else:
            fn = ""
        img_tensor = from_numpy(self.img[index].astype(float)).float()
        gt_tensor = from_numpy(self.gt[index].astype(float)).float()
        cmap_tensor = from_numpy(self.cmap[index].astype(float)).float()

        return (img_tensor, gt_tensor, cmap_tensor, fn)

    def __len__(self):
        if self.init_only:
            return self.num_patch
        return len(self.img)

    def get_params(self):
        return self.parameters


from monai.transforms import (
    MapTransform,
    RandFlipd,
    RandBiasFieldd,
    RandGaussianNoised,
    RandShiftIntensityd,
    RandSpatialCropSamplesd,
    RandSpatialCropd,
    ToTensord,
    Compose,
)


class LoadImageD(MapTransform):
    def __init__(self, use_costmap, n_channel):
        super(LoadImageD).__init__()
        self.use_costmap = use_costmap
        self.n_channel = n_channel

    def __call__(self, data):
        img_data = {}
        img_data["label"] = load_img(data, img_type="label", n_channel=self.n_channel)
        img_data["img"] = load_img(data, img_type="input", n_channel=self.n_channel)
        if self.use_costmap:
            img_data["costmap"] = load_img(
                data, img_type="costmap", n_channel=self.n_channel
            )
        return img_data


class PadImageD(MapTransform):
    def __init__(self, padding, keys):
        super(PadImageD).__init__()
        self.padding = padding
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            data[key] = np.pad(
                data[key],
                (
                    (0, 0),
                    (0, 0),
                    (self.padding[1], self.padding[1]),
                    (self.padding[2], self.padding[2]),
                ),
                "symmetric",
            )
            data[key] = np.pad(
                data[key],
                ((0, 0), (self.padding[0], self.padding[0]), (0, 0), (0, 0)),
                "constant",
            )
        return data


class RandomPatchesD(MapTransform):
    def __init__(self, check_crop, size_in, size_out, keys, num_patch):
        super(RandomPatchesD).__init__()
        self.check_crop = check_crop
        self.size_in = size_in
        self.size_out = size_out
        self.keys = keys
        self.num_patch = num_patch

    def __call__(self, data):
        if self.check_crop:
            cropper = RandSpatialCropd(self.keys, self.size_in, random_size=False)
            num_fail = 0
            n_patches = 0
            while n_patches < self.num_patch:
                additional_data = cropper(data)
                if np.count_nonzero(additional_data["costmap"] > 1e-5) < 1000:
                    num_fail += 1
                    assert num_fail < 50, "Failed to generate valid crops."
                else:
                    for key in additional_data:
                        data[key] += additional_data[key]
                    n_patches += 1
        else:
            cropper = RandSpatialCropSamplesd(
                self.keys, self.size_in, self.num_patch, random_size=False
            )
            data = cropper(data)
        # crop costmap (if available) and label to match model output size
        if self.size_in != self.size_out:
            for key in self.keys:
                if key == "img":
                    continue
                for i in range(len(data[key])):
                    data[key][i] = data[key][i][
                        0 : self.size_in[0], 0 : self.size_in[1], 0 : self.size_in[2]
                    ]

        return data


class RandomRotationD(MapTransform):
    def __init__(self, use_costmap):
        super(RandomRotationD).__init__()
        self.use_costmap = use_costmap

    def __call__(self, data):
        deg = random.randrange(1, 180)
        trans = RandomAffine(
            scales=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            degrees=(0, 0, 0, 0, deg, deg),
            default_pad_value=0,
            image_interpolation="bspline",
            center="image",
        )
        out_img = trans(np.transpose(data["img"], (0, 3, 2, 1)))
        data["img"] = np.transpose(out_img, (0, 3, 2, 1))

        trans_label = RandomAffine(
            scales=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            degrees=(0, 0, 0, 0, deg, deg),
            default_pad_value=0,
            image_interpolation="nearest",
            center="image",
        )
        # rotate label and costmap
        out_label = trans_label(np.transpose(data["label"], (0, 3, 2, 1)))
        data["label"] = np.transpose(out_label, (0, 3, 2, 1))
        if self.use_costmap:
            out_map = trans_label(np.transpose(data["costmap"], (0, 3, 2, 1)))
            data["costmap"] = np.transpose(out_map, (0, 3, 2, 1))

        return data


class UniversalDataset_redo_transforms(Dataset):
    """
    Multipurpose dataset for training and validation. Randomly crops images, labels,
    and costmaps into user-specified number of patches. Users can specify which
    augmentations to apply.
    """

    def __init__(
        self,
        filenames: Sequence[str],
        num_patch: int,
        size_in: Sequence[int],
        size_out: Sequence[int],
        n_channel: int,
        use_costmap: bool = True,
        transforms: Sequence[str] = [],
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
        self.data = {"img": [], "label": []}
        if use_costmap:
            self.data["costmap"] = []
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
        self.init_only = init_only
        if init_only:
            num_patch = 1
        num_data = len(filenames)
        shuffle(filenames)
        self.filenames = None
        if not patchize:
            print("Validating on", filenames)
            self.filenames = filenames
        num_patch_per_img = np.zeros((num_data,), dtype=int)
        print(filenames)
        print(num_data, num_patch)
        if num_data >= num_patch:
            # take one patch from each image
            num_patch_per_img[:num_patch] = 1
        else:  # assign how many patches to take form each img
            basic_num = num_patch // num_data
            # assign each image the same number of patches to extract
            num_patch_per_img[:] = basic_num

            # assign 1 more patch to the first few images to get the total patch number
            num_patch_per_img[: (num_patch - basic_num * num_data)] = (
                num_patch_per_img[: (num_patch - basic_num * num_data)] + 1
            )

        self.padding = [(x - y) // 2 for x, y in zip(size_in, size_out)]

        # basepath = "//allen/aics/assay-dev/users/Benji/transform_test/"
        # extract patches from images until num_patch reached
        import time

        t1 = time.time()
        tsfrm = self.select_transforms(160)
        for count, (fn, n_patch) in enumerate(zip(filenames, num_patch_per_img)):
            if n_patch == 0 or count > num_patch:
                break
            aug_data = tsfrm(fn)
            for key in self.data:  # costmap, img, label
                for i in aug_data:  # each patch
                    self.data[key] += i[key]
        print(time.time() - t1)

    def select_transforms(self, num_patch):
        all_keys = ["img", "label"]
        params = self.parameters
        if params["use_costmap"]:
            all_keys.append("costmap")
        transform_fns = [
            LoadImageD(
                use_costmap=params["use_costmap"], n_channel=params["n_channel"]
            ),
            PadImageD(self.padding, keys=["img"]),
        ]
        if "RF" in params["transforms"]:
            flipper = RandFlipd(keys=all_keys, prob=1, spatial_axis=-1)
            transform_fns.append(flipper)
        if "RR" in params["transforms"]:
            transform_fns.append(RandomRotationD(use_costmap=params["use_costmap"]))
        if "RBF" in params["transforms"]:
            transform_fns.append(RandBiasFieldd(keys=["img"], coeff_range=(0.0, 0.01)))
        if "RN" in params["transforms"]:
            transform_fns.append(RandGaussianNoised(keys=["img"], std=0.001))
        if "RI" in params["transforms"]:
            transform_fns.append(RandShiftIntensityd(keys=["img"], offsets=0.08))
        if params["patchize"]:
            transform_fns.append(
                RandomPatchesD(
                    check_crop=params["check_crop"],
                    size_in=params["size_in"],
                    size_out=params["size_out"],
                    keys=all_keys,
                    num_patch=num_patch,
                )
            )
        transform_fns.append(ToTensord(keys=all_keys))
        return Compose(transform_fns)

    def __getitem__(self, index):
        if self.init_only:
            return torch.zeros(0)
        if self.filenames is not None:
            fn = self.filenames[index]
        else:
            fn = ""

        if self.parameters["use_costmap"]:
            costmap = self.data["costmap"][index]
        else:
            costmap = []

        return (self.data["img"][index], self.data["label"][index], costmap, fn)

    def __len__(self):
        if self.init_only:
            return self.num_patch
        return len(self.data["img"])

    def get_params(self):
        return self.parameters


def patchize(
    img: np.ndarray, pr: Sequence[int], patch_size: Sequence[int]
) -> Tuple[List[List[int]], List[np.ndarray]]:
    """
    Break an image into z * y * x patches specified by pr

    Parameters
    ----------
    img: 4d CZYX order numpy array
    pr: length 3 list specifying number of patches to divide in z,y,x dimensions
    patch_size: inference patch size to make sure that the patches are large
                enough for inference

    Return: list of [i,j,k] start points of a patch and corresponding list of
            np.array imgs
    """
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
    assert len(img.shape) == 4, f"Expected 4D image, got {len(img.shape)}-D array"
    assert len(pr) == 3, f"Expected pr to have length 3, got length {len(pr)}"

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


# TODO deal with timelapse images
class TestDataset(IterableDataset):
    def __init__(self, config: Dict, fns=List[str]):
        """
        Dataset to load, resize, normalize, and return testing images when needed for
        inference

        Parameters
        ----------
        config: user-provided preferences to specify how to shape and normalize images
                in preparation for prediction
        Return: None
        """
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
        elif self.inf_config["name"] == "cv":
            filenames = []
            for fn in fns:
                filenames.append(fn + ".ome.tif")
        else:
            from glob import glob

            if type(self.inf_config["InputDir"]) == str:
                self.inf_config["InputDir"] = [self.inf_config["InputDir"]]

            filenames = []
            for folder in self.inf_config["InputDir"]:
                fns = glob(folder + "/*" + self.inf_config["DataType"])
                fns.sort()
                filenames += fns
        print("Predicting on", len(filenames), "files")

        self.filenames = filenames
        self.start = None
        self.end = None
        self.all_img_info = []

    def pad_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Pad image so model output is the same size as the input. Padding is symmetric
        in xy and constant in z

        Parameters
        ----------
        image: 4d CZYX order numpy array

        Return: padded image
        """
        if len(image.shape) == 5:
            image = np.squeeze(image, axis=0)
        padding = [(x - y) // 2 for x, y in zip(self.size_in, self.size_out)]
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

    def patchize_wrapper(
        self,
        pr: Sequence[int],
        fn: str,
        img: np.ndarray,
        patch_size: Tuple,
        tt: int,
        timelapse: bool,
    ) -> Dict:
        """
        Create dictionary with information necessary for inference

        Parameters
        ----------
        fn: filename
        img: 4d CZYX order numpy array
        pr: length 3 list specifying number of patches to divide in z,y,x dimensions
        patch_size: inference patch size to make sure that the patches are large enough
                    for inference
        tt: timepoint
        timelapse: whether image is a timelapse

        Return: Dictionary containing image filename, tensor image, shape of input
                image, ijk index of patch from original image, how many patches
                original image was split into, and timepoint
        """
        if pr == [1, 1, 1]:
            return_dicts = [
                {
                    "fn": fn,
                    "img": self.pad_image(img),
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
                    "img": self.pad_image(patch),
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
        if len(self.all_img_info) == 0:  # load new image if no images have been loaded
            fn = self.filenames[self.current_index]
            imgs = load_img(fn, self.load_type, self.nchannel, self.config["InputCh"])
            # only one image unless timelapse
            for tt, img in enumerate(imgs):
                img = resize(img, self.config)
                img = image_normalization(img, self.config["Normalization"])
                # generate patch info
                self.all_img_info += self.patchize_wrapper(
                    self.patchize_ratio,
                    fn,
                    img,
                    self.size_in,
                    tt,
                    self.timelapse,
                )
            self.current_index += 1  # next iteration load the next file
        return self.all_img_info.pop()  # pop patch/tp
