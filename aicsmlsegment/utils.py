import numpy as np
import logging
import sys
from typing import List
from aicsimageio import AICSImage
from scipy.ndimage import zoom
from scipy import ndimage as ndi
from scipy import stats
import yaml
import torch
from monai.networks.layers import Norm, Act
from monai.transforms import CropForeground, RandSpatialCrop, RandSpatialCropSamples
import os
import datetime


REQUIRED_CONFIG_FIELDS = {
    True: {
        "model": ["name"],
        "checkpoint_dir": None,
        "learning_rate": None,
        "weight_decay": None,
        "epochs": None,
        "save_every_n_epoch": None,
        "loss": ["name", "loss_weight"],
        "loader": [
            "name",
            "datafolder",
            "batch_size",
            "PatchPerBuffer",
            "NumWorkers",
            "Transforms",
        ],
        "validation": ["metric", "leaveout", "OutputCh", "validate_every_n_epoch"],
    },
    False: {
        "model": ["name"],
        "model_path": None,
        "OutputCh": None,
        "OutputDir": None,
        "InputCh": None,
        "ResizeRatio": None,
        "Normalization": None,
        "Threshold": None,
        "RuntimeAug": None,
        "batch_size": None,
        "mode": ["name"],
        "NumWorkers": None,
    },
}
OPTIONAL_CONFIG_FIELDS = {
    True: {
        "resume": None,
        "scheduler": ["name", "verbose"],
        "gpus": None,
        "dist_backend": None,
        "callbacks": ["name"],
        "SWA": ["swa_start", "swa_lr", "annealing_epochs", "annealing_strategy"],
        "tensorboard": None,
        "precision": None,
        "loader": [
            "epoch_shuffle",
        ],
    },
    False: {
        "gpus": None,
        "dist_backend": None,
        "model": ["norm", "act", "features", "dropout"],
        "large_image_resize": None,
        "precision": None,
        "segmentation_name": None,
    },
}

GPUS = torch.cuda.device_count()
DEFAULT_CONFIG = {
    "SWA": None,
    "resume": None,
    "scheduler": {"name": None},
    "gpus": GPUS,
    "dist_backend": "ddp" if GPUS > 1 else None,
    "tensorboard": None,
    "callbacks": {"name": None},
    "precision": 32,
    "large_image_resize": [1, 1, 1],
    "epoch_shuffle": None,
    "segmentation_name": "segmentation",
}

MODEL_PARAMETERS = {
    "basic_unet": {
        "Optional": [
            "features",
            "act",
            "norm",
            "dropout",
        ],
        "Required": ["dimensions", "in_channels", "out_channels", "patch_size"],
    },
    "unet_xy": {
        "Optional": [],
        "Required": ["nchannel", "nclass", "size_in", "size_out"],
    },
    "unet_xy_zoom": {
        "Optional": [],
        "Required": ["nchannel", "nclass", "size_in", "size_out", "zoom_ratio"],
    },
    "unet_xy_zoom_0pad": {
        "Optional": [],
        "Required": ["nchannel", "nclass", "size_in", "size_out", "zoom_ratio"],
    },
    "unet_xy_zoom_0pad_stridedconv": {
        "Optional": [],
        "Required": ["nchannel", "nclass", "size_in", "size_out", "zoom_ratio"],
    },
    "unet_xy_zoom_0pad_nopadz_stridedconv": {
        "Optional": [],
        "Required": ["nchannel", "nclass", "size_in", "size_out", "zoom_ratio"],
    },
    "unet_xy_zoom_stridedconv": {
        "Optional": [],
        "Required": ["nchannel", "nclass", "size_in", "size_out", "zoom_ratio"],
    },
    "unet_xy_zoom_dilated": {
        "Optional": [],
        "Required": ["nchannel", "nclass", "size_in", "size_out", "zoom_ratio"],
    },
    "sdunet_xy": {
        "Optional": [],
        "Required": ["nchannel", "nclass", "size_in", "size_out"],
    },
    "unet": {
        "Optional": [
            "kernel_size",
            "up_kernel_size",
            "num_res_units",
            "act",
            "norm",
            "dropout",
        ],
        "Required": [
            "dimensions",
            "in_channels",
            "out_channels",
            "channels",
            "strides",
            "patch_size",
        ],
    },
    "dynunet": {
        "Optional": ["norm_name", "deep_supr_num", "res_block", "deep_supervision"],
        "Required": [
            "spatial_dims",
            "in_channels",
            "out_channels",
            "kernel_size",
            "strides",
            "upsample_kernel_size",
            "patch_size",
        ],
    },
    "extended_dynunet": {
        "Optional": ["norm_name", "deep_supr_num", "res_block", "deep_supervision"],
        "Required": [
            "spatial_dims",
            "in_channels",
            "out_channels",
            "kernel_size",
            "strides",
            "upsample_kernel_size",
            "patch_size",
        ],
    },
    "segresnet": {
        "Optional": [
            "dropout_prob",
            "norm_name",
            "num_groups",
            "use_conv_final",
            "blocks_down",
            "blocks_up",
            "upsample_mode",
            "init_filters",
        ],
        "Required": [
            "patch_size",
            "spatial_dims",
            "in_channels",
            "out_channels",
        ],
    },
    "segresnetvae": {
        "Optional": [
            "vae_estimate_std",
            "vae_default_std",
            "vae_nz",
            "init_filters",
            "dropout_prob",
            "norm_name",
            "num_groups",
            "use_conv_final",
            "blocks_down",
            "blocks_up",
            "upsample_mode",
        ],
        "Required": ["patch_size", "spatial_dims", "in_channels", "out_channels"],
    },
    "extended_vnet": {
        "Optional": ["act", "dropout_prob", "dropout_dim"],
        "Required": [
            "spatial_dims",
            "in_channels",
            "out_channels",
            "patch_size",
        ],
    },
}

ACTIVATIONS = {
    "LeakyReLU": Act.LEAKYRELU,
    "PReLU": Act.PRELU,
    "ReLU": Act.RELU,
    "ReLU6": Act.RELU6,
}

NORMALIZATIONS = {
    "batch": Norm.BATCH,
    "instance": Norm.INSTANCE,
}


def get_model_configurations(config):
    model_config = config["model"]
    model_parameters = {}

    assert (
        model_config["name"] in MODEL_PARAMETERS
    ), f"{model_config['name']} is not a supported model name, supported model names are {list(MODEL_PARAMETERS.keys())}"
    all_parameters = MODEL_PARAMETERS[model_config["name"]]

    # allow users to overwrite specific parameters
    for param in all_parameters["Optional"]:
        # if optional parameters are not specified, skip them to use monai defaults
        if param in model_config and not model_config[param] is None:
            if param == "norm":
                try:
                    model_parameters[param] = NORMALIZATIONS[model_config[param]]
                except KeyError:
                    print(f"{model_config[param]} is not an acceptable normalization.")
                    quit()
            elif param == "act":
                try:
                    model_parameters[param] = ACTIVATIONS[model_config[param]]
                except KeyError:
                    print(f"{model_config[param]} is not an acceptable activation.")
                    quit()
            else:
                model_parameters[param] = model_config[param]
    # find parameters that must be included
    for param in all_parameters["Required"]:
        assert (
            param in model_config
        ), f"{param} is required for model {model_config['name']}"
        model_parameters[param] = model_config[param]

    return model_parameters


def validate_config(config, train):
    # make sure that all required elements are in the config file
    for key in REQUIRED_CONFIG_FIELDS[train]:
        assert (
            key in config and not config[key] is None
        ), f"{key} is required in the config file"
        if REQUIRED_CONFIG_FIELDS[train][key]:
            for key2 in REQUIRED_CONFIG_FIELDS[train][key]:
                assert (
                    key2 in config[key] and config[key][key2] is not None
                ), f"{key2} is required in {key} configuration."

    # check for optional elements and replace them with defaults if not provided
    for key in OPTIONAL_CONFIG_FIELDS[train]:
        if key not in config or config[key] is None:
            config[key] = DEFAULT_CONFIG[key]

    if GPUS == 1:
        config["dist_backend"] = None

    model_config = get_model_configurations(config)

    return config, model_config


def load_config(config_file, train):
    config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    config, model_config = validate_config(config, train)
    config["date"] = datetime.datetime.now().strftime("%b %d, %Y %H:%M")
    return config, model_config


def create_unique_run_directory(config, train):
    # directory to check for config in
    subdir_names = {True: "/run_", False: "/prediction_"}
    if train:
        dir_name = config["checkpoint_dir"]
    else:
        dir_name = config["OutputDir"]
    if os.path.exists(dir_name):
        subfolders = [x for x in os.walk(dir_name)][0][1]
        # only look at run_ or prediction_ folders
        run_numbers = [
            int(sub.split("_")[1])
            for sub in subfolders
            if subdir_names[train][1:] in sub
        ]
        if len(subfolders) > 0:
            most_recent_run_number = max(run_numbers)
            most_recent_run_dir = (
                dir_name + subdir_names[train] + str(most_recent_run_number)
            )
            most_recent_config, _ = load_config(
                most_recent_run_dir + "/config.yaml",
                train=train,
            )
            # HACK - this will combine runs with the same config files that are run within a minute of one another.
            # multi gpu case - don't create a new run folder on non-rank 0 gpu
            if most_recent_config == config:
                return most_recent_run_dir
        else:
            most_recent_run_number = 0
    else:
        os.makedirs(dir_name)
        most_recent_run_number = 0

    new_run_folder_name = subdir_names[train] + str(most_recent_run_number + 1)
    os.makedirs(dir_name + new_run_folder_name)
    if train:
        os.makedirs(dir_name + new_run_folder_name + "/validation_results")

    with open(dir_name + new_run_folder_name + "/config.yaml", "w") as config_file:
        yaml.dump(config, config_file, default_flow_style=False)
    return dir_name + new_run_folder_name


def get_samplers(num_training_data, validation_ratio, my_seed):
    from torch.utils.data import sampler as torch_sampler

    indices = list(range(num_training_data))
    split = int(np.floor(validation_ratio * num_training_data))

    np.random.seed(my_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = torch_sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch_sampler.SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler


def simple_norm(img, a, b, m_high=-1, m_low=-1):
    idx = np.ones(img.shape, dtype=bool)
    if m_high > 0:
        idx = np.logical_and(idx, img < m_high)
    if m_low > 0:
        idx = np.logical_and(idx, img > m_low)
    img_valid = img[idx]
    m, s = stats.norm.fit(img_valid.flat)
    strech_min = max(m - a * s, img.min())
    strech_max = min(m + b * s, img.max())
    img[img > strech_max] = strech_max
    img[img < strech_min] = strech_min
    img = (img - strech_min + 1e-8) / (strech_max - strech_min + 1e-8)
    return img


def get_adjusted_min_max(img, a, b, m_high=-1, m_low=-1):
    idx = np.ones(img.shape, dtype=bool)
    if m_high > 0:
        idx = np.logical_and(idx, img < m_high)
    if m_low > 0:
        idx = np.logical_and(idx, img > m_low)
    img_valid = img[idx]
    m, s = stats.norm.fit(img_valid.flat)
    strech_min = max(m - a * s, img.min())
    strech_max = min(m + b * s, img.max())
    return strech_min, strech_max


def background_sub(img, r):
    struct_img_smooth = ndi.gaussian_filter(img, sigma=r, mode="nearest", truncate=3.0)
    struct_img_smooth_sub = img - struct_img_smooth
    struct_img = (struct_img_smooth_sub - struct_img_smooth_sub.min()) / (
        struct_img_smooth_sub.max() - struct_img_smooth_sub.min()
    )
    return struct_img


def input_normalization(img, args):
    nchannel = img.shape[0]
    args.Normalization = int(args.Normalization)
    for ch_idx in range(nchannel):
        struct_img = img[
            ch_idx, :, :, :
        ]  # note that struct_img is only a view of img, so changes made on struct_img also affects img
        if args.Normalization == 0:  # min-max normalization
            struct_img = (struct_img - struct_img.min() + 1e-8) / (
                struct_img.max() - struct_img.min() + 1e-7
            )
        elif args.Normalization == 1:  # mem: DO NOT CHANGE (FIXED FOR CAAX PRODUCTION)
            m, s = stats.norm.fit(struct_img.flat)
            strech_min = max(m - 2 * s, struct_img.min())
            strech_max = min(m + 11 * s, struct_img.max())
            struct_img[struct_img > strech_max] = strech_max
            struct_img[struct_img < strech_min] = strech_min
            struct_img = (struct_img - strech_min + 1e-8) / (
                strech_max - strech_min + 1e-8
            )
            img[ch_idx, :, :, :] = struct_img[:, :, :]
        elif args.Normalization == 2:  # nuc
            # struct_img = simple_norm(struct_img, 2.5, 10, 1000, 300)
            struct_img = simple_norm(struct_img, 2.5, 10)
            img[ch_idx, :, :, :] = struct_img[:, :, :]
        elif args.Normalization == 4:
            struct_img = simple_norm(struct_img, 1, 15)
            img[ch_idx, :, :, :] = struct_img[:, :, :]
        elif args.Normalization == 7:  # cardio_wga
            struct_img = simple_norm(struct_img, 1, 6)
            img[ch_idx, :, :, :] = struct_img[:, :, :]
        elif (
            args.Normalization == 10
        ):  # lamin hipsc, DO NOT CHANGE (FIXED FOR LAMNB1 PRODUCTION)
            img_valid = struct_img[struct_img > 4000]
            m, s = stats.norm.fit(img_valid.flat)
            m, s = stats.norm.fit(struct_img.flat)
            strech_min = struct_img.min()
            strech_max = min(m + 25 * s, struct_img.max())
            struct_img[struct_img > strech_max] = strech_max
            struct_img = (struct_img - strech_min + 1e-8) / (
                strech_max - strech_min + 1e-8
            )
            img[ch_idx, :, :, :] = struct_img[:, :, :]
        elif args.Normalization == 12:  # nuc
            struct_img = background_sub(struct_img, 50)
            struct_img = simple_norm(struct_img, 2.5, 10)
            img[ch_idx, :, :, :] = struct_img[:, :, :]
        elif args.Normalization == 11:
            struct_img = background_sub(struct_img, 50)
            # struct_img = simple_norm(struct_img, 2.5, 10)
            img[ch_idx, :, :, :] = struct_img[:, :, :]
        elif args.Normalization == 13:  # cellmask
            # struct_img[struct_img>10000] = struct_img.min()
            struct_img = background_sub(struct_img, 50)
            struct_img = simple_norm(struct_img, 2, 11)
            img[ch_idx, :, :, :] = struct_img[:, :, :]
        elif args.Normalization == 14:
            struct_img = simple_norm(struct_img, 1, 10)
            img[ch_idx, :, :, :] = struct_img[:, :, :]
        elif args.Normalization == 15:  # lamin
            struct_img[struct_img > 4000] = struct_img.min()
            struct_img = background_sub(struct_img, 50)
            img[ch_idx, :, :, :] = struct_img[:, :, :]
        elif args.Normalization == 16:  # lamin/h2b
            struct_img = background_sub(struct_img, 50)
            struct_img = simple_norm(struct_img, 1.5, 6)
            img[ch_idx, :, :, :] = struct_img[:, :, :]
        elif args.Normalization == 17:  # lamin
            struct_img = background_sub(struct_img, 50)
            struct_img = simple_norm(struct_img, 1, 10)
            img[ch_idx, :, :, :] = struct_img[:, :, :]
        elif args.Normalization == 18:  # h2b
            struct_img = background_sub(struct_img, 50)
            struct_img = simple_norm(struct_img, 1.5, 10)
            img[ch_idx, :, :, :] = struct_img[:, :, :]
        elif args.Normalization == 19:  # EMT lamin
            struct_img[struct_img > 4000] = struct_img.min()
            struct_img = simple_norm(struct_img, 1, 15)
            img[ch_idx, :, :, :] = struct_img[:, :, :]
        elif args.Normalization == 22:
            print("No Normalization")
            img[ch_idx, :, :, :] = struct_img[:, :, :]
        else:
            print("no normalization recipe found")
            quit()
    return img


def image_normalization(img, config):
    if type(config) is dict:
        ops = config["ops"]
        nchannel = img.shape[0]
        assert len(ops) == nchannel
        for ch_idx in range(nchannel):
            ch_ops = ops[ch_idx]["ch"]
            struct_img = img[ch_idx, :, :, :]
            for transform in ch_ops:
                if transform["name"] == "background_sub":
                    struct_img = background_sub(struct_img, transform["sigma"])
                elif transform["name"] == "auto_contrast":
                    param = transform["param"]
                    if len(param) == 2:
                        struct_img = simple_norm(struct_img, param[0], param[1])
                    elif len(param) == 4:
                        struct_img = simple_norm(
                            struct_img, param[0], param[1], param[2], param[3]
                        )
                    else:
                        print("bad paramter for auto contrast")
                        quit()
                else:
                    print(transform["name"])
                    print("other normalization methods are not supported yet")
                    quit()

                img[ch_idx, :, :, :] = struct_img[:, :, :]
    else:
        args_norm = lambda: None
        args_norm.Normalization = config

        img = input_normalization(img, args_norm)

    return img


def load_single_image(args, fn, time_flag=False):

    if time_flag:
        img = fn[:, args.InputCh, :, :]
        img = img.astype(float)
        img = np.transpose(img, axes=(1, 0, 2, 3))
    else:
        data_reader = AICSImage(fn)
        if isinstance(args.InputCh, List):
            channel_list = args.InputCh
        else:
            channel_list = [args.InputCh]
        img = data_reader.get_image_data("CZYX", S=0, T=0, C=channel_list)

    # normalization
    if args.mode == "train":
        for ch_idx in range(args.nchannel):
            struct_img = img[
                ch_idx, :, :, :
            ]  # note that struct_img is only a view of img, so changes made on struct_img also affects img
            struct_img = (struct_img - struct_img.min()) / (
                struct_img.max() - struct_img.min()
            )
    elif not args.Normalization == 0:
        img = input_normalization(img, args)

    # rescale
    if len(args.ResizeRatio) > 0:
        img = zoom(
            img,
            (1, args.ResizeRatio[0], args.ResizeRatio[1], args.ResizeRatio[2]),
            order=1,
        )

    return img


def compute_iou(prediction, gt, cmap):
    if type(prediction) == torch.Tensor:
        prediction = prediction.detach().cpu().numpy()
    if type(gt) == torch.Tensor:
        gt = gt.detach().cpu().numpy()
    if type(cmap) == torch.Tensor:
        cmap = cmap.detach().cpu().numpy()
    if prediction.shape[1] == 2:  # take foreground channel
        prediction = prediction[:, 1, :, :, :]
    if len(prediction.shape) == 4:
        prediction = np.expand_dims(prediction, axis=1)
    if len(cmap.shape) == 3:  # cost function doesn't take cmap
        cmap = np.ones_like(prediction)
    area_i = np.logical_and(prediction, gt)
    area_i[cmap == 0] = False
    area_u = np.logical_or(prediction, gt)
    area_u[cmap == 0] = False

    return np.count_nonzero(area_i) / np.count_nonzero(area_u)


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
