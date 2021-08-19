import numpy as np
import torch
from pathlib import Path, PurePosixPath
from aicsmlsegment.multichannel_sliding_window import sliding_window_inference
from aicsmlsegment.fnet_prediction_torch import predict_piecewise


def flip(img: np.ndarray, axis: int) -> torch.Tensor:
    """
    Inputs:
        img: image to be flipped
        axis: axis along which to flip image. Should be indexed from the channel
                dimension
    Outputs:
        (1,C,Z,Y,X)-shaped tensor
    flip input img along axis
    """
    out_img = img.detach().clone()
    for ch_idx in range(out_img.shape[0]):
        str_im = out_img[ch_idx, :, :, :]
        out_img[ch_idx, :, :, :] = torch.flip(str_im, dims=[axis])

    return torch.unsqueeze(out_img, dim=0)


def apply_on_image(
    model,
    input_img: torch.Tensor,
    args: dict,
    squeeze: bool,
    to_numpy: bool,
    softmax: bool,
    model_name,
    extract_output_ch: bool,
) -> np.ndarray:
    """
    Highest level API to perform inference on an input image through a model with
    or without runtime augmentation. If runtime augmentation is selected (via
    "RuntimeAug" in config yaml file), perform inference on both original image
    and flipped images (3 version flipping along X, Y, Z) and average results.

    Inputs:
        model: pytorch model with a forward method
        model_name: the name of the model
        input_img: tensor that model should be run on
        args: Object containing inference arguments
            RuntimeAug: boolean, if True inference is run on each of 4 flips
                and final output is averaged across each of these augmentations
            SizeOut: size of sliding window for inference
            OutputCh: channel to extract label from
        squeeze: boolean, if true removes the batch dimension in the output image
        to_numpy: boolean, if true converts output to a numpy array and send to cpu

    Returns: 4 or 5 dimensional numpy array or tensor with result of model.forward
             on input_img
    """

    if not args["RuntimeAug"]:
        return model_inference(
            model,
            input_img,
            args,
            model_name=model_name,
            squeeze=squeeze,
            to_numpy=to_numpy,
            extract_output_ch=extract_output_ch,
            softmax=softmax,
        )
    else:
        out0, vae_loss = model_inference(
            model,
            input_img,
            args,
            squeeze=False,
            to_numpy=False,
            softmax=softmax,
            model_name=model_name,
        )
        input_img = input_img[0]  # remove batch_dimension for flip
        for i in range(3):
            aug = flip(input_img, axis=i)
            out, loss = model_inference(
                model,
                aug,
                args,
                squeeze=True,
                to_numpy=False,
                softmax=softmax,
                model_name=model_name,
                extract_output_ch=extract_output_ch,
            )
            aug_flip = flip(out, axis=i)
            out0 += aug_flip
            vae_loss += loss

        out0 /= 4
        vae_loss /= 4
        if to_numpy:
            out0 = out0.detach().cpu().numpy()
        return out0, vae_loss


def get_supported_model_names():
    print(Path(__file__).parent)
    net_list = sorted(Path(__file__).parent.glob("./NetworkArchitecture/*.py"))
    all_names = [PurePosixPath(p.as_posix()).stem for p in net_list]

    # clean up names case by case for current models, future models will need to
    # use module name
    all_names.remove("vnet")
    all_names.append("extended_vnet")

    all_names.remove("dynunet")
    all_names.append("extended_dynunet")

    from inspect import ismodule, getmembers
    import monai.networks.nets as nets

    flist = [o[0] for o in getmembers(nets) if ismodule(o[1])]
    known_unsupport = [
        "autoencoder",
        "classifier",
        "fullyconnectednet",
        "generator",
        "regressor",
        "torchvision_fc",
        "varautoencoder",
    ]
    for mname in known_unsupport:
        flist.remove(mname)

    flist = ["monai.networks.nets" + v for v in flist]

    # only special case
    all_names.append("segresnetvae")
    print(all_names + flist)


def model_inference(
    model,
    input_img: torch.Tensor,
    args,
    model_name: str,
    squeeze: bool = False,
    to_numpy: bool = False,
    extract_output_ch: bool = True,
    softmax: bool = False,
):
    """
    perform model inference and extract output channel
    """
    if args["size_in"] == args["size_out"]:
        dims_max = [0] + args["size_in"]
        overlaps = [int(0.1 * dim) for dim in dims_max]
        result = predict_piecewise(
            model,
            input_img[0],
            dims_max=dims_max,
            overlaps=overlaps,
        )
        for i in range(input_img.shape[0]):
            output = predict_piecewise(
                model,
                input_img[i],
                dims_max=dims_max,
                overlaps=overlaps,
                mode="fast",
            )
            if i == 0:
                result = output
            else:
                result = torch.cat((result, output), dim=0)
        vae_loss = 0
    else:
        input_image_size = np.array((input_img.shape)[-3:])
        added_padding = np.array(
            [2 * ((x - y) // 2) for x, y in zip(args["size_in"], args["size_out"])]
        )
        original_image_size = input_image_size - added_padding
        with torch.no_grad():
            result, vae_loss = sliding_window_inference(
                inputs=input_img,
                roi_size=args["size_in"],
                out_size=args["size_out"],
                original_image_size=original_image_size,
                sw_batch_size=1,
                predictor=model.forward,
                overlap=0.25,
                mode="gaussian",
                model_name=model_name,
            )
    if softmax:
        result = torch.nn.Softmax(dim=1)(result)
    if extract_output_ch:
        # old models
        if type(args["OutputCh"]) == list and len(args["OutputCh"]) >= 2:
            args["OutputCh"] = args["OutputCh"][1]
        result = result[:, args["OutputCh"], :, :, :]
    if not squeeze:
        result = torch.unsqueeze(result, dim=1)
    if to_numpy:
        result = result.detach().cpu().numpy()
        # if uncertainty_map is not None: uncertainty_map = uncertainty_map.detach().cpu().numpy()
    # if uncertainty_map is not None:
    #     return result, vae_loss, uncertainty_map
    return result, vae_loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv3d") != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()


def load_checkpoint(checkpoint_path, model):
    """Loads model from a given checkpoint_path, included for backwards compatibility
    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
    Returns:
        state
    """
    import os

    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    # device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    state = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    if "model_state_dict" in state:
        try:
            model.load_state_dict(state["model_state_dict"])
        except RuntimeError:
            # HACK all keys need "model." appended to them sometimes
            new_state_dict = {}
            for key in state["model_state_dict"]:
                new_state_dict["model." + key] = state["model_state_dict"][key]
            model.load_state_dict(new_state_dict)
    elif "state_dict" in state:
        try:
            model.load_state_dict(state["state_dict"])
        except RuntimeError:
            # HACK all keys need "model." appended to them sometimes
            new_state_dict = {}
            for key in state["state_dict"]:
                new_state_dict["model." + key] = state["state_dict"][key]
            model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state)

    # TODO: add an option to load training status

    return state

def add_dropout_layer(model, dropout_rate, batchnorm_flag):
    """
    Add dropout out layer in an existing model
    Args:
        model: existing model, can only be unet at this moment
        dropout_rate: dropout rate, 0 means no dropout
        batchnorm_flag: whether the existing model has batchnorm_flag, this will affect where to insert the dropout layer
    Returns:
        modified model
    """
    target_modules = ['ec3', 'ec4', 'dc3', 'dc2']
    # we want to add dropout layer to the ec3, ec4 and dc3, dc4 module
    for module_name in target_modules:
        module = getattr(model, module_name)
        layers = list(module.children())
        if batchnorm_flag:
            layers.insert(3, torch.nn.Dropout3d(p=dropout_rate, inplace=True))
            layers.insert(7, torch.nn.Dropout3d(p=dropout_rate, inplace=True))
        else:
            layers.insert(2, torch.nn.Dropout3d(p=dropout_rate, inplace=True))
            layers.insert(5, torch.nn.Dropout3d(p=dropout_rate, inplace=True))
        module = torch.nn.Sequential(*layers)
        setattr(model, module_name, module)
    return model