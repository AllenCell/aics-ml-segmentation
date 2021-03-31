import numpy as np
import torch

from aicsmlsegment.multichannel_sliding_window import sliding_window_inference


def flip(img: np.ndarray, axis: int):
    """
    Inputs:
        img: image to be flipped
        axis: axis along which to flip image. Should be indexed from the channel
                dimension
    Outputs:
        (1,C,Z,X,Y)-shaped tensor
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
    sigmoid: bool,
    model_name,
    extract_output_ch: bool,
) -> np.ndarray:
    """
    Inputs:
        model: pytorch model with a forward method
        input_img: tensor that model should be run on
        args: Object containing inference arguments
            RuntimeAug: boolean, if True inference is run on each of 4 flips
                and final output is averaged across each of these augmentations
            SizeOut: size of sliding window for inference
            OutputCh: channel to extract label from
        squeeze: boolean, if true removes the batch dimension in the output image
        to_numpy: boolean, if true converts output to a numpy array and send to cpu

    Perform inference on an input img through a model with or without runtime augmentation.
    If runtime augmentation is selected, perform inference on flipped images and average results.
    returns: 4 or 5 dimensional numpy array or tensor with result of model.forward on input_img
    """

    if not args["RuntimeAug"]:
        return model_inference(
            model,
            input_img,
            args,
            model_name,
            squeeze,
            to_numpy,
            extract_output_ch,
            sigmoid,
        )
    else:
        out0, vae_loss = model_inference(
            model,
            input_img,
            args,
            squeeze=False,
            to_numpy=False,
            sigmoid=sigmoid,
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
                sigmoid=sigmoid,
                model_name=model_name,
                extract_output_ch=extract_output_ch,
            )
            aug_flip = flip(out, axis=i)
            out0 += aug_flip
            vae_loss += loss

        out0 /= 4
        vae_loss /= 4
        if to_numpy:
            out0 = out0.cpu().detach().numpy()
        return out0, vae_loss


def model_inference(
    model,
    input_img,
    args,
    model_name,
    squeeze=False,
    to_numpy=False,
    extract_output_ch=True,
    sigmoid=False,
    softmax=False,
):
    """
    perform model inference and extract output channel
    """
    input_image_size = np.array((input_img.shape)[-3:])
    added_padding = np.array(
        [2 * ((x - y) // 2) for x, y in zip(args["size_in"], args["size_out"])]
    )
    original_image_size = input_image_size - added_padding
    with torch.no_grad():
        result, vae_loss = sliding_window_inference(
            inputs=input_img.cuda(),
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
        result = torch.nn.Softmax()(result)

    if extract_output_ch:
        # old models
        if type(args["OutputCh"]) == list and len(args["OutputCh"]) >= 2:
            args["OutputCh"] = args["OutputCh"][1]
        result = result[:, args["OutputCh"], :, :, :]

    if sigmoid:
        result = torch.nn.Sigmoid()(result)
    if not squeeze:
        result = torch.unsqueeze(result, dim=1)
    if to_numpy:
        result = result.cpu().detach().numpy()
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
    else:
        model.load_state_dict(state)

    # TODO: add an option to load training status

    return state
