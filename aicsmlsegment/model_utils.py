import numpy as np
import torch
from monai.inferers import sliding_window_inference
from torch.autograd import Variable


def flip(img: np.ndarray, axis: int, to_tensor=True):
    """
    Inputs:
        img: image to be flipped
        axis: axis along which to flip image. Should be indexed from the channel dimension
        to_tensor: whether to unsqueeze to (1,C,Z,X,Y) and convert to tensor before returning
    Outputs:
        if to_tensor is True: (1,C,Z,X,Y)-shaped tensor
        if to_tensor is false: numpy array of shape (C,Z,X,Y)
    flip input img along axis
    """

    out_img = img.copy()
    for ch_idx in range(out_img.shape[0]):
        str_im = out_img[ch_idx, :, :, :]
        out_img[ch_idx, :, :, :] = np.flip(str_im, axis=axis)

    if to_tensor:  # used for inference, also have to unsqueeze for sliding window
        return torch.unsqueeze(
            torch.as_tensor(out_img.astype(np.float32), dtype=torch.float), dim=0
        )
    else:
        return out_img


def apply_on_image(
    model,
    input_img: torch.Tensor,
    args: dict,
    squeeze: bool,
    to_numpy: bool,
    sigmoid: bool,
    original_image_shape=None,
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
        if model.name == "BasicUNet":
            return model_inference(model, input_img, args, squeeze, to_numpy, sigmoid)
        elif model.name in ["unet_xy", "unet_xy_zoom"]:
            out = old_model_inference(
                model, input_img, model.final_activation, args, original_image_shape
            )
            out = out.cpu().numpy()
            return out
    else:
        if model.name == "BasicUNet":
            out0 = model_inference(
                model, input_img, args, squeeze=False, to_numpy=True, sigmoid=sigmoid
            )
        elif model.name in ["unet_xy", "unet_xy_zoom"]:
            out0 = old_model_inference(
                model, input_img, model.final_activation, args, original_image_shape
            )
            out0 = out0.cpu().numpy()

        input_img = input_img.cpu().numpy()[0]  # remove batch_dimension for flip
        for i in range(3):
            aug = flip(input_img, axis=i, to_tensor=True)
            if model.name == "BasicUnet":
                out = model_inference(
                    model, aug, args, squeeze=True, to_numpy=True, sigmoid=sigmoid
                )
            elif model.name in ["unet_xy", "unet_xy_zoom"]:
                out = old_model_inference(
                    model, aug, model.final_activation, args, original_image_shape
                )
                out = out.cpu().numpy()
            aug_flip = flip(out, axis=i, to_tensor=False)
            out0 += aug_flip
        # average across flipped predictions
        out0 /= 4
        return out0


def old_model_inference(model, input_img, softmax, args, original_img_size):
    # convert to CZYX
    if len(input_img.shape) == 5:
        input_img = torch.squeeze(input_img, dim=0)  # add batch
    if len(original_img_size) == 5:
        original_img_size = original_img_size[1:]

    # output channels specified for each auxiliary outoput
    if type(args["OutputCh"]) == list and len(args["OutputCh"] > 2):
        output_ch = args["OutputCh"][1]
    else:
        output_ch = args["OutputCh"]

    # shape of image before padding during loading step
    output_img = np.zeros(original_img_size)
    model.eval()

    # loop through the image patch by patch
    num_step_z = int(np.floor(original_img_size[-3] / args["size_out"][0]) + 1)
    num_step_y = int(np.floor(original_img_size[-2] / args["size_out"][1]) + 1)
    num_step_x = int(np.floor(original_img_size[-1] / args["size_out"][2]) + 1)

    with torch.no_grad():
        for ix in range(num_step_x):
            if ix < num_step_x - 1:
                xa = ix * args["size_out"][2]
            else:
                xa = original_img_size[-1] - args["size_out"][2]

            for iy in range(num_step_y):
                if iy < num_step_y - 1:
                    ya = iy * args["size_out"][1]
                else:
                    ya = original_img_size[-2] - args["size_out"][1]

                for iz in range(num_step_z):
                    if iz < num_step_z - 1:
                        za = iz * args["size_out"][0]
                    else:
                        za = original_img_size[-3] - args["size_out"][0]

                    # build the input patch
                    input_patch = input_img[
                        :,
                        za : (za + args["size_in"][0]),
                        ya : (ya + args["size_in"][1]),
                        xa : (xa + args["size_in"][2]),
                    ]

                    tmp_out = model(Variable(input_patch.cuda()).unsqueeze(0))
                    # model outputs list of three logits from each auxiliary head
                    # for inference, we only care about the first one
                    prob = softmax(tmp_out[0])  # label  # softmax(label)
                    out_flat_tensor = prob.cpu().data.float()
                    out_tensor = out_flat_tensor.view(
                        args["size_out"][0],
                        args["size_out"][1],
                        args["size_out"][2],
                        args["nclass"][0],
                    )
                    out_nda = out_tensor.numpy()

                    output_img[
                        :,
                        za : (za + args["size_out"][0]),
                        ya : (ya + args["size_out"][1]),
                        xa : (xa + args["size_out"][2]),
                    ] = out_nda[:, :, :, output_ch]

    output_img = torch.from_numpy(output_img)
    return output_img


def model_inference(
    model,
    input_img,
    args,
    squeeze=False,
    to_numpy=False,
    extract_output_ch=True,
    sigmoid=False,
):
    """
    perform model inference and extract output channel
    """
    with torch.no_grad():
        result = sliding_window_inference(
            inputs=input_img.cuda(),
            roi_size=args["size_in"],
            sw_batch_size=1,
            predictor=model.forward,
            overlap=0.25,
            mode="gaussian",
        )

    if extract_output_ch:
        result = result[:, args["OutputCh"], :, :, :]
    if sigmoid:
        result = torch.nn.Sigmoid()(result)
    if not squeeze:
        result = torch.unsqueeze(result, dim=1)
    if to_numpy:
        result = result.cpu().numpy()
    return result


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
