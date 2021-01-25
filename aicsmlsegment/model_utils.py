import numpy as np
import torch
from monai.inferers import sliding_window_inference


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
    model, input_img: torch.Tensor, args: dict, squeeze: bool, to_numpy: bool
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
        return model_inference(model, input_img, args, squeeze, to_numpy)
    else:
        out0 = model_inference(model, input_img, args, squeeze=False, to_numpy=True)

        input_img = input_img.cpu().numpy()[0]  # remove batch_dimension for flip
        for i in range(3):
            aug = flip(input_img, axis=i, to_tensor=True)
            out = model_inference(model, aug, args, squeeze=True, to_numpy=True)
            aug_flip = flip(out, axis=i, to_tensor=False)
            out0 += aug_flip

        out0 /= 4
        return out0


def model_inference(model, input_img, args, squeeze=False, to_numpy=False):
    """
    perform model inference and extract output channel
    """
    with torch.no_grad():
        result = sliding_window_inference(
            inputs=input_img.cuda(),
            roi_size=args["size_out"],
            sw_batch_size=1,
            predictor=model.forward,
            overlap=0.25,
            mode="gaussian",
        )
        result = result[:, args["OutputCh"], :, :, :]
    if not squeeze:
        result = torch.unsqueeze(result, dim=1)
    if to_numpy:
        result = result.cpu().numpy()
    return result
