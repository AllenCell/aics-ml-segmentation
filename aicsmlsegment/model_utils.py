import numpy as np
import torch
from monai.inferers import sliding_window_inference


def flip(img: np.ndarray, axis: int, to_tensor=True) -> torch.Tensor:
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
    model, input_img, args: dict, squeeze: bool, to_numpy: bool
) -> np.ndarray:
    """
    Inputs:
        model: pytorch model with a forward method
        input_img: numpy array that model should be run on
        args: Object containing inference arguments
            RuntimeAug: boolean, if True inference is run on each of 4 flips
                and final output is averaged across each of these augmentations
            SizeOut: size of sliding window for inference
        squeeze: boolean, if true removes the batch dimension in the output image
        to_numpy: boolean, if true converts output to a numpy array and send to cpu

    Perform inference on an input img through a model with or without runtime augmentation.
    If runtime augmentation is selected, perform inference on flipped images and average results.
    returns: 4 or 5 dimensional numpy array or tensor with result of model.forward on input_img
    """
    if type(input_img) == np.ndarray:
        input_img = np.expand_dims(
            input_img, axis=0
        )  # add batch_dimension for sliding window inference
        input_img = torch.from_numpy(input_img).float()

    if not args["RuntimeAug"]:
        return model_inference(model, input_img, args, squeeze, to_numpy)
    else:
        out0 = model_inference(model, input_img, args, squeeze=False, to_numpy=True)

        input_img = input_img.cpu().numpy()[0]  # remove batch_dimension
        for i in range(3):
            aug = flip(input_img, axis=i)
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
            sw_batch_size=args["inference_batch_size"],
            predictor=model.forward,
            overlap=0.25,
            mode="gaussian",
        )
        result = result[:, args["OutputCh"], :, :, :]
    if not squeeze:
        result = torch.unsqueeze(result, dim=0)  # remove batch dimension
    if to_numpy:
        result = result.cpu().numpy()
    return result


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])
