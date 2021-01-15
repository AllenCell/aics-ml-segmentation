import numpy as np
import torch
from monai.inferers import sliding_window_inference


def flip(img: np.ndarray, axis: int, to_tensor=True, inplace=False) -> torch.Tensor:
    """
    flip input img along axis and confert to tensor
    """
    if inplace:
        out_img = img
    else:
        out_img = img.copy()
    for ch_idx in range(out_img.shape[0]):
        str_im = out_img[ch_idx, :, :, :]
        out_img[ch_idx, :, :, :] = np.flip(str_im, axis=axis)
    if to_tensor:
        return torch.as_tensor(out_img.astype(np.float32), dtype=torch.float)
    else:
        return out_img


def apply_on_image(model, input_img, args, squeeze, to_numpy):
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
    if len(input_img.shape) == 4:
        input_img = np.expand_dims(
            input_img, axis=0
        )  # add batch_dimension for sliding window inference

    if not args["RuntimeAug"]:
        input_img = torch.from_numpy(input_img).float()
        return model_inference(model, input_img, args, squeeze, to_numpy)
    else:
        print("doing runtime augmentation")
        input_img_tensor = torch.as_tensor(
            input_img.astype(np.float32), dtype=torch.float
        )
        out0 = model_inference(
            model, input_img_tensor, args, squeeze=False, to_numpy=True
        )

        for i in range(3):
            aug = flip(input_img, axis=i)
            out = model_inference(model, aug, args, squeeze=False, to_numpy=True)
            aug_flip = flip(out, axis=i, to_tensor=False)
            imsave(str(i) + ".tiff", aug_flip)
            out0 += aug_flip

        out0 /= 4

        return out0  # add batch dimension


def model_inference(model, input_img, args, squeeze=False, to_numpy=False):
    print("PERFORMING INFERENCE")
    with torch.no_grad():
        #####HACK UNDO THE .CUDA##############
        result = sliding_window_inference(
            inputs=input_img.cuda(),
            roi_size=args["size_out"],
            sw_batch_size=args["batch_size"],
            predictor=model.forward,
            overlap=0.25,
            mode="gaussian",
            # sigma_scale=0.01,
        )
    if squeeze:
        result = torch.squeeze(result, dim=0)  # remove batch dimension
    if to_numpy:
        result = result.cpu().numpy()
    return result


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])
