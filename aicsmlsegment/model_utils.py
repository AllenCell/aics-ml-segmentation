import numpy as np
import torch
from monai.inferers import sliding_window_inference


def apply_on_image(model, input_img, args, squeeze, to_numpy):
    """
    Perform inference on an input img through a model with or without runtime augmentation.
    If runtime augmentation is selected, perform inference on flipped images and average results.
    returns: 5d np.array NCZYX order
    """
    if len(input_img.shape) == 4:
        input_img = np.expand_dims(
            input_img, axis=0
        )  # add batch_dimension for sliding window inference

    if not args.RuntimeAug:
        input_img = torch.from_numpy(input_img).float().cuda()
        return model_inference(model, input_img, args, squeeze, to_numpy)
    else:
        print("doing runtime augmentation")

        input_img_aug = input_img.copy()
        for ch_idx in range(input_img_aug.shape[0]):
            str_im = input_img_aug[ch_idx, :, :, :]
            input_img_aug[ch_idx, :, :, :] = np.flip(str_im, axis=2)

        input_img_aug_tensor = (
            torch.from_numpy(input_img_aug.astype(float)).float().cuda()
        )
        out1 = model_inference(
            model, input_img_aug_tensor, args, squeeze=True, to_numpy=True
        )

        input_img_aug = []
        input_img_aug = input_img.copy()
        for ch_idx in range(input_img_aug.shape[0]):
            str_im = input_img_aug[ch_idx, :, :, :]
            input_img_aug[ch_idx, :, :, :] = np.flip(str_im, axis=1)

        input_img_aug_tensor = (
            torch.from_numpy(input_img_aug.astype(float)).float().cuda()
        )
        out2 = model_inference(
            model, input_img_aug_tensor, args, squeeze=True, to_numpy=True
        )

        input_img_aug = []
        input_img_aug = input_img.copy()
        for ch_idx in range(input_img_aug.shape[0]):
            str_im = input_img_aug[ch_idx, :, :, :]
            input_img_aug[ch_idx, :, :, :] = np.flip(str_im, axis=0)

        input_img_aug_tensor = (
            torch.from_numpy(input_img_aug.astype(float)).float().cuda()
        )
        out3 = model_inference(
            model, input_img_aug_tensor, args, squeeze=True, to_numpy=True
        )

        input_img_tensor = torch.from_numpy(input_img.astype(float)).float().cuda()
        out0 = model_inference(
            model, input_img_tensor, args, squeeze=True, to_numpy=True
        )

        for ch_idx in range(len(out0)):
            out0[ch_idx] = 0.25 * (
                out0[ch_idx]
                + np.flip(out1[ch_idx], axis=2)
                + np.flip(out2[ch_idx], axis=1)
                + np.flip(out3[ch_idx], axis=0)
            )
        return np.expand_dims(out0, axis=0)  # add batch dimension


def model_inference(model, input_img, args, squeeze=False, to_numpy=False):
    with torch.no_grad():
        result = sliding_window_inference(
            inputs=input_img,
            roi_size=args.size_out,
            sw_batch_size=1,
            predictor=model.forward,
            overlap=0.25,
            # mode="gaussian",
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
