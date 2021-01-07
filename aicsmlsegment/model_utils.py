import numpy as np
import torch
from torch.autograd import Variable
from monai.inferers import sliding_window_inference


def apply_on_image(model, input_img, softmax, args):
    # print("apply on image")
    if len(input_img.shape) == 4:
        input_img = np.expand_dims(input_img, axis=0)  # add batch_dimension

    if not args.RuntimeAug:
        input_img = torch.from_numpy(input_img).float().cuda()
        return model_inference(model, input_img, softmax, args)
    else:
        print("doing runtime augmentation")

        input_img_aug = input_img.copy()
        for ch_idx in range(input_img_aug.shape[0]):
            str_im = input_img_aug[ch_idx, :, :, :]
            input_img_aug[ch_idx, :, :, :] = np.flip(str_im, axis=2)

        input_img_aug_tensor = torch.from_numpy(input_img_aug.astype(float)).float()
        # out1 = model_inference(model, input_img_aug_tensor, softmax, args)
        out1 = sliding_window_inference(
            input_img_aug_tensor, args.size_out, 1, model.forward
        )

        input_img_aug = []
        input_img_aug = input_img.copy()
        for ch_idx in range(input_img_aug.shape[0]):
            str_im = input_img_aug[ch_idx, :, :, :]
            input_img_aug[ch_idx, :, :, :] = np.flip(str_im, axis=1)

        input_img_aug_tensor = torch.from_numpy(input_img_aug.astype(float)).float()
        # out2 = model_inference(model, input_img_aug_tensor, softmax, args)
        out2 = sliding_window_inference(
            input_img_aug_tensor, args.size_out, 1, model.forward
        )

        input_img_aug = []
        input_img_aug = input_img.copy()
        for ch_idx in range(input_img_aug.shape[0]):
            str_im = input_img_aug[ch_idx, :, :, :]
            input_img_aug[ch_idx, :, :, :] = np.flip(str_im, axis=0)

        input_img_aug_tensor = torch.from_numpy(input_img_aug.astype(float)).float()
        # out3 = model_inference(model, input_img_aug_tensor, softmax, args)
        out3 = sliding_window_inference(
            input_img_aug_tensor, args.size_out, 1, model.forward
        )

        input_img_tensor = torch.from_numpy(input_img.astype(float)).float()
        # out0 = model_inference(model, input_img_tensor, softmax, args)
        out0 = sliding_window_inference(
            input_img_tensor, args.size_out, 1, model.forward
        )

        for ch_idx in range(len(out0)):
            out0[ch_idx] = 0.25 * (
                out0[ch_idx]
                + np.flip(out1[ch_idx], axis=3)
                + np.flip(out2[ch_idx], axis=2)
                + np.flip(out3[ch_idx], axis=1)
            )

        return out0


def model_inference(model, input_img, softmax, args):
    with torch.no_grad():
        result = sliding_window_inference(
            inputs=input_img,
            roi_size=args.size_out,
            sw_batch_size=1,
            predictor=model.forward,
            overlap=0.25,
        )
    return result

    model.eval()

    if args.size_in == args.size_out:
        # print("expand dims")
        # img_pad = np.expand_dims(input_img, axis=0)  # add batch dimension
        if not type(input_img) is np.ndarray:
            img_pad = input_img.numpy()
        else:
            img_pad = input_img
    else:  # zero padding on input image
        padding = [(x - y) // 2 for x, y in zip(args.size_in, args.size_out)]
        img_pad0 = np.pad(
            input_img,
            ((0, 0), (0, 0), (padding[1], padding[1]), (padding[2], padding[2])),
            "symmetric",  # 'constant')
        )

        img_pad = np.pad(
            img_pad0, ((0, 0), (padding[0], padding[0]), (0, 0), (0, 0)), "constant"
        )

    output_img = np.zeros(input_img.shape)

    # loop through the image patch by patch
    num_step_z = int(np.floor(input_img.shape[1] / args.size_out[0]) + 1)
    num_step_y = int(np.floor(input_img.shape[2] / args.size_out[1]) + 1)
    num_step_x = int(np.floor(input_img.shape[3] / args.size_out[2]) + 1)

    with torch.no_grad():
        for ix in range(num_step_x):
            if ix < num_step_x - 1:
                xa = ix * args.size_out[2]
            else:
                xa = input_img.shape[3] - args.size_out[2]

            for iy in range(num_step_y):
                if iy < num_step_y - 1:
                    ya = iy * args.size_out[1]
                else:
                    ya = input_img.shape[2] - args.size_out[1]

                for iz in range(num_step_z):
                    if iz < num_step_z - 1:
                        za = iz * args.size_out[0]
                    else:
                        za = input_img.shape[1] - args.size_out[0]

                    # build the input patch
                    input_patch = img_pad[
                        :,
                        za : (za + args.size_in[0]),
                        ya : (ya + args.size_in[1]),
                        xa : (xa + args.size_in[2]),
                    ]
                    input_img_tensor = torch.from_numpy(
                        input_patch.astype(float)
                    ).float()

                    tmp_out = model(Variable(input_img_tensor.cuda()).unsqueeze(0))
                    label = tmp_out[:, args.OutputCh, :, :, :]

                    prob = softmax(label)

                    out_flat_tensor = prob.cpu().data.float()

                    out_tensor = out_flat_tensor.view(
                        args.size_out[0],
                        args.size_out[1],
                        args.size_out[2],
                        # args.nclass,
                    )
                    out_nda = out_tensor.numpy()
                    output_img[
                        0,
                        za : (za + args.size_out[0]),
                        ya : (ya + args.size_out[1]),
                        xa : (xa + args.size_out[2]),
                    ] = out_nda[:, :, :]
                    # args.OutputCh]

    return torch.from_numpy(output_img).cuda()


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])
