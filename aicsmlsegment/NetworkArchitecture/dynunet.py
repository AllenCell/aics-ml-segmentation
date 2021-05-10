from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from monai.networks.blocks.dynunet_block import (
    UnetBasicBlock,
    UnetOutBlock,
    UnetResBlock,
    UnetUpBlock,
)
from monai.networks.nets.dynunet import DynUNetSkipLayer


class DynUNet(nn.Module):
    """
    modified from https://docs.monai.io/en/latest/_modules/monai/networks/nets/dynunet.html#DynUNet  # noqa E501
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        upsample_kernel_size: Sequence[Union[Sequence[int], int]],
        norm_name: str = "instance",
        deep_supervision: bool = False,
        deep_supr_num: int = 1,
        res_block: bool = False,
    ):
        super(DynUNet, self).__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.upsample_kernel_size = upsample_kernel_size
        self.norm_name = norm_name
        self.conv_block = UnetResBlock if res_block else UnetBasicBlock
        self.filters = [
            2 ** (5 + i) for i in range(len(strides))
        ]  # REMOVE FILTER LIMIT
        self.input_block = self.get_input_block()
        self.downsamples = self.get_downsamples()
        self.bottleneck = self.get_bottleneck()
        self.upsamples = self.get_upsamples()
        self.output_block = self.get_output_block(0)
        self.deep_supervision = deep_supervision
        self.deep_supervision_heads = self.get_deep_supervision_heads()
        self.deep_supr_num = deep_supr_num
        self.apply(self.initialize_weights)
        self.check_kernel_stride()
        self.check_deep_supr_num()

        # initialize the typed list of supervision head outputs so that Torchscript
        # can recognize what's going on
        self.heads: List[torch.Tensor] = [torch.rand(1)] * (
            len(self.deep_supervision_heads) + 1
        )

        def create_skips(index, downsamples, upsamples, superheads, bottleneck):

            if len(downsamples) != len(upsamples):
                raise AssertionError(f"{len(downsamples)} != {len(upsamples)}")
            if (len(downsamples) - len(superheads)) not in (1, 0):
                raise AssertionError(f"{len(downsamples)}-(0,1) != {len(superheads)}")

            if (
                len(downsamples) == 0
            ):  # bottom of the network, pass the bottleneck block
                return bottleneck
            if index == 0:  # don't associate a supervision head with self.input_block
                current_head, rest_heads = nn.Identity(), superheads
            elif (
                not self.deep_supervision
            ):  # bypass supervision heads by passing nn.Identity in place of a real one
                current_head, rest_heads = nn.Identity(), superheads[1:]
            else:
                current_head, rest_heads = superheads[0], superheads[1:]

            # create the next layer down, this will stop at the bottleneck layer
            next_layer = create_skips(
                1 + index, downsamples[1:], upsamples[1:], rest_heads, bottleneck
            )

            return DynUNetSkipLayer(
                index,
                self.heads,
                downsamples[0],
                upsamples[0],
                current_head,
                next_layer,
            )

        self.skip_layers = create_skips(
            0,
            [self.input_block] + list(self.downsamples),
            self.upsamples[::-1],
            self.deep_supervision_heads,
            self.bottleneck,
        )

    def check_kernel_stride(self):
        kernels, strides = self.kernel_size, self.strides
        error_msg = (
            "length of kernel_size and strides should be the same, and no less than 3."
        )
        if not (len(kernels) == len(strides) and len(kernels) >= 3):
            raise AssertionError(error_msg)

        for idx in range(len(kernels)):
            kernel, stride = kernels[idx], strides[idx]
            if not isinstance(kernel, int):
                error_msg = (
                    f"length of kernel_size in block {idx} should "
                    "be the same as spatial_dims."
                )
                if len(kernel) != self.spatial_dims:
                    raise AssertionError(error_msg)
            if not isinstance(stride, int):
                error_msg = (
                    f"length of stride in block {idx} should be "
                    "the same as spatial_dims."
                )
                if len(stride) != self.spatial_dims:
                    raise AssertionError(error_msg)

    def check_deep_supr_num(self):
        deep_supr_num, strides = self.deep_supr_num, self.strides
        num_up_layers = len(strides) - 1
        if deep_supr_num >= num_up_layers:
            raise AssertionError(
                "deep_supr_num should be less than the number of up sample layers."
            )
        if deep_supr_num < 1:
            raise AssertionError("deep_supr_num should be larger than 0.")

    def forward(self, x):
        out = self.skip_layers(x)
        out = self.output_block(out)
        if self.training and self.deep_supervision:
            out_all = [out]
            feature_maps = self.heads[1 : self.deep_supr_num + 1]
            for feature_map in feature_maps:
                out_all.append(interpolate(feature_map, out.shape[2:]))
            return torch.stack(out_all, dim=1)
        return out

    def get_input_block(self):
        return self.conv_block(
            self.spatial_dims,
            self.in_channels,
            self.filters[0],
            self.kernel_size[0],
            self.strides[0],
            self.norm_name,
        )

    def get_bottleneck(self):
        return self.conv_block(
            self.spatial_dims,
            self.filters[-2],
            self.filters[-1],
            self.kernel_size[-1],
            self.strides[-1],
            self.norm_name,
        )

    def get_output_block(self, idx: int):
        return UnetOutBlock(
            self.spatial_dims,
            self.filters[idx],
            self.out_channels,
        )

    def get_downsamples(self):
        inp, out = self.filters[:-2], self.filters[1:-1]
        strides, kernel_size = self.strides[1:-1], self.kernel_size[1:-1]
        return self.get_module_list(inp, out, kernel_size, strides, self.conv_block)

    def get_upsamples(self):
        inp, out = self.filters[1:][::-1], self.filters[:-1][::-1]
        strides, kernel_size = self.strides[1:][::-1], self.kernel_size[1:][::-1]
        upsample_kernel_size = self.upsample_kernel_size[::-1]
        return self.get_module_list(
            inp, out, kernel_size, strides, UnetUpBlock, upsample_kernel_size
        )

    def get_module_list(
        self,
        in_channels: List[int],
        out_channels: List[int],
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        conv_block: nn.Module,
        upsample_kernel_size: Optional[Sequence[Union[Sequence[int], int]]] = None,
    ):
        layers = []
        if upsample_kernel_size is not None:
            for in_c, out_c, kernel, stride, up_kernel in zip(
                in_channels, out_channels, kernel_size, strides, upsample_kernel_size
            ):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "upsample_kernel_size": up_kernel,
                }
                layer = conv_block(**params)
                layers.append(layer)
        else:
            for in_c, out_c, kernel, stride in zip(
                in_channels, out_channels, kernel_size, strides
            ):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                }
                layer = conv_block(**params)
                layers.append(layer)
        return nn.ModuleList(layers)

    def get_deep_supervision_heads(self):
        return nn.ModuleList(
            [self.get_output_block(i + 1) for i in range(len(self.upsamples) - 1)]
        )

    @staticmethod
    def initialize_weights(module):
        name = module.__class__.__name__.lower()
        if "conv3d" in name or "conv2d" in name:
            nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif "norm" in name:
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.zeros_(module.bias)
