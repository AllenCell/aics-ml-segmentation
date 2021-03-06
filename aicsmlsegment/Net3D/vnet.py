from monai.networks.nets.vnet import (
    DownTransition,
    UpTransition,
    OutputTransition,
    InputTransition,
)
from typing import Dict, Tuple, Union

import torch.nn as nn


class VNet(nn.Module):
    """
    V-Net based on `Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
    <https://arxiv.org/pdf/1606.04797.pdf>`_.
    Adapted from `the official Caffe implementation
    <https://github.com/faustomilletari/VNet>`_. and `another pytorch implementation
    <https://github.com/mattmacy/vnet.pytorch/blob/master/vnet.py>`_.
    The model supports 2D or 3D inputs.

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        in_channels: number of input channels for the network. Defaults to 1.
            The value should meet the condition that ``16 % in_channels == 0``.
        out_channels: number of output channels for the network. Defaults to 1.
        act: activation type in the network. Defaults to ``("elu", {"inplace": True})``.
        dropout_prob: dropout ratio. Defaults to 0.5. Defaults to 3.
        dropout_dim: determine the dimensions of dropout. Defaults to 3.

            - ``dropout_dim = 1``, randomly zeroes some of the elements for each channel.
            - ``dropout_dim = 2``, Randomly zeroes out entire channels (a channel is a 2D feature map).
            - ``dropout_dim = 3``, Randomly zeroes out entire channels (a channel is a 3D feature map).
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        act: Union[Tuple[str, Dict], str] = ("elu", {"inplace": True}),
        dropout_prob: float = 0.5,
        dropout_dim: int = 3,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise AssertionError("spatial_dims can only be 2 or 3.")

        self.in_tr = InputTransition(spatial_dims, in_channels, 16, act)
        self.down_tr32 = DownTransition(spatial_dims, 16, 1, act)
        self.down_tr64 = DownTransition(spatial_dims, 32, 2, act)
        self.down_tr128 = DownTransition(
            spatial_dims, 64, 3, act, dropout_prob=dropout_prob
        )
        self.down_tr256 = DownTransition(
            spatial_dims, 128, 2, act, dropout_prob=dropout_prob
        )
        self.down_tr512 = DownTransition(
            spatial_dims, 256, 2, act, dropout_prob=dropout_prob
        )
        self.up_tr512 = UpTransition(
            spatial_dims, 512, 512, 2, act, dropout_prob=dropout_prob
        )
        self.up_tr256 = UpTransition(
            spatial_dims, 256, 256, 2, act, dropout_prob=dropout_prob
        )
        self.up_tr128 = UpTransition(
            spatial_dims, 256, 128, 2, act, dropout_prob=dropout_prob
        )
        self.up_tr64 = UpTransition(spatial_dims, 128, 64, 1, act)
        self.up_tr32 = UpTransition(spatial_dims, 64, 32, 1, act)
        self.out_tr = OutputTransition(spatial_dims, 32, out_channels, act)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out512 = self.down_tr512(out256)
        x = self.up_tr512(out512, out256)
        x = self.up_tr256(out256, out128)
        x = self.up_tr128(x, out64)
        x = self.up_tr64(x, out32)
        x = self.up_tr32(x, out16)
        x = self.out_tr(x)
        return x
