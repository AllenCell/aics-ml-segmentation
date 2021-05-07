import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):

    """
    3D adaptation of https://arxiv.org/ftp/arxiv/papers/2004/2004.03466.pdf
    This network is similar to a standard UNet, but it exchanges the 2 stacked
    convolutions followed by pooling for encoding with concatenation of a series of
    dilated convolutions. The Decoder unit is similar, except pooling is replaced by
    upsampling and concatenation with the encoder map.
    """

    def __init__(self, in_channel, n_classes, loss, test_mode):
        self.in_channel = in_channel
        self.n_classes = n_classes[0]
        self.loss = loss
        super(UNet3D, self).__init__()

        self.input = self.conv_relu(self.in_channel, 32, 1, 1)
        self.one_by_oneconv = self.oneconv(32, self.n_classes)
        self.pool = nn.MaxPool3d((1, 2, 2))

        # self.conv1 = self.conv_relu(
        #     32, 32, stride=(1, 2, 2), kernel=(1, 2, 2), norm=False
        # )
        # self.conv2 = self.conv_relu(
        #     64, 64, stride=(1, 2, 2), kernel=(1, 2, 2), norm=False
        # )
        # self.conv3 = self.conv_relu(
        #     128, 128, stride=(1, 2, 2), kernel=(1, 2, 2), norm=False
        # )
        # self.conv4 = self.conv_relu(
        #     256, 256, stride=(1, 2, 2), kernel=(1, 2, 2), norm=False
        # )

        self.upsample_64 = self.upsample(64)
        self.upsample_128 = self.upsample(128)
        self.upsample_256 = self.upsample(256)
        self.upsample_512 = self.upsample(512)

        self.down_32_1 = self.conv_relu(32, 32, padding=1, dilation=1)
        self.down_32_3 = self.conv_relu(32, 16, padding=3, dilation=3)
        self.down_32_6 = self.conv_relu(16, 8, padding=6, dilation=6)
        self.down_32_9 = self.conv_relu(8, 4, padding=9, dilation=9)
        self.down_32_12 = self.conv_relu(4, 4, padding=12, dilation=12)
        self.down_block1_fns = [
            # self.conv1,
            self.pool,
            self.down_32_1,
            self.down_32_3,
            self.down_32_6,
            self.down_32_9,
            self.down_32_12,
        ]

        self.down_64_1 = self.conv_relu(64, 64, padding=1, dilation=1)
        self.down_64_3 = self.conv_relu(64, 32, padding=3, dilation=3)
        self.down_64_6 = self.conv_relu(32, 16, padding=6, dilation=6)
        self.down_64_9 = self.conv_relu(16, 8, padding=9, dilation=9)
        self.down_64_12 = self.conv_relu(8, 8, padding=12, dilation=12)
        self.down_block2_fns = [
            # self.conv2,
            self.pool,
            self.down_64_1,
            self.down_64_3,
            self.down_64_6,
            self.down_64_9,
            self.down_64_12,
        ]

        self.down_128_1 = self.conv_relu(128, 128, padding=1, dilation=1)
        self.down_128_3 = self.conv_relu(128, 64, padding=3, dilation=3)
        self.down_128_6 = self.conv_relu(64, 32, padding=6, dilation=6)
        self.down_128_9 = self.conv_relu(32, 16, padding=9, dilation=9)
        self.down_128_12 = self.conv_relu(16, 16, padding=12, dilation=12)
        self.down_block3_fns = [
            # self.conv3,
            self.pool,
            self.down_128_1,
            self.down_128_3,
            self.down_128_6,
            self.down_128_9,
            self.down_128_12,
        ]

        self.down_256_1 = self.conv_relu(256, 256, padding=1, dilation=1)
        self.down_256_3 = self.conv_relu(256, 128, padding=3, dilation=3)
        self.down_256_6 = self.conv_relu(128, 64, padding=6, dilation=6)
        self.down_256_9 = self.conv_relu(64, 32, padding=9, dilation=9)
        self.down_256_12 = self.conv_relu(32, 32, padding=12, dilation=12)
        self.down_block4_fns = [
            # self.conv4,
            self.pool,
            self.down_256_1,
            self.down_256_3,
            self.down_256_6,
            self.down_256_9,
            self.down_256_12,
        ]

        self.up_768_1 = self.conv_relu(256 + 512, 128, padding=1, dilation=1)
        self.up_512_3 = self.conv_relu(128, 64, padding=3, dilation=3)
        self.up_512_6 = self.conv_relu(64, 32, padding=6, dilation=6)
        self.up_512_9 = self.conv_relu(32, 16, padding=9, dilation=9)
        self.up_512_12 = self.conv_relu(16, 16, padding=12, dilation=12)
        self.up_block1_fns = [
            self.upsample_512,
            self.up_768_1,
            self.up_512_3,
            self.up_512_6,
            self.up_512_9,
            self.up_512_12,
        ]

        self.up_384_1 = self.conv_relu(128 + 256, 64, padding=1, dilation=1)
        self.up_256_3 = self.conv_relu(64, 32, padding=3, dilation=3)
        self.up_256_6 = self.conv_relu(32, 16, padding=6, dilation=6)
        self.up_256_9 = self.conv_relu(16, 8, padding=9, dilation=9)
        self.up_256_12 = self.conv_relu(8, 8, padding=12, dilation=12)
        self.up_block2_fns = [
            self.upsample_256,
            self.up_384_1,
            self.up_256_3,
            self.up_256_6,
            self.up_256_9,
            self.up_256_12,
        ]

        self.up_192_1 = self.conv_relu(64 + 128, 32, padding=1, dilation=1)
        self.up_128_3 = self.conv_relu(32, 16, padding=3, dilation=3)
        self.up_128_6 = self.conv_relu(16, 8, padding=6, dilation=6)
        self.up_128_9 = self.conv_relu(8, 4, padding=9, dilation=9)
        self.up_128_12 = self.conv_relu(4, 4, padding=12, dilation=12)
        self.up_block3_fns = [
            self.upsample_128,
            self.up_192_1,
            self.up_128_3,
            self.up_128_6,
            self.up_128_9,
            self.up_128_12,
        ]

        self.up_64_1 = self.conv_relu(64, 16, padding=1, dilation=1)
        self.up_64_3 = self.conv_relu(16, 8, padding=3, dilation=3)
        self.up_64_6 = self.conv_relu(8, 4, padding=6, dilation=6)
        self.up_64_9 = self.conv_relu(4, 2, padding=9, dilation=9)
        self.up_64_12 = self.conv_relu(2, 2, padding=12, dilation=12)
        self.up_block4_fns = [
            self.upsample_64,
            self.up_64_1,
            self.up_64_3,
            self.up_64_6,
            self.up_64_9,
            self.up_64_12,
        ]

    def upsample(self, in_channels):
        return nn.ConvTranspose3d(
            in_channels,
            in_channels,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=0,
            output_padding=0,
            bias=True,
        )

    def oneconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def conv_relu(
        self,
        in_channels,
        out_channels,
        padding=0,
        dilation=1,
        stride=1,
        kernel=3,
        norm=True,
    ):
        if norm:
            return nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride=stride,
                    padding=padding,
                    bias=True,
                    dilation=dilation,
                ),
                nn.InstanceNorm3d(out_channels, affine=False),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride=stride,
                    padding=padding,
                    bias=True,
                    dilation=dilation,
                ),
                # nn.ReLU(),
            )

    def down_block(self, x, fns):
        """
        x: input tensor
        fns: list of functions to be sequentially applied to x
        """
        outputs = [x]
        # output of previous step as input to next step
        for fn in fns:
            outputs.append(fn(outputs[-1]))
        # concat all output except original input
        return torch.cat(outputs[-5:], dim=1)

    def up_block(self, x, cat, fns):
        upsample = fns[0](x)  # don't apply upsample later in fns loop
        if cat is not None:
            upsample = torch.cat((upsample, cat), dim=1)
        outputs = [upsample]
        for fn in fns[1:]:
            outputs.append(fn(outputs[-1]))
        return torch.cat(outputs[-5:], dim=1)

    def forward(self, x):
        x = self.input(x)  # 1 channel -> 32 channels
        down1 = self.down_block(x, self.down_block1_fns)  # 32 ch -> 64 ch, pool xy
        down2 = self.down_block(down1, self.down_block2_fns)  # 64->128ch, pool xy
        down3 = self.down_block(down2, self.down_block3_fns)  # 128->56ch, pool xy
        down4 = self.down_block(down3, self.down_block4_fns)  # 256->512ch, pool xy

        up1 = self.up_block(down4, down3, self.up_block1_fns)  # 512->256, upsample xy
        up2 = self.up_block(up1, down2, self.up_block2_fns)  # 256->128, upsample xy
        up3 = self.up_block(up2, down1, self.up_block3_fns)  # 128->64, upsample xy
        up4 = self.up_block(up3, None, self.up_block4_fns)  # 64->32 upsample xy
        out = self.one_by_oneconv(up4)  # 32->2ch

        if self.loss == "Aux":
            return [out]
        return out
