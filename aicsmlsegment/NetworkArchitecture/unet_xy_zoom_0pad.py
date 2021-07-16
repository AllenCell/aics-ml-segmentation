import torch
import torch.nn as nn


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channel,
        n_classes,
        down_ratio,
        test_mode=True,
        batchnorm_flag=True,
        dropout=0,
    ):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.test_mode = test_mode
        self.dropout = dropout
        super(UNet3D, self).__init__()

        k = down_ratio

        self.ec1 = self.encoder(
            self.in_channel, 32, batchnorm=batchnorm_flag, padding=(1, 1, 1)
        )  # in --> 64
        self.ec2 = self.encoder(
            64, 64, batchnorm=batchnorm_flag, padding=(1, 1, 1)
        )  # 64 --> 128
        self.ec3 = self.encoder(
            128, 128, batchnorm=batchnorm_flag, padding=(1, 1, 1)
        )  # 128 --> 256
        self.ec4 = self.encoder(
            256, 256, batchnorm=batchnorm_flag, padding=(1, 1, 1)
        )  # 256 -->512

        self.pool0 = nn.MaxPool3d((1, k, k))
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.pool3 = nn.MaxPool3d((1, 2, 2))

        self.up3 = nn.ConvTranspose3d(
            512,
            512,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=0,
            output_padding=0,
            bias=True,
        )
        self.up2 = nn.ConvTranspose3d(
            256,
            256,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=0,
            output_padding=0,
            bias=True,
        )
        self.up1 = nn.ConvTranspose3d(
            128,
            128,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=0,
            output_padding=0,
            bias=True,
        )
        self.up0 = nn.ConvTranspose3d(
            64,
            64,
            kernel_size=(1, k, k),
            stride=(1, k, k),
            padding=0,
            output_padding=0,
            bias=True,
        )

        self.dc3 = self.decoder(
            256 + 512, 256, batchnorm=batchnorm_flag, padding=(1, 1, 1)
        )
        self.dc2 = self.decoder(
            128 + 256, 128, batchnorm=batchnorm_flag, padding=(1, 1, 1)
        )
        self.dc1 = self.decoder(
            64 + 128, 64, batchnorm=batchnorm_flag, padding=(1, 1, 1)
        )
        self.dc0 = self.decoder(64, 64, batchnorm=batchnorm_flag, padding=(1, 1, 1))

        self.predict0 = nn.Conv3d(64, n_classes[0], 1)

        self.up1a = nn.ConvTranspose3d(
            128,
            n_classes[1],
            kernel_size=(1, 2 * k, 2 * k),
            stride=(1, 2 * k, 2 * k),
            padding=0,
            output_padding=0,
            bias=True,
        )
        self.up2a = nn.ConvTranspose3d(
            256,
            n_classes[2],
            kernel_size=(1, 4 * k, 4 * k),
            stride=(1, 4 * k, 4 * k),
            padding=0,
            output_padding=0,
            bias=True,
        )

        self.conv2a = nn.Conv3d(
            n_classes[2], n_classes[2], 3, stride=1, padding=(1, 1, 1), bias=True
        )
        self.conv1a = nn.Conv3d(
            n_classes[1], n_classes[1], 3, stride=1, padding=(1, 1, 1), bias=True
        )

        self.predict2a = nn.Conv3d(n_classes[2], n_classes[2], 1)
        self.predict1a = nn.Conv3d(n_classes[1], n_classes[1], 1)

        self.numClass = n_classes[0]
        self.numClass1 = n_classes[1]
        self.numClass2 = n_classes[2]

        # a property will be used when calling this model in model zoo
        self.final_activation = nn.Softmax(dim=1)

        self.k = k
        # self.numClass_combine = n_classes[3]

    def encoder(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=True,
        batchnorm=False,
        dropout=0,
    ):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.BatchNorm3d(out_channels, affine=False),
                nn.ReLU(),
                nn.Dropout3d(p=dropout),
                nn.Conv3d(
                    out_channels,
                    2 * out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.BatchNorm3d(2 * out_channels, affine=False),
                nn.ReLU(),
                nn.Dropout3d(p=dropout),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.ReLU(),
                nn.Dropout3d(p=dropout),
                nn.Conv3d(
                    out_channels,
                    2 * out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.ReLU(),
                nn.Dropout3d(p=dropout),
            )
        return layer

    def decoder(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=True,
        batchnorm=False,
        dropout=0,
    ):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.BatchNorm3d(out_channels, affine=False),
                nn.ReLU(),
                nn.Dropout3d(p=dropout),
                nn.Conv3d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.BatchNorm3d(out_channels, affine=False),
                nn.ReLU(),
                nn.Dropout3d(p=dropout),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.ReLU(),
                nn.Dropout3d(p=dropout),
                nn.Conv3d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.ReLU(),
                nn.Dropout3d(p=dropout),
            )
        return layer

    def forward(self, x):
        x0 = self.pool0(x)
        down1 = self.ec1(x0)
        x1 = self.pool1(down1)
        down2 = self.ec2(x1)

        x2 = self.pool2(down2)
        down3 = self.ec3(x2)
        x3 = self.pool3(down3)
        u3 = self.ec4(x3)

        d3 = torch.cat((self.up3(u3), down3), 1)
        u2 = self.dc3(d3)
        d2 = torch.cat((self.up2(u2), down2), 1)
        u1 = self.dc2(d2)
        d1 = torch.cat((self.up1(u1), down1), 1)
        u0 = self.dc1(d1)

        d0 = self.up0(u0)

        predict00 = self.predict0(self.dc0(d0))
        # predict00 = F.pad(predict00, (-30, -30, -30, -30, -5, -5))
        if self.test_mode:
            return [predict00]

        p1a = self.predict1a(self.conv1a(self.up1a(u1)))
        p2a = self.predict2a(self.conv2a(self.up2a(u2)))  # fix +5
        return [predict00, p1a, p2a]
