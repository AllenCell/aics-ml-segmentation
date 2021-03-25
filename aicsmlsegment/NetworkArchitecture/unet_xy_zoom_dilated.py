import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    def __init__(
        self, in_channel, n_classes, down_ratio, batchnorm_flag=True, test_mode=False
    ):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.test_mode = test_mode
        super(UNet3D, self).__init__()

        k = down_ratio

        self.ec1 = self.encoder(
            self.in_channel,
            8,
            batchnorm=batchnorm_flag,
            padding=(1, k, k),
            dilation=(1, k, k),
        )  # in --> 16
        self.ec2 = self.encoder(
            16, 16, batchnorm=batchnorm_flag, padding=(1, 2, 2), dilation=(1, 2, 2)
        )  # 16 --> 32
        self.ec3 = self.encoder(
            32, 32, batchnorm=batchnorm_flag, padding=(1, 2, 2), dilation=(1, 2, 2)
        )  # 32 --> 64
        self.ec4 = self.encoder(
            64, 64, batchnorm=batchnorm_flag, padding=(1, 2, 2), dilation=(1, 2, 2)
        )  # 64 -->128

        self.dc3 = self.decoder(
            64 + 128,
            64,
            batchnorm=batchnorm_flag,
            padding=(1, 2, 2),
            dilation=(1, 2, 2),
        )
        self.dc2 = self.decoder(
            32 + 64, 32, batchnorm=batchnorm_flag, padding=(1, 2, 2), dilation=(1, 2, 2)
        )
        self.dc1 = self.decoder(
            16 + 32, 16, batchnorm=batchnorm_flag, padding=(1, 2, 2), dilation=(1, 2, 2)
        )
        self.dc0 = self.decoder(
            16, 16, batchnorm=batchnorm_flag, padding=(1, k, k), dilation=(1, k, k)
        )

        self.predict0 = nn.Conv3d(16, n_classes[0], 1)

        self.conv2a = nn.Conv3d(
            64, n_classes[2], 3, stride=1, padding=(1, 1, 1), bias=True
        )
        self.conv1a = nn.Conv3d(
            32, n_classes[1], 3, stride=1, padding=(1, 1, 1), bias=True
        )

        self.predict2a = nn.Conv3d(n_classes[2], n_classes[2], 1)
        self.predict1a = nn.Conv3d(n_classes[1], n_classes[1], 1)

        self.softmax = F.log_softmax  # nn.LogSoftmax(1)

        self.final_activation = nn.Softmax(dim=1)

        self.numClass = n_classes[0]
        self.numClass1 = n_classes[1]
        self.numClass2 = n_classes[2]

        self.k = k

    def encoder(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=True,
        batchnorm=False,
        dilation=1,
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
                    dilation=dilation,
                ),
                nn.BatchNorm3d(out_channels, affine=False),
                nn.ReLU(),
                nn.Conv3d(
                    out_channels,
                    2 * out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    dilation=dilation,
                ),
                nn.BatchNorm3d(2 * out_channels, affine=False),
                nn.ReLU(),
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
                    dilation=dilation,
                ),
                nn.ReLU(),
                nn.Conv3d(
                    out_channels,
                    2 * out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    dilation=dilation,
                ),
                nn.ReLU(),
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
        dilation=1,
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
                    dilation=dilation,
                ),
                nn.BatchNorm3d(out_channels, affine=False),
                nn.ReLU(),
                nn.Conv3d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    dilation=dilation,
                ),
                nn.BatchNorm3d(out_channels, affine=False),
                nn.ReLU(),
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
                    dilation=dilation,
                ),
                nn.ReLU(),
                nn.Conv3d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    dilation=dilation,
                ),
                nn.ReLU(),
            )
        return layer

    def forward(self, x):
        down1 = self.ec1(x)
        down2 = self.ec2(down1)
        down3 = self.ec3(down2)
        u3 = self.ec4(down3)

        d3 = torch.cat((u3, down3), 1)
        u2 = self.dc3(d3)
        d2 = torch.cat((u2, down2), 1)
        u1 = self.dc2(d2)
        d1 = torch.cat((u1, down1), 1)
        u0 = self.dc1(d1)

        predict00 = self.predict0(self.dc0(u0))
        p0_final = predict00.permute(
            0, 2, 3, 4, 1
        ).contiguous()  # move the class channel to the last dimension
        p0_final = p0_final.view(p0_final.numel() // self.numClass, self.numClass)
        p0_final = self.softmax(p0_final, dim=1)
        if self.test_mode:
            return [p0_final]

        p1a = self.predict1a(self.conv1a(u1))
        p1_final = p1a.permute(
            0, 2, 3, 4, 1
        ).contiguous()  # move the class channel to the last dimension
        p1_final = p1_final.view(p1_final.numel() // self.numClass1, self.numClass1)
        p1_final = self.softmax(p1_final, dim=1)

        p2a = self.predict2a(self.conv2a(u2))
        p2_final = p2a.permute(
            0, 2, 3, 4, 1
        ).contiguous()  # move the class channel to the last dimension
        p2_final = p2_final.view(p2_final.numel() // self.numClass2, self.numClass2)
        p2_final = self.softmax(p2_final, dim=1)

        """
        p_combine0 = self.predict_final(self.conv_final(torch.cat((p0, p1a, p2a), 1)))  # BCZYX
        p_combine = p_combine0.permute(0, 2, 3, 4, 1).contiguous() # move the class channel to the last dimension
        p_combine = p_combine.view(p_combine.numel() // self.numClass_combine, self.numClass_combine)
        p_combine = self.softmax(p_combine)
        """
        return [p0_final, p1_final, p2_final]
