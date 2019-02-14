import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channel, n_classes, batchnorm_flag=True):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D, self).__init__()

        self.ec1 = self.encoder(self.in_channel, 32, batchnorm=batchnorm_flag)
        self.ec2 = self.encoder(64, 64, batchnorm=batchnorm_flag)
        self.ec3 = self.encoder(128, 128, batchnorm=batchnorm_flag)
        self.ec4 = self.encoder(256, 256, batchnorm=batchnorm_flag)

        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(2)

        self.up3 = nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)
        self.dc3 = self.decoder(256 + 512, 256, batchnorm=batchnorm_flag)
        self.up2 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)
        self.dc2 = self.decoder(128 + 256, 128, batchnorm=batchnorm_flag)
        self.up1 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)
        self.dc1 = self.decoder(64 + 128, 64, batchnorm=batchnorm_flag)

        self.dc0 = nn.Conv3d(64, n_classes, 1)
        self.softmax = F.log_softmax

        self.numClass = n_classes

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels, affine=False),
                nn.ReLU(),
                nn.Conv3d(out_channels, 2*out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(2*out_channels, affine=False),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU(),
                nn.Conv3d(out_channels, 2*out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer


    def decoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels, affine=False),
                nn.ReLU(),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels, affine=False),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU(),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer

    def forward(self, x):

        down1 = self.ec1(x)
        x1 = self.pool1(down1)
        down2 = self.ec2(x1)
        x2 = self.pool2(down2)
        down3 = self.ec3(x2)
        x3 = self.pool3(down3)

        u3 = self.ec4(x3)

        d3 = torch.cat((self.up3(u3), F.pad(down3,(-4,-4,-4,-4,-4,-4))), 1)
        u2 = self.dc3(d3)
        d2 = torch.cat((self.up2(u2), F.pad(down2,(-16,-16,-16,-16,-16,-16))), 1)
        u1 = self.dc2(d2)
        d1 = torch.cat((self.up1(u1), F.pad(down1,(-40,-40,-40,-40,-40,-40))), 1)
        u0 = self.dc1(d1)
        out = self.dc0(u0)

        out = out.permute(0, 2, 3, 4, 1).contiguous() # move the class channel to the last dimension
        out = out.view(out.numel() // self.numClass, self.numClass)
        out = self.softmax(out, dim=1)

        return out
