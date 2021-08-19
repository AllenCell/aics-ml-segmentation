import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as f

class BasicBlock(nn.Module):
    def __init__(self, channel):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(channel)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class DownsampleBlock(nn.Module):
    def __init__(self, channel_in, channel):
        super(DownsampleBlock, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, channel, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(channel)
        self.downsample = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += torch.cat((residual, residual), 1)
        out = self.relu(out)

        return out

class ResNet3d_18(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(ResNet3d_18, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=True)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
        self.layer10 = BasicBlock(64)
        self.layer11 = BasicBlock(64)
        self.layer20 = DownsampleBlock(64, 128)
        self.layer21 = BasicBlock(128)
        self.layer30 = DownsampleBlock(128, 256)
        self.layer31 = BasicBlock(256)
        self.layer40 = DownsampleBlock(256, 512)
        self.layer41 = BasicBlock(512)

        # self.avgpool = nn.AvgPool3d(
        #     (math.ceil(clip_length // 8), math.ceil(crop_shape[1] / 32), math.ceil(crop_shape[0] / 32)))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.layers = []

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer20(x)
        x = self.layer21(x)
        x = self.layer30(x)
        x = self.layer31(x)

        x = self.layer40(x)
        x = self.layer41(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    x = torch.randn(1,3,10,150,150)
    model = ResNet3d_18(in_channel=3, num_classes=2)
    y = model(x)
    print(f'y:{y.shape}')