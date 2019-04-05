import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channel, n_classes, batchnorm_flag=True):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D, self).__init__()

        self.ec1 = self.encoder(self.in_channel, 32, batchnorm=batchnorm_flag) # in --> 64
        self.ec2 = self.encoder(64, 64, batchnorm=batchnorm_flag) # 64 --> 128
        self.ec3 = self.encoder(128, 128, batchnorm=batchnorm_flag) # 128 --> 256
        self.ec4 = self.encoder(256, 256, batchnorm=batchnorm_flag) # 256 -->512

        self.pool1 = nn.MaxPool3d((1,2,2))
        self.pool2 = nn.MaxPool3d((1,2,2))
        self.pool3 = nn.MaxPool3d((1,2,2))

        self.up3 = nn.ConvTranspose3d(512, 512, kernel_size=(1,2,2), stride=(1,2,2), padding=0, output_padding=0, bias=True)
        self.up2 = nn.ConvTranspose3d(256, 256, kernel_size=(1,2,2), stride=(1,2,2), padding=0, output_padding=0, bias=True)
        self.up1 = nn.ConvTranspose3d(128, 128, kernel_size=(1,2,2), stride=(1,2,2), padding=0, output_padding=0, bias=True)

        self.dc3 = self.decoder(256 + 512, 256, batchnorm=batchnorm_flag)
        self.dc2 = self.decoder(128 + 256, 128, batchnorm=batchnorm_flag)
        self.dc1 = self.decoder(64 + 128, 64, batchnorm=batchnorm_flag)
        
        self.dc0 = nn.Conv3d(64, n_classes[0], 1)

        self.up2a = nn.ConvTranspose3d(256, n_classes[2], kernel_size=(1,8,8), stride=(1,4,4), padding=0, output_padding=0, bias=True)
        self.up1a = nn.ConvTranspose3d(128, n_classes[1], kernel_size=(1,4,4), stride=(1,2,2), padding=0, output_padding=0, bias=True)
      
        self.conv2a = nn.Conv3d(n_classes[2], n_classes[2], 3, stride=1, padding=0, bias=True)
        self.conv1a = nn.Conv3d(n_classes[1], n_classes[1], 3, stride=1, padding=0, bias=True)

        self.predict2a = nn.Conv3d(n_classes[2], n_classes[2], 1)
        self.predict1a = nn.Conv3d(n_classes[1], n_classes[1], 1)

        #self.conv_final = nn.Conv3d(n_classes[0]+n_classes[1]+n_classes[2], n_classes[0]+n_classes[1]+n_classes[2], 3, stride=1, padding=1, bias=True)
        #self.predict_final = nn.Conv3d(n_classes[0]+n_classes[1]+n_classes[2], n_classes[3], 1)

        self.softmax = F.log_softmax  # nn.LogSoftmax(1)

        self.final_activation = nn.Softmax(dim=1)

        self.numClass = n_classes[0]
        self.numClass1 = n_classes[1]
        self.numClass2 = n_classes[2]
        #self.numClass_combine = n_classes[3]

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels, affine=False),
                nn.ReLU(),
                nn.Conv3d(out_channels, 2*out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(2*out_channels, affine=False),
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
                nn.BatchNorm3d(out_channels, affine=False),
                nn.ReLU(),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels, affine=False),
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

        d3 = torch.cat((self.up3(u3), F.pad(down3,(-4,-4,-4,-4,-2,-2))), 1)
        u2 = self.dc3(d3)

        d2 = torch.cat((self.up2(u2), F.pad(down2,(-16,-16,-16,-16,-6,-6))), 1)
        u1 = self.dc2(d2)

        d1 = torch.cat((self.up1(u1), F.pad(down1,(-40,-40,-40,-40,-10,-10))), 1)
        u0 = self.dc1(d1)

        p0 = self.dc0(u0)

        p1a = F.pad(self.predict1a(self.conv1a(self.up1a(u1))),(-2,-2,-2,-2, -1, -1))
        p2a = F.pad(self.predict2a(self.conv2a(self.up2a(u2))),(-7,-7,-7,-7,-3,-3))
     
        p0_final = p0.permute(0, 2, 3, 4, 1).contiguous() # move the class channel to the last dimension
        p0_final = p0_final.view(p0_final.numel() // self.numClass, self.numClass)
        p0_final = self.softmax(p0_final, dim=1)

        p1_final = p1a.permute(0, 2, 3, 4, 1).contiguous() # move the class channel to the last dimension
        p1_final = p1_final.view(p1_final.numel() // self.numClass1, self.numClass1)
        p1_final = self.softmax(p1_final, dim=1)

        p2_final = p2a.permute(0, 2, 3, 4, 1).contiguous() # move the class channel to the last dimension
        p2_final = p2_final.view(p2_final.numel() // self.numClass2, self.numClass2)
        p2_final = self.softmax(p2_final, dim=1)

        '''
        p_combine0 = self.predict_final(self.conv_final(torch.cat((p0, p1a, p2a), 1)))  # BCZYX
        p_combine = p_combine0.permute(0, 2, 3, 4, 1).contiguous() # move the class channel to the last dimension
        p_combine = p_combine.view(p_combine.numel() // self.numClass_combine, self.numClass_combine)
        p_combine = self.softmax(p_combine)
        '''

        return [p0_final, p1_final, p2_final]
