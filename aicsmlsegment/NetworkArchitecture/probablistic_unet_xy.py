import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl

"""
code is modified from: https://github.com/stefanknegt/Probabilistic-Unet-Pytorch
"""
def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.Conv3d or type(m) == nn.ConvTranspose3d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv3d or type(m) == nn.ConvTranspose3d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

class UNet3D(nn.Module):
    def __init__(
        self, in_channel, n_classes, down_ratio, test_mode=True, batchnorm_flag=True
    ):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.test_mode = test_mode
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

        self.predict0 = nn.Conv3d(64, n_classes, 1)

        self.numClass = n_classes

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
                nn.Conv3d(
                    out_channels,
                    2 * out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
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
                nn.Conv3d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.ReLU(),
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

        return self.dc0(d0)

class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of num_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """
    def __init__(self, input_channels, num_filters, num_convs_per_block, initializers, padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            #To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += 1

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]
            
            if i != 0:
                layers.append(nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0, ceil_mode=True))
            
            layers.append(nn.Conv3d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(num_convs_per_block-1):
                layers.append(nn.Conv3d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output

class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, num_filters, num_convs_per_block, latent_dim, initializers, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.num_convs_per_block = num_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters, self.num_convs_per_block, initializers, posterior=self.posterior)
        self.conv_layer = nn.Conv3d(num_filters[-1], 2 * self.latent_dim, (1,1,1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):

        #If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)
        self.show_enc = encoding

        #We only want the mean of the resulting zxhxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)
        encoding = torch.mean(encoding, dim=4, keepdim=True)

        #Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)

        #We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:,:self.latent_dim]
        log_sigma = mu_log_sigma[:,self.latent_dim:]

        #This is a multivariate normal with diagonal covariance matrix sigma
        #https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        return dist

class Fcomb(nn.Module):
    """
    A function composed of num_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, num_convs_fcomb, initializers, use_tile=True):
        super(Fcomb, self).__init__()
        self.num_channels = num_output_channels #output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2,3,4]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.num_convs_fcomb = num_convs_fcomb 
        self.name = 'Fcomb'

        if self.use_tile:
            layers = []

            #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv3d(self.num_filters+self.latent_dim, self.num_filters, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(num_convs_fcomb-2):
                layers.append(nn.Conv3d(self.num_filters, self.num_filters, kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            self.last_layer = nn.Conv3d(self.num_filters, self.num_classes, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            # print(f'feature_map:{feature_map.shape}, z:{z.shape}')
            z = torch.unsqueeze(z,2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z,3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])
            z = torch.unsqueeze(z,4)
            z = self.tile(z, 4, feature_map.shape[self.spatial_axes[2]])

            #Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            output = self.layers(feature_map)
            return self.last_layer(output)


class ProbabilisticUnet(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(self, input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=6, num_convs_fcomb=4, beta=10.0):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.num_convs_per_block = 3
        self.num_convs_fcomb = num_convs_fcomb
        self.initializers = {'w':'he_normal', 'b':'normal'}
        self.beta = beta
        self.z_prior_sample = 0
        self.unet = UNet3D(self.input_channels, self.num_classes, down_ratio=2, batchnorm_flag=True)
        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.num_convs_per_block, self.latent_dim,  self.initializers)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.num_convs_per_block, self.latent_dim, self.initializers, posterior=True)
        self.fcomb = Fcomb(64, self.latent_dim, self.input_channels, self.num_classes, self.num_convs_fcomb, {'w':'orthogonal', 'b':'normal'}, use_tile=True)
        self.unet.apply(init_weights)

    def forward(self, patch, segm, training=True, use_prior_latent=False):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        prior_latent_space = self.prior.forward(patch)
        unet_features = self.unet.forward(patch)
        if training:
            posterior_latent_space = self.posterior.forward(patch, segm)
            if use_prior_latent:
                z_posterior = prior_latent_space.rsample()
            else:
                z_posterior = posterior_latent_space.rsample()
            output = self.fcomb.forward(unet_features, z_posterior)
            return output, prior_latent_space, posterior_latent_space
        else:
            z_posterior = prior_latent_space.rsample()
            output = self.fcomb.forward(unet_features, z_posterior)
            return output

    def kl_divergence(self, prior_latent_space, posterior_latent_space, analytic=True, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        """
        if analytic:
            #Need to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        else:
            if z_posterior is None:
                z_posterior = posterior_latent_space.rsample()
            log_posterior_prob = posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, criterion, output, segm, cmap, prior_latent_space, posterior_latent_space, analytic_kl=True):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """
        z_posterior = posterior_latent_space.rsample()
        
        kl = torch.mean(self.kl_divergence(prior_latent_space, posterior_latent_space, analytic=analytic_kl, z_posterior=z_posterior))

        #Here we use the posterior sample sampled above
        reconstruction_loss = criterion(output, segm, cmap)
        reconstruction_loss = torch.sum(reconstruction_loss)
        mean_reconstruction_loss = torch.mean(reconstruction_loss)

        # print(f'mean_reconstruction_loss:{mean_reconstruction_loss}, kl:{kl}')

        return -(mean_reconstruction_loss + self.beta * kl)

    # def elbo(self, output, segm, prior_latent_space, posterior_latent_space, analytic_kl=True):
    #     """
    #     Calculate the evidence lower bound of the log-likelihood of P(Y|X)
    #     """
    #     z_posterior = posterior_latent_space.rsample()

    #     criterion = nn.CrossEntropyLoss()
        
    #     kl = torch.mean(self.kl_divergence(prior_latent_space, posterior_latent_space, analytic=analytic_kl, z_posterior=z_posterior))

    #     #Here we use the posterior sample sampled above
    #     reconstruction_loss = criterion(output, segm.squeeze(dim=1).long())
    #     reconstruction_loss = torch.sum(reconstruction_loss)
    #     mean_reconstruction_loss = torch.mean(reconstruction_loss)

    #     return -(reconstruction_loss + self.beta * kl)

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='7'
    model = ProbabilisticUnet(input_channels=1, num_classes=2, num_filters=[32,64,128,192], latent_dim=2, num_convs_fcomb=4, beta=10.0)
    model.cuda()
    x = torch.randn([1,1,40,256,256]).cuda()
    mask = torch.ones([1,1,40,256,256]).cuda()
    for i in range(1000):
        output, prior_latent_space, posterior_latent_space = model.forward(x, mask, training=True, use_prior_latent=False)
        elbo = model.elbo(output, mask, prior_latent_space, posterior_latent_space)
        reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior) + l2_regularisation(model.fcomb.layers)
        print(f'elbo:{elbo},reg_loss:{reg_loss}')
        loss = -elbo + 1e-5 * reg_loss
        print(f'loss:{loss}')