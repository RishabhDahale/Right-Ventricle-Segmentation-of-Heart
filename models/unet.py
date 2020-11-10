import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1)


class DownConv(nn.Module):
    def __init__(self, in_channel, out_channel, pooling=True, batchnorm=False):
        super(DownConv, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.pooling = pooling
        self.batchnorm = batchnorm
    
        self.conv1 = conv3x3(self.in_channel, self.out_channel)
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.conv2 = conv3x3(self.out_channel, self.out_channel)
        self.bn2 = nn.BatchNorm2d(self.out_channel)

        if self.pooling:
            self.poolLayer = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        if self.batchnorm:
            x = self.bn1(x)

        x = self.conv2(x)
        x = F.relu(x)
        if self.batchnorm:
            x = self.bn2(x)

        beforeDownsampling = x
        if self.pooling:
            x = self.poolLayer(x)
        return x, beforeDownsampling


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
                                mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(2 * self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        return x


class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, depth=5, batchnorm=False, start_filts=64, up_mode='transpose', merge_mode='concat'):
        """
        :param num_classes:
        :param in_channels:
        :param depth:
        :param start_filts:
        :param up_mode: should be one among 'transpose', 'upsample'
        :param merge_mode: should in one among 'concat', 'add'

        NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        """
        super(UNet, self).__init__()
        self.up_mode = up_mode
        self.merge_mode = merge_mode
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.batchnorm = batchnorm

        self.down_convs = []
        self.up_convs = []

        # encoder
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling, batchnorm=self.batchnorm)
            self.down_convs.append(down_conv)

        # decoder
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)
        x = F.sigmoid(x)
        return x


if __name__ == "__main__":
    """
    testing
    """
    model = UNet(3, depth=5, merge_mode='concat', batchnorm=True)
    # print(model)
    x = torch.autograd.Variable(torch.FloatTensor(np.random.random((1, 3, 128, 128))))
    out = model(x)
    loss = torch.sum(out)
    loss.backward()


