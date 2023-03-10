"""
Reference: https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
Implementation of Wide Resnet as NAS backbone architecture
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    first applies batch norm and relu before applying convolution
    we can change the order of operations if needed
    """

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv1d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet1d(nn.Module):
    """
    wide resnet
    """

    def __init__(
        self,
        depth,
        num_classes,
        input_shape,
        output_shape,
        widen_factor=1,
        dropRate=0.0,
        in_channels=3,
    ):
        super(WideResNet1d, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv1d(
            in_channels, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm1d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def pretty_x(self, x):
        if self.input_shape[0] != 1:
            # The time dimension is nonempty, which requires a transpose
            x = x[:, :, :, 0, 0]
            x = x.transpose(2, 1)
        elif self.input_shape[2] != 1:
            # A spatial dimension is nonempty
            x = x[:, 0, :, :, 0]
        elif self.input_shape[3] != 1:
            # A spatial dimension is nonempty
            x = x[:, 0, :, 0, :]
        else:
            # All spatial and time dimensions are empty,
            # use channels as the time dimension
            x = x[:, :, :, 0, 0]
        return x

    def forward_partial(self, x):
        x = self.pretty_x(x)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return out

    def get_lout(self):
        x = torch.randr(1, *self.input_shape)
        with torch.no_grad():
            out = self.forward_partial(x)
            return out.shape[2]

    def forward(self, x):
        out = self.forward_partial(x)

        out = nn.AdaptiveAvgPool1d(1)(out)

        out = out.view(out.size(0), -1)
        return self.fc(out)
