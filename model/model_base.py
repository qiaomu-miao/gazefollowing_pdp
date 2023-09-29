import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckConvLSTM(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckConvLSTM, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.bn_ds = nn.BatchNorm2d(planes * self.expansion)

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # RW edit: handles batch_size==1
        if out.shape[0] > 1:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # RW edit: handles batch_size==1
        if out.shape[0] > 1:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # RW edit: handles batch_size==1
        if out.shape[0] > 1:
            out = self.bn3(out)

        if self.downsample is not None:
            # RW edit: handles batch_size==1
            if out.shape[0] > 1:
                residual = self.downsample(x)
                residual = self.bn_ds(residual)
            else:
                residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out