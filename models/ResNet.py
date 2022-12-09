# torch 版本 ---> 1.12.1+cu116
import torch
from torch import nn
from typing import List, Optional


class ShortcutProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """
            快捷连接
            :param in_channels: 输入通道数
            :param out_channels: 输出通道数
            :param stride: 卷积步长（the same stride on the shortcut connection, to match the feature-map size.）
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, stride), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """
            两层卷积的残差块
            :param in_channels: 输入通道数
            :param out_channels: 输出通道数
            :param stride: 卷积步长
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        temp = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        shortcut = self.shortcut(temp)
        return self.act2(x + shortcut)


class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int, stride: int):
        """
            三层卷积的残差块
            :param in_channels: 输入通道数
            :param bottleneck_channels: 3X3卷积的通道数
            :param out_channels: 输出通道数
            :param stride: 3X3卷积的步长
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=(3, 3), stride=(stride, stride),
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()
        self.act3 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        temp = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        shortcut = self.shortcut(temp)
        return self.act3(x + shortcut)


class ResNetBase(nn.Module):
    def __init__(self, n_blocks: List[int], n_channels: List[int], n_bottlenecks: Optional[List[int]] = None,
                 num_classes: int = 1000):
        """
            ResNet主体网络
            :param n_blocks: 每个特征图尺寸的残差块数量列表
            :param n_channels: 每个特征图尺寸的通道数列表
            :param n_bottlenecks: 瓶颈结构通道数列表，如果为None，则不使用瓶颈结构，只使用残差块。
            :param num_classes: 分类数，默认为1000
        """
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.act = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        blocks = []
        prev_channels = 64
        for i, channels in enumerate(n_channels):
            # if i != 0 and prev_channels != channels:
            #     stride = 2
            # else:
            #     stride = 1
            stride = 2 if i != 0 and prev_channels != channels else 1
            if n_bottlenecks is None:
                blocks.append(ResidualBlock(prev_channels, channels, stride=stride))
            else:
                blocks.append(BottleneckResidualBlock(prev_channels, n_bottlenecks[i], channels, stride=stride))
            prev_channels = channels
            for _ in range(n_blocks[i] - 1):
                if n_bottlenecks is None:
                    blocks.append(ResidualBlock(channels, channels, stride=1))
                else:
                    blocks.append(BottleneckResidualBlock(channels, n_bottlenecks[i], channels, stride=1))
        self.blocks = nn.Sequential(*blocks)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if n_bottlenecks:
            self.fc = nn.Linear(n_channels[-1], num_classes)
        else:
            self.fc = nn.Linear(n_channels[-1], num_classes)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.max_pool(self.act(self.bn(self.conv(x))))
        x = self.blocks(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return self.sm(self.fc(x))
