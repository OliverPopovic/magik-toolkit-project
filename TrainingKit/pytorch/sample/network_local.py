import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
    
class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, use_bn=True):
        super().__init__()
        layers = [nn.Linear(in_features, out_features, bias=True)]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU6(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class NetworkT40Local(nn.Module):
    """
    A plain PyTorch approximation of the repo's Network_T40.

    It preserves the broad tensor flow:
    BN -> conv -> pool -> conv -> concat -> conv -> conv -> upsample
    -> conv -> pool -> conv -> add -> BN/PReLU -> conv stack -> fc -> fc
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.input_bn = nn.BatchNorm2d(3)

        self.conv0 = ConvBlock(3, 32, 3, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(32, 32, 3, 1, 1)
        self.conv2 = ConvBlock(64, 32, 3, 2, 1)
        self.conv3 = ConvBlock(32, 64, 3, 1, 1)

        self.unpool = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv4 = ConvBlock(64, 64, 3, 1, 1)
        self.conv5 = ConvBlock(64, 64, 3, 1, 1)

        self.post_add = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )

        self.conv6 = ConvBlock(64, 128, 3, 2, 1)
        self.conv7 = ConvBlock(128, 128, 3, 1, 1)
        self.conv8 = ConvBlock(128, 256, 3, 2, 1)
        self.conv9 = ConvBlock(256, 256, 3, 1, 1)
        self.conv10 = ConvBlock(256, 256, 3, 2, 1)

        self.fc1 = FCBlock(256, 128, use_bn=True)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.input_bn(x)

        conv0 = self.conv0(x)
        max1 = self.maxpool(conv0)

        conv1 = self.conv1(max1)
        concat1 = torch.cat([max1, conv1], dim=1)

        conv2 = self.conv2(concat1)
        conv3 = self.conv3(conv2)

        unpool1 = self.unpool(conv3)
        conv4 = self.conv4(unpool1)

        max2 = self.maxpool(conv4)
        conv5 = self.conv5(max2)

        concat2 = max2 + conv5
        concat2 = self.post_add(concat2)

        conv6 = self.conv6(concat2)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)
        conv9 = self.conv9(conv8)
        conv10 = self.conv10(conv9)

        x = torch.flatten(conv10, 1)  # [B, 256, 1, 1] -> [B, 256]
        x = self.fc1(x)
        x = self.fc2(x)
        return x