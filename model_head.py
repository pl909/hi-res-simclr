import torch
import torch.nn as nn
"""
DenseNet Custom Head:

This module contains custom layers and two variations of a head designed for use with a DenseNet architecture. 
These custom heads are named DenseNet448 and DenseNet896, and as the names suggest, they are tailored for 
different input sizes and depths.

Key Components:
1. ConvBNReLU: A module that combines a 2D convolution, batch normalization, and an optional ReLU activation.
2. ResIdentity: A residual block where the input is added to the processed output without any changes.
3. ResConv: A residual block with a convolutional skip connection.
4. DenseNet448 and DenseNet896: Two custom head modules differing in depth and size.

Note: These heads can be integrated with the main DenseNet architecture as per requirement.
"""


class ConvBNReLU(nn.Module):
    """Combines a 2D convolution, followed by batch normalization and optionally ReLU activation."""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, apply_activation=True):
        super(ConvBNReLU, self).__init__()
        self.apply_activation = apply_activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.apply_activation:
            x = self.relu(x)
        return x

class ResIdentity(nn.Module):
    """Residual block with identity skip connection."""
    def __init__(self, in_channels, filters):
        super(ResIdentity, self).__init__()
        f1, f2 = filters
        # Main path consists of three convolutional layers
        self.main = nn.Sequential(
            ConvBNReLU(in_channels, f1, kernel_size=1),
            ConvBNReLU(f1, f1, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(f1, f2, kernel_size=1, apply_activation=False),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Add the input to the processed output
        return self.relu(x + self.main(x))

class ResConv(nn.Module):
    """Residual block with convolutional skip connection."""
    def __init__(self, in_channels, s, filters):
        super(ResConv, self).__init__()
        f1, f2 = filters
        # Main path consists of three convolutional layers
        self.main = nn.Sequential(
            ConvBNReLU(in_channels, f1, kernel_size=1, stride=s),
            ConvBNReLU(f1, f1, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(f1, f2, kernel_size=1, apply_activation=False),
        )
        # Skip connection with convolutional layer
        self.skip = ConvBNReLU(in_channels, f2, kernel_size=1, stride=s, apply_activation=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Add the convolutionally processed input to the main path output
        return self.relu(self.main(x) + self.skip(x))

class DenseNet448(nn.Module):
    """Custom head module with fewer layers."""
    def __init__(self):
        super(DenseNet448, self).__init__()
        self.conv1 = ConvBNReLU(3, 32, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResConv(32, s=1, filters=(32, 128))
        self.res2 = ResIdentity(128, filters=(32, 128))
        self.res3 = ResIdentity(128, filters=(32, 128))
        self.res4 = ResIdentity(128, filters=(32, 128))
        self.conv2 = ConvBNReLU(128, 64, kernel_size=1, stride=2)

    def forward(self, x):
        # Define the forward pass for the DenseNet448
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.conv2(x)
        return x

class DenseNet896(nn.Module):
    """Custom head module with more layers for deeper networks."""
    def __init__(self):
        super(DenseNet896, self).__init__()

        # Initial layers
        self.conv1 = ConvBNReLU(3, 16, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # First set of residual blocks
        self.res1 = ResConv(16, s=1, filters=(16, 64))
        self.res2 = ResIdentity(64, filters=(16, 64))
        self.res3 = ResIdentity(64, filters=(16, 64))

        # Second set of residual blocks
        self.res4 = ResConv(64, s=2, filters=(32, 128))
        self.res5 = ResIdentity(128, filters=(32, 128))
        self.res6 = ResIdentity(128, filters=(32, 128))
        self.res7 = ResIdentity(128, filters=(32, 128))

        # Final convolution
        self.conv2 = ConvBNReLU(128, 64, kernel_size=1, stride=2)

    def forward(self, x):
        # Define the forward pass for the DenseNet896
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.conv2(x)
        return x
