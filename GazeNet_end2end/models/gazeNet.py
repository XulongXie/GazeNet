import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data

import math


# Mish activation function
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, t):
        # Mish function equation：x * tanh(ln(1 + e^x))， with ln(1 + e^x) or softplus
        return t * torch.tanh(F.softplus(t))


# like CSPdarknet53 convolution block：Conv2D + BN + Mish
class BasicConv(nn.Module):
    # with some parameters to build this block, otherwise need to build a lot of different blocks
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1):
        super(BasicConv, self).__init__()
        # convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias = False)
        # BN
        self.bn = nn.BatchNorm2d(out_channels)
        # Mish
        self.activation = Mish()
    def forward(self, t):
        # convolution
        t = self.conv(t)
        # BN
        t = self.bn(t)
        # activation
        t = self.activation(t)
        return t


# modul for eyes
class GazeTrackerModul(nn.Module):
    # Used for both eyes (with shared weights)
    def __init__(self):
        super(GazeTrackerModul, self).__init__()
        # first convolution block 224,224,1 -> 224,224,32
        self.conv1 = BasicConv(1, 32, kernel_size = 7, stride = 1)
        # down sampling 224,224,32 -> 112,112,32
        self.down_sampling1 = nn.MaxPool2d(3, stride = 2, padding = 1, dilation = 1, return_indices = False, ceil_mode = False)
        # second convolution block 112,112,32 -> 112,112,32
        self.conv2 = BasicConv(32, 32, kernel_size = 5, stride = 1)
        # down sampling 112,112,32 -> 56,56,32
        self.down_sampling2 = nn.MaxPool2d(3, stride = 2, padding = 1, dilation = 1, return_indices = False, ceil_mode = False)
        # third convolution block 56,56,32 -> 56,56,32
        self.conv3 = BasicConv(32, 32, kernel_size = 3, stride = 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal_(m.weight)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # the weights of the convolutional layer are normally distributed
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, t):
        t = self.conv1(t)
        t = self.down_sampling1(t)
        t = self.conv2(t)
        t = self.down_sampling2(t)
        t = self.conv3(t)
        # flatten 56*56*32
        t = t.view(t.size(0), -1)
        return t


# The whole modul
class GazeNetModel(nn.Module):

    def __init__(self):
        super(GazeNetModel, self).__init__()
        self.GazeModel = GazeTrackerModul()
        # Fully Connected Layers
        self.gazeFC = nn.Sequential(
            nn.Linear(56 * 56 * 32, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, 9),
        )

    def forward(self, img):
        # features extraction
        features = self.GazeModel(img)
        # full connection
        t = self.gazeFC(features)
        t = F.softmax(t, dim = 1)

        return t
