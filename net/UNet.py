from __future__ import print_function

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up_conv(x)
        return x


class Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(Net, self).__init__()

        layer1_ch = 64
        layer2_ch = 128
        layer3_ch = 256
        layer4_ch = 512
        layer5_ch = 1024

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = ConvBlock(ch_in=img_ch, ch_out=layer1_ch)
        self.Conv2 = ConvBlock(ch_in=layer1_ch, ch_out=layer2_ch)
        self.Conv3 = ConvBlock(ch_in=layer2_ch, ch_out=layer3_ch)
        self.Conv4 = ConvBlock(ch_in=layer3_ch, ch_out=layer4_ch)
        # self.Conv5 = ConvBlock(ch_in=layer4_ch, ch_out=layer5_ch)
        #
        # self.Up5 = UpConv(ch_in=layer5_ch, ch_out=layer4_ch)
        # self.Up_conv5 = ConvBlock(ch_in=layer5_ch, ch_out=layer4_ch)

        self.Up4 = UpConv(ch_in=layer4_ch, ch_out=layer3_ch)
        self.Up_conv4 = ConvBlock(ch_in=layer4_ch, ch_out=layer3_ch)

        self.Up3 = UpConv(ch_in=layer3_ch, ch_out=layer2_ch)
        self.Up_conv3 = ConvBlock(ch_in=layer3_ch, ch_out=layer2_ch)

        self.Up2 = UpConv(ch_in=layer2_ch, ch_out=layer1_ch)
        self.Up_conv2 = ConvBlock(ch_in=layer2_ch, ch_out=layer1_ch)

        self.Conv_1x1 = nn.Conv2d(layer1_ch, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        # decoding

        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
