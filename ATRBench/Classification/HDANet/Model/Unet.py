# -*- coding: utf-8 -*-
import torch
import argparse
import matplotlib.pyplot as plt
import sys
import cv2
sys.path.append('..')
from torchvision import models
from scipy.io import loadmat, savemat
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from captum.attr import (GuidedGradCam, IntegratedGradients, NoiseTunnel, DeepLift, DeepLiftShap, ShapleyValueSampling,
                         KernelShap, GradientShap, FeatureAblation)
import random
import math
import os
import random
from os import listdir
from os.path import isfile, join


def crop(x1, x2):
    '''
        conv output shape = (input_shape - Filter_shape + 2 * padding)/stride + 1
    '''

    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX//2,
                                diffY // 2, diffY - diffY//2))

    x = torch.cat([x2, x1], dim=1)
    return x


class BN_Conv2d(nn.Module):
    """
    BN_CONV_RELU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False) -> object:
        super(BN_Conv2d, self).__init__()
        self.padding = nn.ReflectionPad2d(padding)
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.padding(x)
        return F.relu(self.seq(x))


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, padding=2, bias=False):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            BN_Conv2d(in_ch, out_ch, 5, stride=1, padding=padding, bias=bias),
            BN_Conv2d(out_ch, out_ch, 5, stride=1, padding=padding, bias=bias)
        )

    def forward(self, input):
        return self.conv(input)


def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-8)


class PrimaryCaps(nn.Module):
    """Primary capsule layer."""

    def __init__(self, num_conv_units, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()

        # Each conv unit stands for a single capsule.
        # self.conv = nn.Conv2d(in_channels=in_channels,
        #                       out_channels=out_channels * num_conv_units,
        #                       kernel_size=kernel_size,
        #                       stride=stride)
        self.conv = BN_Conv2d(in_channels, out_channels * num_conv_units, kernel_size, stride=stride, padding=0, bias=False)
        self.out_channels = out_channels

    def forward(self, x):
        # Shape of x: (batch_size, in_channels, height, weight)
        # Shape of out: num_capsules * (batch_size, out_channels, height, weight)
        out = self.conv(x)
        # Flatten out: (batch_size, num_capsules * height * weight, out_channels)
        batch_size = out.shape[0]
        return squash(out.contiguous().view(batch_size, -1, self.out_channels), dim=-1)


class DigitCaps(nn.Module):
    """Digit capsule layer."""

    def __init__(self, in_dim, in_caps, num_caps, dim_caps, num_routing):
        """
        Initialize the layer.

        Args:
            in_dim: 		Dimensionality (i.e. length) of each capsule vector.
            in_caps: 		Number of input capsules if digits layer.
            num_caps: 		Number of capsules in the capsule layer
            dim_caps: 		Dimensionality, i.e. length, of the output capsule vector.
            num_routing:	Number of iterations during routing algorithm
        """
        super(DigitCaps, self).__init__()
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.num_routing = num_routing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.W = nn.Parameter(0.01 * torch.randn(1, num_caps, in_caps, dim_caps, in_dim),
                              requires_grad=True)
    def forward(self, x):
        batch_size = x.size(0)
        # (batch_size, in_caps, in_dim) -> (batch_size, 1, in_caps, in_dim, 1)
        x = x.unsqueeze(1).unsqueeze(4)
        #
        # W @ x =
        # (1, num_caps, in_caps, dim_caps, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
        # (batch_size, num_caps, in_caps, dim_caps, 1)
        u_hat = torch.matmul(self.W, x)
        # (batch_size, num_caps, in_caps, dim_caps)
        u_hat = u_hat.squeeze(-1)
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()

        b = torch.zeros(batch_size, self.num_caps, self.in_caps, 1).to(self.device)

        for route_iter in range(self.num_routing - 1):
            # (batch_size, num_caps, in_caps, 1) -> Softmax along num_caps
            c = b.softmax(dim=1)

            # element-wise multiplication
            # (batch_size, num_caps, in_caps, 1) * (batch_size, in_caps, num_caps, dim_caps) ->
            # (batch_size, num_caps, in_caps, dim_caps) sum across in_caps ->
            # (batch_size, num_caps, dim_caps)
            s = (c * temp_u_hat).sum(dim=2)
            # apply "squashing" non-linearity along dim_caps
            v = squash(s)
            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, num_caps, in_caps, dim_caps) @ (batch_size, num_caps, dim_caps, 1)
            # -> (batch_size, num_caps, in_caps, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b += uv

        # last iteration is done on the original u_hat, without the routing weights update
        c = b.softmax(dim=1)
        s = (c * u_hat).sum(dim=2)
        # apply "squashing" non-linearity along dim_caps
        v = squash(s)

        return v


def MLP(dim, projection_size, hidden_size=64):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=64):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )


class Unet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, num_classes=3):
        super(Unet, self).__init__()
        base_channel = 32
        avgpool = 8

        self.conv1 = DoubleConv(in_ch, base_channel)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(base_channel, base_channel * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(base_channel * 2, base_channel * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(base_channel * 4, base_channel * 8)

        self.mask = nn.Sequential(DoubleConv(base_channel * 8, 1), nn.Tanh())
        # ConvTranspose2d output = (input-1)stride+outputpadding -2padding+kernelsize
        self.up6 = DoubleConv(1, base_channel * 8)
        self.conv6 = DoubleConv(base_channel * 16, base_channel * 8)
        self.up7 = nn.ConvTranspose2d(base_channel * 8, base_channel * 4, 2, stride=2)
        self.conv7 = DoubleConv(base_channel * 8, base_channel * 4)
        self.up8 = nn.ConvTranspose2d(base_channel * 4, base_channel * 2, 2, stride=2)
        self.conv8 = DoubleConv(base_channel * 4, base_channel * 2)
        self.up9 = nn.ConvTranspose2d(base_channel * 2, base_channel, 2, stride=2)
        self.conv9 = DoubleConv(base_channel * 2, base_channel)
        self.conv10 = nn.Sequential(nn.Conv2d(base_channel, out_ch, 1), nn.ReLU(), nn.Tanh())

        self.Project = SimSiamMLP(8, 4)
        self.MLP = MLP(4, 4)

        self.avg = nn.AdaptiveMaxPool2d((avgpool, avgpool))
        # Primary capsule
        self.primary_caps = PrimaryCaps(num_conv_units=16,
                                        in_channels=base_channel * 15,
                                        out_channels=8,
                                        kernel_size=3,
                                        stride=1)
        # Digit capsule
        self.digit_caps = DigitCaps(in_dim=8,
                                    in_caps=16 * 6 * 6,
                                    num_caps=num_classes,
                                    dim_caps=8,
                                    num_routing=4)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        mask = self.mask(c4)

        feature1 = torch.cat([self.avg(c1), self.avg(c2), self.avg(c3), self.avg(c4)], 1) \
            .mul(self.avg(mask).repeat(1, 32 * 15, 1, 1))
        feature2 = self.primary_caps(feature1)
        out = self.digit_caps(feature2)
        logits = torch.norm(out, dim=-1)

        project = self.Project(feature2.view(-1, 8))
        predict = self.MLP(project)

        up_6 = self.up6(mask)
        merge6 = crop(up_6, c4)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = crop(up_7, c3)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = crop(up_8, c2)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = crop(up_9, c1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        return logits