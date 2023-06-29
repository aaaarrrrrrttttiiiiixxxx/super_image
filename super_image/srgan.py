import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch import nn
import numpy as np
from torchvision.models import vgg19
from torchmetrics import TotalVariation
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F

from datasets import load_dataset

import torchmetrics
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ModelSummary
from lightning.pytorch.tuner import Tuner

from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.callbacks import Callback
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.transforms import Normalize

import matplotlib.pyplot as plt
import multiprocessing
from collections import OrderedDict
from torchvision import datasets, transforms, utils
import torchvision


class Conv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides=1,
                 padding='same',
                 dilation=1,
                 groups=1,
                 activation=nn.ReLU,
                 if_act=True,
                 if_batch_norm=True):
        super().__init__()
        layers = []
        if (if_batch_norm):
            bias = False
        else:
            bias = True
        conv2D = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=strides,
                           padding=padding, dilation=dilation, groups=groups, bias=bias)
        batch_norm = nn.BatchNorm2d(out_channels)
        layers.append(conv2D)
        if (if_batch_norm):
            layers.append(batch_norm)
        if (if_act):
            layers.append(activation)
        self.convolution2D = nn.Sequential(*layers)

    def forward(self, x):
        return self.convolution2D(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv_act_bn_1 = Conv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3),
                                    activation=nn.PReLU())
        self.conv_act_bn_2 = Conv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3),
                                    activation=nn.PReLU(), if_act=False)

    def forward(self, x):
        residual = self.conv_act_bn_1(x)
        residual = self.conv_act_bn_2(residual)
        return residual + x


class UpsampleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(UpsampleBlock, self).__init__()
        conv_1 = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), if_act=False,
                        if_batch_norm=False)
        pixel_shuffle = nn.PixelShuffle(2)
        activation = nn.PReLU()
        self.upsampling = nn.Sequential(*[conv_1, pixel_shuffle, activation])

    def forward(self, x):
        x = self.upsampling(x)
        return x


class SRGANGenerator(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 upsampling_rate):
        super(SRGANGenerator, self).__init__()
        self.first_res_conv = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(9, 9),
                                     activation=nn.PReLU(), if_batch_norm=False)
        self.residual_seq = nn.Sequential(*[ResidualBlock(out_channels) for i in range(16)])
        self.last_res_conv = Conv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3),
                                    if_act=False, if_batch_norm=True)
        self.upsampling_seq = nn.Sequential(
            *[UpsampleBlock(in_channels=out_channels, out_channels=(out_channels * upsampling_rate)) for i in
              range(upsampling_rate // 2)])
        self.last_conv = Conv2D(in_channels=out_channels, out_channels=in_channels, kernel_size=(9, 9),
                                activation=nn.PReLU(), if_batch_norm=False)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.first_res_conv(x)
        new_x = self.residual_seq(x)
        new_x = self.last_res_conv(new_x) + x
        new_x = self.upsampling_seq(new_x)
        new_x = self.last_conv(new_x)
        new_x = self.activation(new_x)
        return new_x


# метод улучшения при помощи srgan
def srgan_upscale(filename):
    generator = SRGANGenerator(in_channels=3, out_channels=64, upsampling_rate=4)
    generator.load_state_dict(torch.load('models/srgan_generator.pth'))

    low_r_real = cv2.imread(filename)
    low_r_real = cv2.cvtColor(low_r_real, cv2.COLOR_BGR2RGB)
    low_r_real = torch.as_tensor(low_r_real).permute(2, 0, 1) / 255

    generator.eval()
    high_r_fake = (generator(low_r_real[None, :])[0])
    high_r_fake = high_r_fake.permute(1, 2, 0).detach().numpy()
    high_r_fake = (high_r_fake + 1) / 2
    return high_r_fake


class EUpsampleBlock(nn.Module):
    def __init__(self,
                 in_channels):
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                      nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                                                stride=1, padding=1),
                                      nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.upsample(x)


class DenseResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 channels=32,
                 residual_beta=0.2):
        super(DenseResidualBlock, self).__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()

        for i in range(5):
            self.blocks.append(
                Conv2D(in_channels=in_channels + channels * i, out_channels=channels if i <= 3 else in_channels,
                       kernel_size=(3, 3), activation=nn.LeakyReLU(0.2, inplace=True), if_act=True if i <= 3 else False,
                       if_batch_norm=False)
            )

    def forward(self, x):
        new_inputs = x
        for block in self.blocks:
            out = block(new_inputs)
            new_inputs = torch.cat([new_inputs, out], dim=1)
        return self.residual_beta * out + x


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 residual_beta=0.2
                 ):
        super(BasicBlock, self).__init__()
        self.residual_beta = residual_beta
        self.block = nn.Sequential(*[DenseResidualBlock(in_channels) for _ in range(3)])

    def forward(self, x):
        return self.block(x) * self.residual_beta + x


class ESRGANGenerator(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=64,
                 upsampling_rate=4):
        super().__init__()
        self.first_res_conv = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                     activation=nn.LeakyReLU(0.2, inplace=True), if_batch_norm=False)
        self.residual_seq = nn.Sequential(*[BasicBlock(out_channels) for i in range(23)])
        self.last_res_conv = Conv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3),
                                    if_act=False, if_batch_norm=False)
        self.upsampling_seq = nn.Sequential(
            *[EUpsampleBlock(in_channels=out_channels) for i in range(upsampling_rate // 2)])
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
        )
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.first_res_conv(x)
        new_x = self.residual_seq(x)
        new_x = self.last_res_conv(new_x) + x
        new_x = self.upsampling_seq(new_x)
        new_x = self.final(new_x)
        return self.activation(new_x)


# метод улучшения при помощи esrgan
def esrgan_upscale(filename):
    generator = ESRGANGenerator(in_channels=3, out_channels=64, upsampling_rate=4)
    generator.load_state_dict(torch.load('models/esrgan_generator.pth'))

    low_r_real = cv2.imread(filename)
    low_r_real = cv2.cvtColor(low_r_real, cv2.COLOR_BGR2RGB)
    low_r_real = torch.as_tensor(low_r_real).permute(2, 0, 1) / 255

    generator.eval()
    with torch.no_grad():
        high_r_fake = (generator(low_r_real[None, :])[0])
    high_r_fake = high_r_fake.permute(1, 2, 0).detach().numpy()
    high_r_fake = (high_r_fake + 1) / 2
    return high_r_fake
