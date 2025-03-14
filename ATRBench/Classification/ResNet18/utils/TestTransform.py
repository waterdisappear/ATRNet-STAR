"""
Data processing
The mat cell array {number, 1} size needs to be crop to the same size
The amplitude is not processed and the phase is 0 to 2pi radians
2022/3 liweijie
"""
import sys
sys.path.append('..')
import time
import matplotlib as mpl
import torch
import os
import numpy as np
import math
from torchvision import models
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
import random
import skimage
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, RandomRotation
import matplotlib.pyplot as plt
from skimage import util
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']
# torch.manual_seed(1)


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        y = torch.zeros(x.shape)
        for i in range(x.shape[0]):
            if random.random() < self.p:
                y[i, 0, :, :] = self.fn(x[i, 0].clone().detach())
            else:
                y[i, 0, :, :] = x[i, 0].clone().detach()
        return y


class Cuda(object):
    def __init__(self):  # ...是要传入的多个参数
        self.p = 1
    def __call__(self, img):  # __call__函数还是只有一个参数传入
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return img.to(device)


class TEST_AddPepperNoise(object):
    """"
    Args:
        snr (float): Signal Noise Rate
        p (float): 概率值， 依概率执行
    """

    def __init__(self, snr):
        self.snr = snr

    def __call__(self, pic):
        # 原始图像的概率（这里为0.95）
        img = pic.clone().detach()
        signal_pct = self.snr
        # 噪声概率共0.05
        noise_pct = (1 - signal_pct)
        # 按一定概率对（h,w,1）的矩阵使用0，1，2这三个数字进行掩码：掩码为0（原始图像）的概率signal_pct，掩码为1（盐噪声）的概率noise_pct/2.，掩码为2（椒噪声）的概率noise_pct/2.
        mask = np.random.choice((0, 1), size=img.shape, p=[signal_pct, noise_pct])
        mask = torch.from_numpy(mask)
        # plt.subplot(311)
        # plt.imshow(img.cpu())
        noise = torch.rand(size=img.shape)
        img[mask == 1] = noise[mask==1]
        # plt.subplot(312)
        # plt.imshow(img.cpu())
        # plt.subplot(313)
        # plt.imshow(noise.cpu())
        # plt.show()
        # print('1')
        return img


class TEST_AddGussain(object):
    """"
    Args:
        snr (float): Signal Noise Rate
        p (float): 概率值， 依概率执行
    """

    def __init__(self, snr):
        self.snr = snr

    def __call__(self, pic):
        # 原始图像的概率（这里为0.95）
        img = pic.clone().detach()
        snr = 10 ** (self.snr / 10.0)


        xpower = torch.sum(img.flatten() ** 2)/img.shape[0]/img.shape[1]
        npower = xpower / snr

        noise = torch.normal(mean=0, std=np.sqrt(npower), size=pic.shape)

        out = img + noise
        # img = torch.abs(img)

        return (out-out.min()).div(out.max()-out.min())


class TEST_SpecklNoise(object):

    def __init__(self, SNR=0.2):
        self.scale = SNR

    def __call__(self, pic):
        # 把img转化成ndarry的形式.
        img = pic.clone().detach()
        scl = self.scale
        noise = np.random.exponential(scl, img.shape)
        noise[noise>scl] = scl
        out = img * noise
        # out[out > 1] = 1  # 避免有值超过255而反转
        # out[out < 0] = 0  # 避免有值超过255而反转
        # noisy = skimage.util.random_noise(img, mode='speckle', var=var)
        # plt.subplot(311)
        # plt.imshow(img)
        # plt.subplot(312)
        # plt.imshow(out)
        # plt.subplot(313)
        # plt.imshow(out-img)
        # plt.show()
        return (out-out.min()).div(out.max()-out.min())


class TEST_Ersion(object):

    def __init__(self, SNR):
        self.SNR = SNR

    def __call__(self, pic):
        # 把img转化成ndarry的形式.
        img = pic.clone().detach()
        scl = self.SNR

        # print(img.shape)

        # area = img.shape[0]*img.shape[1]
        # target_area = scl*area
        target_area = self.SNR
        aspect_ratio = random.uniform(0.5, 2)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(int(img.shape[0]/2)-32, int(img.shape[0]/2)+32 - h)
            y1 = random.randint(int(img.shape[0]/2)-32, int(img.shape[0]/2)+32 - w)

            img[x1:x1 + h, y1:y1 + w] = 0
            # plt.imshow(img)
            # plt.show()
            return img

        print('Not Ersion')
        return img

