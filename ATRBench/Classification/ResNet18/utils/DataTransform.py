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
import math
import numpy as np
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


def move(img, mask, p=0.3, angle=5, size=16):  # __call__函数还是只有一个参数传入
    img_ = img.clone().detach()
    mask_ = mask.clone().detach()
    for i in range(img_.shape[0]):
        if np.random.rand() < p:
            ang = np.random.randint(-angle, angle)
            img_[i, :, :, :] = TF.rotate(img[i, :, :, :].squeeze().unsqueeze(0).unsqueeze(0).clone().detach(), ang)
            mask_[i, :, :, :] = TF.rotate(mask[i, :, :, :].squeeze().unsqueeze(0).unsqueeze(0).clone().detach(), ang)
        # if np.random.rand() < p:
        #     mvX, mvY = np.random.randint(-size, size), np.random.randint(-size, size)
        #     img_[i, :, :, :] = torch.cat([img_[i, :, mvX:, :].clone().detach(), img_[i, :, :mvX, :].clone().detach()], 1)
        #     img_[i, :, :, :] = torch.cat([img_[i, :, :, mvY:].clone().detach(), img_[i, :, :, :mvY].clone().detach()], 2)
        #     mask_[i, :, :, :] = torch.cat([mask_[i, :, mvX:, :].clone().detach(), mask_[i, :, :mvX, :].clone().detach()], 1)
        #     mask_[i, :, :, :] = torch.cat([mask_[i, :, :, mvY:].clone().detach(), mask_[i, :, :, :mvY].clone().detach()], 2)
    return img_, mask_


class AddPepperNoise(object):
    """"
    Args:
        snr (float): Signal Noise Rate
        p (float): 概率值， 依概率执行
    """

    def __init__(self, snr=0.95):
        # EOC3 VARIANCE IS 0.7
        self.snr = snr

    def __call__(self, pic):
        # 原始图像的概率（这里为0.95）
        img = pic.clone().detach()
        signal_pct = np.random.uniform(self.snr, 1, 1).squeeze()
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

# class AddLogmNoise(object):
#     """"
#     Args:
#         snr (float): Signal Noise Rate
#         p (float): 概率值， 依概率执行
#     """
#
#     def __init__(self, snr=0.90):
#         # EOC3 VARIANCE IS 0.7
#         self.snr = snr
#
#     def __call__(self, pic):
#         # 原始图像的概率（这里为0.95）
#         img = pic.clone().detach()
#         log_mean = np.random.uniform(-2.5, 3.0, 1).squeeze()
#         log_std = np.random.uniform(0.9, 1.2, 1).squeeze()
#         # 噪声概率共0.05
#         signal_pct = np.random.uniform(self.snr, 1, 1).squeeze()
#         noise_pct = (1 - signal_pct)
#         # 按一定概率对（h,w,1）的矩阵使用0，1，2这三个数字进行掩码：掩码为0（原始图像）的概率signal_pct，掩码为1（盐噪声）的概率noise_pct/2.，掩码为2（椒噪声）的概率noise_pct/2.
#         mask = np.random.choice((0, 1), size=img.shape, p=[signal_pct, noise_pct])
#         mask = torch.from_numpy(mask)
#         # plt.subplot(311)
#         # plt.imshow(img.cpu())
#         noise = torch.from_numpy(np.random.lognormal(mean=log_mean, sigma=log_std, size=img.shape)).float()
#         if noise.max()>1:
#             noise = noise/noise.max()
#         img[mask == 1] = noise[mask==1]
#         # plt.subplot(312)
#         # plt.imshow(img.cpu())
#         # plt.subplot(313)
#         # plt.imshow(noise.cpu())
#         # plt.show()
#         # print('1')
#         return img

class AddGaussianNoise(object):

    def __init__(self, mean=0.1, variance=0.1, amplitude=0.5):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, pic):
        # 把img转化成ndarry的形式
        img = pic.clone().detach()
        N = (self.amplitude + np.random.rand()) * torch.normal(mean=self.mean, std=self.variance, size=img.shape)
        out = N + img
        # out[out > 1] = 1  # 避免有值超过255而反转
        out[out < 0] = 0  # 避免有值超过255而反转
        # print(N.max())
        # plt.subplot(311)
        # plt.imshow(img.squeeze())
        # plt.subplot(312)
        # plt.imshow(out)
        # plt.subplot(313)
        # plt.imshow(out-img.squeeze())
        # plt.show()
        return out.div(out.max())

# class AddGaussianNoise(object):
#     # 为EOC3设计
#     def __init__(self, mean=0.0, variance=0.8, amplitude=0.0):
#         self.mean = mean
#         self.variance = variance
#         self.amplitude = amplitude
#
#     def __call__(self, pic):
#         # 把img转化成ndarry的形式
#
#         img = pic.clone().detach()
#         N = (self.amplitude + np.random.rand()) * torch.normal(mean=self.mean, std=self.variance, size=img.shape)
#         out = N + img
#         # out = torch.abs(out)
#         # out[out > 1] = 1  # 避免有值超过255而反转
#         # out[out < 0] = 0  # 避免有值超过255而反转
#         # print(N.max())
#         # plt.subplot(311)
#         # plt.imshow(img.squeeze())
#         # plt.subplot(312)
#         # plt.imshow(out)
#         # plt.subplot(313)
#         # plt.imshow(out-img.squeeze())
#         # plt.show()
#         return (out-out.min()).div(out.max()-out.min())

# class Ersion(object):
#
#     def __init__(self, SNR=0.2, r1=0.3):
#         self.SNR = SNR
#         self.r1 = r1
#         self.mean = 0
#
#     def __call__(self, pic):
#         # 把img转化成ndarry的形式.
#         img = pic.clone().detach()
#         scl = self.SNR
#
#         area = img.shape[0]*img.shape[1]
#         target_area = random.uniform(0.02, self.SNR)*area
#         aspect_ratio = random.uniform(self.r1, 1 / self.r1)
#
#         h = int(round(math.sqrt(target_area * aspect_ratio)))
#         w = int(round(math.sqrt(target_area / aspect_ratio)))
#
#         if w < img.shape[1] and h < img.shape[0]:
#             x1 = random.randint(0, img.shape[0] - h)
#             y1 = random.randint(0, img.shape[1] - w)
#
#             img[x1:x1 + h, y1:y1 + w] = self.mean
#             return img
#
#         print('Not Ersion')
#         return img



# class AddSpecklNoise(object):
#
#     def __init__(self, scale=0.2):
#         self.scale = scale
#
#     def __call__(self, pic):
#         # 把img转化成ndarry的形式.
#         img = pic.clone().detach()
#         scl = self.scale*np.random.rand()
#         noise = np.random.exponential(scl, img.shape)
#         noise[noise>scl] = scl
#         out = img + img * noise
#         # out[out > 1] = 1  # 避免有值超过255而反转
#         out[out < 0] = 0  # 避免有值超过255而反转
#         # noisy = skimage.util.random_noise(img, mode='speckle', var=var)
#         # plt.subplot(311)
#         # plt.imshow(img)
#         # plt.subplot(312)
#         # plt.imshow(out)
#         # plt.subplot(313)
#         # plt.imshow(out-img)
#         # plt.show()
#         return out.div(out.max())


class Cuda(object):
    def __init__(self):  # ...是要传入的多个参数
        self.p = 1
    def __call__(self, img):  # __call__函数还是只有一个参数传入
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return img.to(device)


augment_fn = T.Compose([
    # RandomApply(MyRotateTransform(), p=0.3),
    # RandomApply(AddSpecklNoise(), p=0.3),
    # RandomApply(AddLogmNoise(), p=0.2),
    RandomApply(AddGaussianNoise(), p=0.2),
    RandomApply(AddPepperNoise(), p=0.2),
    Cuda()],
)

# class MyMove(object):
#     """"
#     Args:
#         snr (float): Signal Noise Rate
#         p (float): 概率值， 依概率执行
#     """
#
#     def __init__(self, size=24):
#         self.size = size
#
#     def __call__(self, pic):
#         img = pic.unsqueeze(0).unsqueeze(0)
#         mvX, mvY = np.random.randint(-self.size, self.size), np.random.randint(-self.size, self.size)
#         img_ = torch.cat([img[:, :, mvX:, :].clone().detach(), img[:, :, :mvX, :].clone().detach()], 1)
#         return img_

class MyRotateTransform(object):
    """"
    Args:
        snr (float): Signal Noise Rate
        p (float): 概率值， 依概率执行
    """

    def __init__(self, angle=5):
        self.angle = angle

    def __call__(self, pic):
        # 原始图像的概率（这里为0.95）
        img = pic.unsqueeze(0).unsqueeze(0)
        ang = np.random.randint(-self.angle, self.angle)
        out = TF.rotate(img, ang).squeeze()
        # plt.subplot(311)
        # plt.imshow(img.squeeze())
        # plt.subplot(312)
        # plt.imshow(out)
        # plt.subplot(313)
        # plt.imshow(out-img.squeeze())
        # plt.show()
        return out


other_augment_fn = T.Compose([
    RandomApply(MyRotateTransform(), p=0.3),
    # RandomApply(MyMove(), p=0.2),
    RandomApply(AddGaussianNoise(), p=0.2),
    RandomApply(AddPepperNoise(), p=0.2),
    Cuda()],
)




