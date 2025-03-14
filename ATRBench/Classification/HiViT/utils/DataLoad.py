"""
Data processing
The mat cell array {number, 1} size needs to be crop to the same size
The amplitude is not processed and the phase is 0 to 2pi radians
2022/3 liweijie
"""
import sys
sys.path.append('..')
import scipy.io as io
import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import itertools
from PIL import Image
from scipy.io import loadmat
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import os
import time
import re
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import xml.etree.ElementTree as ET
from torchvision import transforms, utils
import torch.nn as nn
from captum.attr import (GuidedGradCam, IntegratedGradients, NoiseTunnel, DeepLift, DeepLiftShap, ShapleyValueSampling, KernelShap, GradientShap, FeatureAblation)
import torchvision
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']
torch.manual_seed(1)

from torch.utils import data
import numpy as np
from PIL import Image

data_transform = transforms.Compose([
    transforms.Resize(224),  # 缩放到 96 * 96 大小
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class custom_dset(Dataset):
    def __init__(self,
                 img_path,
                 txt_path,
                 img_transform=None,):
        self.img_list =img_path
        self.label_list = txt_path
        self.img_transform = img_transform

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path)  # 可以读取单通道影像,读取3通道16位tif影像时报错(PIL.UnidentifiedImageError: cannot identify image file),支持4通道8位影像

        label = self.label_list[index]
        label = torch.Tensor([label]).type(torch.LongTensor).squeeze()
        # img = self.loader(img_path)
        img = np.expand_dims(img,axis=2)
        img = np.concatenate([img, img, img], axis=2)
        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, label

    def __len__(self):
        return len(self.label_list)

def load_data(file_dir, transform):
    # path_list = []
    #
    # for root, dirs, files in os.walk(file_dir):
    #     files = sorted(files)
    #     for file in files:
    #         if os.path.splitext(file)[1] == '.tif':
    #             path_list.append(os.path.join(root, file))
    # label_list = np.zeros(len(path_list), dtype="int64")
    # for i, jpeg_path in tqdm(enumerate(path_list)):
    #     tree = ET.parse(jpeg_path.replace('tif', 'xml'))
    #     root = tree.getroot()
    #     target_id = root.find('object').find('target_id')
    #     target_id = np.array(int(target_id.text) - 1, dtype="int64")
    #     label_list[i] = target_id
    #
    # data_set = custom_dset(path_list, label_list, transform)
    data_set = datasets.ImageFolder(file_dir, transform=transform)
    return data_set

# class custom_dset(Dataset):
#     def __init__(self,
#                  img_path,
#                  txt_path,
#                  img_transform=None,):
#         self.img_list =img_path
#         self.label_list = txt_path
#         self.img_transform = img_transform
#
#     def __getitem__(self, index):
#         img_path = self.img_list[index]
#         img = Image.open(img_path)  # 可以读取单通道影像,读取3通道16位tif影像时报错(PIL.UnidentifiedImageError: cannot identify image file),支持4通道8位影像
#
#         label = self.label_list[index]
#         label = torch.Tensor([label]).type(torch.LongTensor).squeeze()
#         # img = self.loader(img_path)
#         img = np.expand_dims(img,axis=2)
#         img = np.concatenate([img, img, img], axis=2)
#         img = Image.fromarray(img)
#         if self.img_transform is not None:
#             img = self.img_transform(img)
#
#         return img, label
#
#     def __len__(self):
#         return len(self.label_list)
#
# def load_data(file_dir, transform):
#     path_list = []
#
#     for root, dirs, files in os.walk(file_dir):
#         files = sorted(files)
#         for file in files:
#             if os.path.splitext(file)[1] == '.tif':
#                 path_list.append(os.path.join(root, file))
#     label_list = np.zeros(len(path_list), dtype="int64")
#     for i, jpeg_path in tqdm(enumerate(path_list)):
#         tree = ET.parse(jpeg_path.replace('tif', 'xml'))
#         root = tree.getroot()
#         target_id = root.find('object').find('target_id')
#         target_id = np.array(int(target_id.text) - 1, dtype="int64")
#         label_list[i] = target_id
#
#     data_set = custom_dset(path_list, label_list, transform)
#     return data_set
#
