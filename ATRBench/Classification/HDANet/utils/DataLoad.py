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


class Train_dataset(data.Dataset):
    def __init__(self, jpg_list):
        self.jpg_list = jpg_list

    def __getitem__(self,index):

        jpeg_path = self.jpg_list[index]
        jpeg = Image.open(self.jpg_list[index])  # 可以读取单通道影像,读取3通道16位tif影像时报错(PIL.UnidentifiedImageError: cannot identify image file),支持4通道8位影像
        jpeg = np.array(jpeg)
        data = np.expand_dims(jpeg, axis=0)
        mask = sailency(data=data, id=0)

        tree = ET.parse(jpeg_path.replace('tif', 'xml'))
        root = tree.getroot()
        target_id = root.find('object').find('target_id')
        label = np.array(int(target_id.text)-1, dtype="int64")

        return torch.from_numpy(data).type(torch.FloatTensor), torch.from_numpy(mask).type(torch.FloatTensor).squeeze().unsqueeze(0), torch.from_numpy(label)

    def __len__(self):
        return len(self.jpg_list)

class Test_dataset(data.Dataset):
    def __init__(self, jpg_list):
        self.jpg_list = jpg_list

    def __getitem__(self,index):
        jpeg_path = self.jpg_list[index]
        jpeg = Image.open(self.jpg_list[index])  # 可以读取单通道影像,读取3通道16位tif影像时报错(PIL.UnidentifiedImageError: cannot identify image file),支持4通道8位影像
        jpeg = np.array(jpeg)
        data = np.expand_dims(jpeg, axis=1)

        tree = ET.parse(jpeg_path.replace('tif', 'xml'))
        root = tree.getroot()
        target_id = root.find('object').find('target_id')
        label = np.array(int(target_id.text)-1, dtype="int64")

        return torch.from_numpy(data).type(torch.FloatTensor), torch.from_numpy(label)

    def __len__(self):
        return len(self.jpg_list)

class SAR_VGG16(torch.nn.Module):

    def __init__(self, feature_extract=True, num_classes=10):
        super(SAR_VGG16, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # torch.nn.MaxPool2d(kernel_size=2, stride=1)
        # batsize,channel,height,length 20*3*128*128

        # 3 * 128 * 128
        # 导入VGG16模型
        cfgs = {
            'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M'],
        }
        # 加载features部分
        def make_features(cfg: list):
            """
            提取特征网络结构，
            cfg.list：传入配置变量，只需要传入对应配置的列表
            """
            layers = []  # 空列表，用来存放所创建的每一层结构
            in_channels = 3  # 输入数据的深度，RGB图像深度数为3
            for v in cfg:
                if v == "M":
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                    # 若为最大池化层，创建池化操作，并为卷积核大小设置为2，步距设置为2，并将其加入到layers
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    layers += [conv2d, nn.ReLU()]
                    in_channels = v
                    # 创建卷积操作，定义输入深度，配置变量，卷积核大小为3，padding操作为1，并将Conv2d和ReLU激活函数加入到layers列表中
            return nn.Sequential(*layers)
        self.features = make_features(cfgs['vgg16'])
        # 固定特征提取层参数
        # set_parameter_requires_grad(self.features, feature_extract)
        # 加载avgpool层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 改变classifier：分类输出层
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(64, 64),
            nn.Dropout(p=0.5, inplace=False),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = torch.cat([x, x, x], 1)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 512 * 1 * 1)
        out = self.classifier(x)
        return out

# def pic2binary(pic_data):
#     redata = np.zeros(pic_data.shape)
#     for i in range(pic_data.shape[0]):
#         temp = pic_data[i].squeeze()
#         # from scipy.signal import wiener
#         # temp = wiener(temp, [9, 9])
#         # pic = np.uint8(pic / pic.max() * 255)
#         image = np.array(temp*255, dtype='uint8')
#         retval, dst = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
#         dst = cv2.medianBlur(dst, 5)
#         # dst = cv2.erode(dst, None, iterations=1)
#         dst = cv2.dilate(dst, None, iterations=2)
#         # plt.imshow(dst)
#         # plt.show()
#         # # 腐蚀和膨胀是对白色部分而言的，膨胀，白区域变大，最后的参数为迭代次数
#         # dst = cv2.dilate(dst, None, iterations=1)
#         # # 腐蚀，白区域变小
#         # dst = cv2.erode(dst, None, iterations=4)
#         # cv2.namedWindow("Image")  # 图片显示框的名字 这行没啥用
#         # cv2.imshow("Image", dst)  # 图片显示
#         redata[i, 0, :, :] = dst/255
#     return redata


def sailency(data, id=0, target_number=10):
    use_cuda = torch.cuda.is_available()
    torch.cuda.set_device(id)
    device = torch.device("cuda" if use_cuda else "cpu")
    data_set = torch.utils.data.TensorDataset(torch.from_numpy(data).float().type(torch.FloatTensor))
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=50, shuffle=False)
    sa_model = SAR_VGG16().to(device)
    sa_model.load_state_dict(torch.load('./Model/VGG16.pth', map_location=device), False)
    sa_model.eval()
    # way = IntegratedGradients(sa_model)
    # way = NoiseTunnel(IG)
    # way = DeepLift(sa_model)
    way = GuidedGradCam(sa_model, sa_model.features[29])
    # way = IntegratedGradients(sa_model)
    way = NoiseTunnel(way)
    for i, data in tqdm(enumerate(data_loader)):
        a = data[0]
        # print(a.shape)
        # prediction1 = sa_model(a.to(device))
        # pred = (prediction1.data.max(1, keepdim=True)[1]).squeeze()
        a0 = np.zeros(a.shape)
        for j in range(target_number):
            # a0 += way.attribute(a.to(device), target=torch.tensor(j).to(device), baselines=a.to(device) * 0).cpu().detach()
            # a0 += way.attribute(a.to(device), target=torch.tensor(j).to(device)).abs().cpu().detach().numpy()
            a0 += way.attribute(a.to(device), target=torch.tensor(j).to(device), nt_type='smoothgrad_sq', nt_samples=5, stdevs=0.05)\
                        .abs().cpu().detach().numpy()
        # a0 = nt_ig.attribute(a.to(device), nt_type='vargrad', nt_samples=5, target=pred.to(device), stdevs=0.025)
        # a0 = (a0.reshape(a.shape[0]*a.shape[1], -1) /
        #       (torch.max(a0.reshape(a.shape[0]*a.shape[1], -1), 1)[1].data.unsqueeze(1).repeat(1, a.shape[2] * a.shape[3])))\
        #     .reshape(a.shape)
        for j in range(a0.shape[0]):
            temp = a0[j].squeeze()
            max_val = np.nanpercentile(temp.flatten(), 100)
            # min_val = np.nanpercentile(temp.flatten(), 0)
            image = np.array((temp/max_val), dtype='float')
            # image[image > 1] = 1
            image[image < 0.1] = 0
            # from scipy.signal import wiener
            # image = wiener(image, [9, 9])
            a0[j, 0, :, :] = image

        # a[a < 0.2] = 0
        # a0 = pic2binary(a0)
        # plt.subplot(211)
        # plt.imshow(a[0,0,:,:])
        # plt.subplot(212)
        # abs_vals = np.abs(a0[0,0,:,:]).flatten()
        # max_val = np.nanpercentile(abs_vals, 100)
        # plt.imshow(a0[0,0,:,:].squeeze(), cmap="hot", vmin=0, vmax=max_val)
        # plt.show()
        if i == 0:
            bb = a0
        else:
            bb = np.concatenate((bb, a0), axis=0)
    return bb


def crop_transform(picture_size):
    return Compose([
        # Resize(picture_size),
        CenterCrop(picture_size), ])


# def load_data(file_dir):
#     path_list = []
#     for root, dirs, files in os.walk(file_dir):
#         files = sorted(files)
#         for file in files:
#             if os.path.splitext(file)[1] == '.tif':
#                 path_list.append(os.path.join(root, file))
#     data_set = Train_dataset(path_list)
#     return data_set
#
#
# def load_test(file_dir):
#     path_list = []
#     for root, dirs, files in os.walk(file_dir):
#         files = sorted(files)
#         for file in files:
#             if os.path.splitext(file)[1] == '.tif':
#                 path_list.append(os.path.join(root, file))
#     data_set = Test_dataset(path_list)
#     return data_set

def load_data(file_dir, id=0, picture_size=128):
    path_list = []
    jpeg_list = []
    label_list = []
    for root, dirs, files in os.walk(file_dir):
        files = sorted(files)
        for file in files:
            if os.path.splitext(file)[1] == '.tif':
                path_list.append(os.path.join(root, file))
    for jpeg_path in tqdm(path_list):
        jpeg = Image.open(jpeg_path)  # 可以读取单通道影像,读取3通道16位tif影像时报错(PIL.UnidentifiedImageError: cannot identify image file),支持4通道8位影像
        jpeg = np.array(jpeg)
        # jpeg = loadmat(jpeg_path)['Img']
        pic = crop_transform(picture_size)(torch.from_numpy(jpeg))
        jpeg_list.append(np.array(pic.div(pic.max())))
        # jpeg_list.append(np.array(pic))
        tree = ET.parse(jpeg_path.replace('tif', 'xml'))
        root = tree.getroot()
        target_id = root.find('object').find('target_id')
        target_id = np.array(int(target_id.text) - 1, dtype="int64")

        label_list.append(target_id)

    jpeg_list = np.array(jpeg_list)
    data = np.expand_dims(jpeg_list, axis=1)
    mask = sailency(data=data, id=id)
    label = np.array(label_list)
    data_set = torch.utils.data.TensorDataset(torch.from_numpy(data).type(torch.FloatTensor),
                                              torch.from_numpy(mask).type(torch.FloatTensor),
                                              torch.from_numpy(label).type(torch.LongTensor))
    return data_set


def load_test(file_dir, picture_size=128):

    path_list = []
    jpeg_list = []
    label_list = []
    for root, dirs, files in os.walk(file_dir):
        files = sorted(files)
        for file in files:
            if os.path.splitext(file)[1] == '.tif':
                path_list.append(os.path.join(root, file))
    for jpeg_path in tqdm(path_list):
        jpeg = Image.open(jpeg_path)  # 可以读取单通道影像,读取3通道16位tif影像时报错(PIL.UnidentifiedImageError: cannot identify image file),支持4通道8位影像
        jpeg = np.array(jpeg)
        # jpeg = loadmat(jpeg_path)['Img']
        pic = crop_transform(picture_size)(torch.from_numpy(jpeg).div(jpeg.max()))
        jpeg_list.append(np.array(pic))
        # jpeg_list.append(np.array(pic))
        tree = ET.parse(jpeg_path.replace('tif', 'xml'))
        root = tree.getroot()
        target_id = root.find('object').find('target_id')
        target_id = np.array(int(target_id.text)-1, dtype="int64")

        label_list.append(target_id)

    jpeg_list = np.array(jpeg_list)
    data = np.expand_dims(jpeg_list, axis=1)
    label = np.array(label_list)
    data_set = torch.utils.data.TensorDataset(torch.from_numpy(data).type(torch.FloatTensor),
                                              torch.from_numpy(label).type(torch.LongTensor))
    return data_set