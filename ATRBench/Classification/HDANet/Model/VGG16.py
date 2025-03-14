import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
import re
from tqdm import tqdm
import argparse
import collections
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, RandomCrop
from torchvision import models
import torch
import os
import numpy as np
from PIL import Image
from torch import nn
import random
import cv2
from scipy.io import loadmat
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, RandomCrop
import re
from MyModel.utils.DataTransform import RandomApply, AddGaussianNoise, AddPepperNoise


def model_train(model, data_loader, opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    train_acc = 0
    cr = nn.CrossEntropyLoss()
    train_loss = 0
    for i, data in enumerate(data_loader):
        x, y = data
        out = model(x.cuda())
        pred = out.max(1, keepdim=True)[1]  # get the index of the max log-probability
        train_acc += pred.eq(y.to(device).view_as(pred)).sum().item()
        loss = cr(out, y.to(device))
        train_loss += loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("Train loss is:{:.8f}".format(train_loss / len(data_loader)))
    print("Train accuracy is:{:.2f} % ".format(train_acc / len(data_loader.dataset) * 100.))


def model_test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # test_loss = 0
    # pred_all = np.array([[]]).reshape((0, 1))
    # real_all = np.array([[]]).reshape((0, 1))
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    # print("Test Accuracy is:{:.2f} %: ".format(100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


class SAR_VGG16(torch.nn.Module):

    def __init__(self, feature_extract=True, num_classes=10):
        super(SAR_VGG16, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # torch.nn.MaxPool2d(kernel_size=2, stride=1)
        # batsize,channel,height,length 20*3*128*128

        # 3 * 128 * 128
        # 导入VGG16模型
        model = models.vgg16(pretrained=True)
        # 加载features部分
        self.features = model.features
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


class TensorsDataset(torch.utils.data.Dataset):

    '''
    A simple loading dataset - loads the tensor that are passed in input. This is the same as
    torch.utils.data.TensorDataset except that you can add transformations to your data and target tensor.
    Target tensor can also be None, in which case it is not returned.
    '''

    def __init__(self, data_tensor, target_tensor=None, transforms=None, target_transforms=None):
        if target_tensor is not None:
            assert len(data_tensor) == target_tensor.shape[0]
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

        if transforms is None:
            transforms = []
        if target_transforms is None:
            target_transforms = []

        if not isinstance(transforms, list):
            transforms = [transforms]
        if not isinstance(target_transforms, list):
            target_transforms = [target_transforms]

        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):
        jpeg = loadmat(self.data_tensor[index])['Img']
        data_tensor = torch.from_numpy(jpeg)
        for transform in self.transforms:
            data_tensor = transform(data_tensor)
            data_tensor = data_tensor.div(data_tensor.max()).unsqueeze(0).float()
        if self.target_tensor is None:
            return data_tensor

        target_tensor = self.target_tensor[index]
        for transform in self.target_transforms:
            target_tensor = transform(target_tensor)

        return data_tensor, target_tensor

    def __len__(self):
        return len(self.data_tensor)


def load_data(file_dir, transform):
    data_name = re.split('[/\\\]', file_dir)[-2]
    if data_name == 'SOC':
        label_name = {'BMP2': 0, 'BTR70': 1, 'T72': 2, 'BTR_60': 3, '2S1': 4, 'BRDM_2': 5, 'D7': 6, 'T62': 7,
                      'ZIL131': 8, 'ZSU_23_4': 9}
    elif data_name == 'EOC-1':
        label_name = {'2S1': 0, 'BRDM_2': 1, 'T72': 2, 'ZSU_23_4': 3}
    elif data_name == 'EOC-2':
        label_name = {'BRDM_2': 0, 'T72': 1, 'ZSU_23_4': 2}

    path_list = []
    jpeg_list = []
    label_list = []
    for root, dirs, files in os.walk(file_dir):
        files = sorted(files)
        for file in files:
            if os.path.splitext(file)[1] == '.mat':
                path_list.append(os.path.join(root, file))

    for jpeg_path in path_list:
        jpeg_list.append(jpeg_path)
        label_list.append(label_name[re.split('[/\\\]', jpeg_path)[6]])

    data = jpeg_list
    label = np.array(label_list)
    data_set = TensorsDataset(data, torch.from_numpy(label).type(torch.LongTensor), transforms=transform)
    return data_set, label_name


def parameter_setting():
    # argparse settings
    parser = argparse.ArgumentParser(description='Origin Input')
    parser.add_argument('--data_path', type=str, default="../../Data/MSTAR/SOC/",
                        help='where data is stored')
    parser.add_argument('--GPU_ids', type=int, default=0,
                        help='GPU ids')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--fold', type=int, default=10,
                        help='K-fold')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    arg = parameter_setting()
    torch.cuda.set_device(arg.GPU_ids)
    # torch.manual_seed(arg.seed)
    # torch.cuda.manual_seed(arg.seed)
    history = collections.defaultdict(list)  # 记录每一折的各种指标

    train_transforms = Compose([CenterCrop(128)])
    test_transforms = Compose([CenterCrop(128)])
    train_set, label_name = load_data(arg.data_path + 'TRAIN', train_transforms)
    test_set, _ = load_data(arg.data_path + 'TEST', test_transforms)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=arg.batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size=arg.batch_size, shuffle=False)

    model = SAR_VGG16(num_classes=len(label_name))
    # opt = torch.optim.SGD(model.parameters(), lr=arg.lr)
    opt = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=0.004)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
    best_test_accuracy = 0
    for epoch in range(1, arg.epochs + 1):
        print("Epoch is:{}: ".format(epoch))
        model_train(model=model, data_loader=train_loader, opt=opt)
        accuracy = model_test(model, test_loader)
        scheduler.step()
        print("test Accuracy is:{:.2f} %: ".format(accuracy))

    torch.save({'model': model.state_dict()}, 'VGG16.pth')
