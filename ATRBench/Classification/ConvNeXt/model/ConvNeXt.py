import torch
from torch import nn
from torchvision import models


class A_ConvNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=10):
        super(A_ConvNet, self).__init__()
        # ConvTranspose2d output = (input-1)stride+outputpadding -2padding+kernelsize
        self.conv1 = nn.Conv2d(in_ch, 16, 5, 1, 0)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 0)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 6, 1, 0)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(64, 128, 5, 1, 0)
        self.drop = nn.Dropout(0.5)
        self.conv5 = nn.Conv2d(128, num_classes, 3, 1, 0)
        self.Relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.Relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.Relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.Relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.Relu(x)
        x = self.drop(x)
        out = self.conv5(x)
        return out.squeeze()


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Linear(in_planes, in_planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc1(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.fc1(self.max_pool(x).squeeze(-1).squeeze(-1))
        out = avg_out + max_out
        return self.sigmoid(out).unsqueeze(2).unsqueeze(2)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


def atten(x, con, ca ,sa):
    x = con(x)
    x = x*ca(x)
    y = x*sa(x)
    return y


class AMCNN(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(AMCNN, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # torch.nn.MaxPool2d(kernel_size=2, stride=1)
        # batsize,channel,height,length 20*3*128*128
        self.num = num_classes
        # 3 * 128 * 128
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, (3, 3)), nn.BatchNorm2d(16), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, (7, 7)), nn.BatchNorm2d(32), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, (5, 5)), nn.BatchNorm2d(64), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, (5, 5)), nn.BatchNorm2d(128), nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, (5, 5)), nn.BatchNorm2d(256), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(256, 128, (6, 6)), nn.BatchNorm2d(128), nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(128, 64, (5, 5)), nn.BatchNorm2d(64), nn.ReLU())
        self.conv8 = nn.Conv2d(64, self.num, (3, 3))

        self.CA1 = ChannelAttention(16)
        self.SA1 = SpatialAttention()
        self.CA2 = ChannelAttention(32)
        self.SA2 = SpatialAttention()
        self.CA3 = ChannelAttention(64)
        self.SA3 = SpatialAttention()
        self.CA4 = ChannelAttention(128)
        self.SA4 = SpatialAttention()

        self.CA5 = ChannelAttention(256)
        self.SA5 = SpatialAttention()
        self.CA6 = ChannelAttention(128)
        self.SA6 = SpatialAttention()
        self.CA7 = ChannelAttention(64)
        self.SA7 = SpatialAttention()

        self.maxpool = nn.MaxPool2d((2,2))


    def forward(self, x):
        x1 = atten(x, self.conv1, self.CA1, self.SA1)
        x2 = atten(x1, self.conv2, self.CA2, self.SA2)
        x3 = atten(x2, self.conv3, self.CA3, self.SA3)
        x4 = atten(x3, self.conv4, self.CA4, self.SA4)
        x4 = self.maxpool(x4)

        x5 = atten(x4, self.conv5, self.CA5, self.SA5)
        x5 = self.maxpool(x5)
        x6 = atten(x5, self.conv6, self.CA6, self.SA6)
        x6 = self.maxpool(x6)
        x7 = atten(x6, self.conv7, self.CA7, self.SA7)

        out = self.conv8(x7).squeeze(-1).squeeze(-1)
        return out


class BNACNN(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(BNACNN, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # torch.nn.MaxPool2d(kernel_size=2, stride=1)
        # batsize,channel,height,length 20*3*128*128
        self.num = num_classes
        # 3 * 128 * 128
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, (3, 3)), nn.BatchNorm2d(16), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, (7, 7)), nn.BatchNorm2d(32), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, (5, 5)), nn.BatchNorm2d(64), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, (5, 5)), nn.BatchNorm2d(128), nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, (5, 5)), nn.BatchNorm2d(256), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(256, 128, (6, 6)), nn.BatchNorm2d(128), nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(128, 64, (5, 5)), nn.BatchNorm2d(64), nn.ReLU())
        self.conv8 = nn.Conv2d(64, self.num, (3, 3))

        self.maxpool = nn.MaxPool2d((2,2))


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x4 = self.maxpool(x4)

        x5 = self.conv5(x4)
        x5 = self.maxpool(x5)
        x6 = self.conv6(x5)
        x6 = self.maxpool(x6)
        x7 = self.conv7(x6)

        out = self.conv8(x7).squeeze(-1).squeeze(-1)
        return out


class MVGGNet(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(MVGGNet, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # torch.nn.MaxPool2d(kernel_size=2, stride=1)
        model = models.vgg16(pretrained=True)
        self.features = model.features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 256),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x = torch.cat([x, x, x], 1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


class ResNet_34(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet_34, self).__init__()

        model = models.resnet34(pretrained=True)
        # print(model)
        model.fc = nn.Linear(512, num_classes)
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out


class ResNet_50(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet_50, self).__init__()

        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, num_classes)
        self.model = model

    def forward(self, x):
        x = torch.cat([x, x, x], 1)
        out = self.model(x)
        return out


class convnext_1(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(convnext_1, self).__init__()

        model = models.convnext_base(pretrained=True)
        # print(model)
        model.classifier[2] = nn.Linear(1024, num_classes)
        self.model = model

    def forward(self, x):
        # x = torch.cat([x, x, x], 1)
        out = self.model(x)
        return out


class efficientnet_b0(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(efficientnet_b0, self).__init__()

        model = models.efficientnet_b0()
        print(model)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        self.model = model


    def forward(self, x):
        x = torch.cat([x, x, x], 1)
        out = self.model(x)
        return out


class efficientnet_b1(torch.nn.Module):
    def __init__(self, feature_extract=True, num_classes=3):
        super(efficientnet_b1, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        model = models.efficientnet_b1()
        print(model)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        self.model = model


    def forward(self, x):
        x = torch.cat([x, x, x], 1)
        out = self.model(x)
        return out
