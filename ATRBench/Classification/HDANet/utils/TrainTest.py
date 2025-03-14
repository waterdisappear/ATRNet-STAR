import torch
import os
import numpy as np
import sys
sys.path.append('..')
from torchvision import models
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
import random
import cv2
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import matplotlib.pyplot as plt
from captum.attr import DeepLift, GuidedGradCam
from MyModel.utils.DataTransform import augment_fn, move
from typing import Dict, Iterable, Callable
from tqdm import tqdm

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self.features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, input, output):
            self.features[layer_id] = output
        return fn

    def forward(self, x):
        out = self.model(x)
        return out, self.features


def loss_fn(x, y):
    # criterion = nn.CosineSimilarity(dim=1)
    # -(criterion(p1, z2).mean()
    y = y.detach()
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    # print(criterion(x, y).shape)
    # return 2 - 2 * (x * y).sum(dim=-1)
    # return 1/2 - (criterion(x, y).mean())*1/2
    return 1/2 - (x * y).sum(dim=-1).mean()/2


class CapsuleLoss(nn.Module):
    """Combine margin loss & reconstruction loss of capsule network."""

    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super(CapsuleLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda

    def forward(self, logits, labels):
        # Shape of left / right / labels: (batch_size, num_classes)
        # left = (self.upper - logits).relu() ** 2  # True negative
        # right = (logits - self.lower).relu() ** 2  # False positive
        # labels = torch.zeros(logits.shape).cuda().scatter_(1, labels.unsqueeze(1), 1)
        # margin_loss = torch.sum(labels * left) + self.lmda * torch.sum((1 - labels) * right)

        left = (self.upper - logits).relu() ** 2  # True negative
        right = (logits - self.lower).relu() ** 2  # False positive
        # print(logits.shape)
        # print(labels.shape)
        labels = torch.zeros(logits.shape).cuda().scatter_(1, labels.unsqueeze(1), 1)
        margin_loss = (labels * left).sum(-1).mean() + self.lmda * ((1 - labels) * right).sum(-1).mean()

        # Reconstruction loss

        # Combine two losses
        return margin_loss


def model_train(model, data_loader, opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    train_acc = 0
    cr1 = CapsuleLoss()
    # cr1 = nn.CrossEntropyLoss()
    cr2 = loss_fn
    cr3 = nn.BCELoss()
    cr4 = nn.L1Loss()
    # opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=1.5e-6)

    train_loss, train_loss1, train_loss2, train_loss3, train_loss4 = 0, 0, 0, 0, 0
    for i, data in enumerate(data_loader):
        x, mask, y = data
        [x_one, mask_one], [x_two, mask_two] = move(x, mask), move(x, mask)
        image_one, image_two = augment_fn(x_one), augment_fn(x_two)

        median_extractor = FeatureExtractor(model, layers=['MLP', 'Project', 'conv10', 'mask'])

        out_one, median_one = median_extractor(image_one)
        predic_one, project_one, saliency_one = median_one['MLP'], median_one['Project'], median_one['conv10']
        med_mask1 = median_one['mask']
        out_two, median_two = median_extractor(image_two)
        predic_two, project_two, saliency_two = median_two['MLP'], median_two['Project'], median_two['conv10']
        med_mask2 = median_two['mask']

        pred = out_one.max(1, keepdim=True)[1]  # get the index of the max log-probability
        train_acc += pred.eq(y.to(device).view_as(pred)).sum().item()
        pred = out_two.max(1, keepdim=True)[1]  # get the index of the max log-probability
        train_acc += pred.eq(y.to(device).view_as(pred)).sum().item()

        loss1 = cr1(out_one, y.to(device)) + cr1(out_two, y.to(device))
        loss2 = cr2(predic_one, project_two) + cr2(predic_two, project_one)
        loss3 = cr3(saliency_one, mask_one.to(device)) + cr3(saliency_two, mask_two.to(device))
        loss4 = cr4(med_mask1, torch.zeros(med_mask1.shape).to(device)) + cr4(med_mask2, torch.zeros(med_mask1.shape).to(device))

        # loss4 = cr4(a1, torch.zeros(a1.shape).to(device)) + cr4(a2, torch.zeros(a2.shape).to(device)) + \
        #         cr4(a3, torch.zeros(a3.shape).to(device)) + cr4(a4, torch.zeros(a4.shape).to(device)) + \
        #         cr4(b1, torch.zeros(b1.shape).to(device)) + cr4(b2, torch.zeros(b2.shape).to(device)) + \
        #         cr4(b3, torch.zeros(b3.shape).to(device)) + cr4(b4, torch.zeros(b4.shape).to(device))
        # plt.subplot(311)
        # plt.imshow(F.sigmoid(saliency_one[0]).squeeze().cpu().detach().numpy())
        # plt.subplot(312)
        # plt.imshow(med_one[0].squeeze().cpu().detach().numpy())
        # plt.subplot(313)
        # plt.imshow(mask_one[0].squeeze().cpu().detach().numpy())
        # plt.show()
        # loss = loss1 + 1e-1*loss2 + 1e-2*loss3 + 1e-3*loss4
        loss = loss1 + loss2 + 1e-1*loss3 + 1e-2*loss4
        train_loss, train_loss1, train_loss2, train_loss3, train_loss4 = \
            train_loss+loss.item(), train_loss1+loss1.item(), train_loss2+loss2.item(), train_loss3+loss3.item(), train_loss4+loss4.item()
        opt.zero_grad()
        loss.backward()
        opt.step()


    # print("Train loss is:{:.8f}, loss1 is:{:.8f}, loss2 is:{:.8f}, loss3 is:{:.8f}, loss4 is:{:.8f}"
    #       .format(train_loss / len(data_loader), train_loss1 / len(data_loader), train_loss2 / len(data_loader),
    #               train_loss3 / len(data_loader), train_loss4 / len(data_loader)))
    # print("Train accuracy is:{:.2f} % ".format(train_acc / 2/ len(data_loader.dataset) * 100.))
    return train_loss/len(data_loader)

def model_test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_loss = 0
    correct = 0
    pred_all = np.array([[]]).reshape((0, 1))
    real_all = np.array([[]]).reshape((0, 1))
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    # print("Test Accuracy is:{:.2f} %: ".format(100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def model_val(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_loss = 0
    correct = 0
    pred_all = np.array([[]]).reshape((0, 1))
    real_all = np.array([[]]).reshape((0, 1))
    model.eval()
    with torch.no_grad():
        for data, mask, target in test_loader:
            target = target.to(device)
            # data = augment_fn(data)
            data = data.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    # print("Test Accuracy is:{:.2f} %: ".format(100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


from MyModel.utils.TestTransform import RandomApply, TEST_AddPepperNoise,Cuda, TEST_AddGussain, TEST_Ersion

def model_test_eoc3(model, test_loader, SNR):
    EOC3_fn = T.Compose([
        RandomApply(TEST_AddGussain(SNR), p=1),
        Cuda()],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_loss = 0
    correct = 0
    pred_all = np.array([[]]).reshape((0, 1))
    real_all = np.array([[]]).reshape((0, 1))
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            target = target.to(device)
            data = EOC3_fn(data)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    # print("Test Accuracy is:{:.2f} %: ".format(100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def model_test_eoc4(model, test_loader, SNR):
    EOC4_fn = T.Compose([
        RandomApply(TEST_AddPepperNoise(SNR), p=1),
        Cuda()],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_loss = 0
    correct = 0
    pred_all = np.array([[]]).reshape((0, 1))
    real_all = np.array([[]]).reshape((0, 1))
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            target = target.to(device)
            data = EOC4_fn(data)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    # print("Test Accuracy is:{:.2f} %: ".format(100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def model_test_eoc5(model, test_loader, SNR):
    EOC5_fn = T.Compose([
        RandomApply(TEST_Ersion(SNR), p=1),
        Cuda()],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_loss = 0
    correct = 0
    pred_all = np.array([[]]).reshape((0, 1))
    real_all = np.array([[]]).reshape((0, 1))
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            target = target.to(device)
            data = EOC5_fn(data)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    # print("Test Accuracy is:{:.2f} %: ".format(100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)