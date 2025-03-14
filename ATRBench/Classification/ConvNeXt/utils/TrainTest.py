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


def model_train(model, data_loader, opt, sch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    train_acc = 0
    cr1 = nn.CrossEntropyLoss()
    correct = 0

    train_loss, train_loss1, train_loss2, train_loss3, train_loss4 = 0, 0, 0, 0, 0
    for i, data in enumerate(data_loader):
        x, y = data
        y = y.cuda()
        output = model(x.cuda())
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(y.view_as(pred)).sum().item()

        loss = cr1(output, y.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    print("Train Accuracy is:{:.2f} %: ".format(100. * correct / len(data_loader.dataset)))
    return

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
        for data, target in test_loader:
            target = target.to(device)
            # data = augment_fn(data)
            data = data.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    # print("Test Accuracy is:{:.2f} %: ".format(100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

