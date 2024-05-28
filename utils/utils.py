import os
import pickle

import torch

from torch.utils.data import Dataset
from torchvision import transforms
from typing import Callable, Optional

from utils.autoaugment import *
from utils.cutout import Cutout
from torchvision.datasets import VisionDataset

from PIL import Image

transform_cifar_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16),
])

transform_cifar_test = transforms.Compose([
    transforms.ToTensor(),
])

transform_cifar_create_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_mnist_train = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor()
])

transform_mnist_test = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.ToTensor()
])

transform_miniimagenet_train = transforms.Compose([
    transforms.RandomCrop(64, padding=8, fill=128),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=24),
])

transform_miniimagenet_test = transforms.Compose([
    transforms.ToTensor()
])

transform_miniimagenet_create_train = transforms.Compose([
    transforms.RandomCrop(64, padding=8, fill=128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.acc_num = 0
        self.total_num = 0
        self.acc_rate = 0

    def update(self, acc_num, total_num):
        self.acc_num += acc_num
        self.total_num += total_num
        self.acc_rate = self.acc_num / self.total_num


class AEDataset(Dataset):
    def __init__(self, file_paths: list, train=True, target_transform=None):

        self.target_transform = target_transform

        for file_path in file_paths:
            assert os.path.exists(file_path)

        data = pickle.load(file=open(file_paths[0], 'rb'))
        if train:
            self.sample, self.label = data['train_AE'], data['train_label']
        else:
            self.sample, self.label = data['test_AE'], data['test_label']

        self.sample = np.expand_dims(self.sample, axis=1)
        if len(file_paths) > 1:
            for file_path in file_paths[1:]:
                data = pickle.load(file=open(file_path, 'rb'))
                if train:
                    self.sample = np.concatenate((self.sample, np.expand_dims(data['train_AE'], axis=1)), axis=1)
                else:
                    self.sample = np.concatenate((self.sample, np.expand_dims(data['test_AE'], axis=1)), axis=1)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        if self.target_transform is None:
            return torch.Tensor(self.sample[item]/255), self.label[item]
        else:
            return torch.Tensor(self.sample[item] / 255), self.target_transform(self.label[item])


class MiniImageNet(VisionDataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 pixel: str = '64',
                 transform: Optional[Callable] = None
                 ):
        super().__init__(root, transform=transform)
        data = pickle.load(file=open(os.path.join(self.root, 'mini-imagenet-' + pixel + '.pkl'), 'rb'))
        if train:
            self.sample, self.label = data['train_sample'], data['train_label']
        else:
            self.sample, self.label = data['test_sample'], data['test_label']
        self.sample = self.sample.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index: int):
        img, target = self.sample[index], self.label[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self) -> int:
        return len(self.label)
