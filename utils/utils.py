import json
import os
import pickle
from torchvision import transforms
from typing import Callable, Optional
from torchvision.datasets import VisionDataset
from .autoaugment import *
from .cutout import Cutout

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


class ClaPathConfig(object):
    def __init__(self):
        if os.path.exists('classifiers_path.json'):
            with open('classifiers_path.json', 'r', encoding='utf-8') as file:
                self.classifiers_path = json.load(file)
        else:
            self.classifiers_path = {
                'cifar10': {},
                'cifar100': {},
                'miniimagenet': {}
            }
            self.save_config()

    def modify_config(self, dataset_name, classifier_name, classifier_path):
        self.classifiers_path[dataset_name][classifier_name] = classifier_path
        self.save_config()

    def save_config(self):
        with open('classifiers_path.json', 'w', encoding='utf-8') as file:
            json.dump(self.classifiers_path, file, ensure_ascii=False, indent=4)
