import argparse

import torch.optim.lr_scheduler
import torchattacks
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import *

from Classifier.model.densenet import *
from Classifier.model.dpn import *
from Classifier.model.googlenet import *
from Classifier.model.mobilenetv2 import *
from Classifier.model.pyramidnet import *
from Classifier.model.regnet import *
from Classifier.model.resnet import *
from Classifier.model.resnetv2 import *
from Classifier.model.resnext import *
from Classifier.model.senet import *
from Classifier.model.shufflenetv2 import *
from Classifier.model.vgg import *
from Classifier.model.wideresnet import *

def Clean(image, label):
    return image


def main(Amount):
    attacks = {
        'Clean': Clean,
        'AA': torchattacks.AutoAttack(classifier),
        'BIM': torchattacks.BIM(classifier, steps=20),
        'DIM': torchattacks.DIFGSM(classifier, steps=20),
        'FGSM': torchattacks.FGSM(classifier),
        'UPGD': torchattacks.UPGD(classifier, steps=20),
        'VNIM': torchattacks.VNIFGSM(classifier, steps=20),
    }

    input_size = train_loader.dataset.data.shape

    for att in attacks.keys():
        AE = {
            'train_AE': torch.zeros([len(trainSet), input_size[-1], input_size[-2], input_size[-2]]) if (config.data_type == 'all' or config.data_type == 'train') else None,
            'train_label': torch.zeros(len(trainSet)).to(torch.int64) if (config.data_type == 'all' or config.data_type == 'train') else None,
            'test_AE': torch.zeros([len(testSet), input_size[-1], input_size[-2], input_size[-2]]) if (config.data_type == 'all' or config.data_type == 'test') else None,
            'test_label': torch.zeros(len(testSet)).to(torch.int64) if (config.data_type == 'all' or config.data_type == 'test') else None
        }
        if config.data_type == 'all' or config.data_type == 'train':
            with tqdm(total=len(train_loader), desc=f'{config.classifier_name}-{Amount + 1}-{att}-Train', ncols=100,
                      unit=' batches') as pbar:
                for k, (image, label) in enumerate(train_loader):
                    label = label.to(device)
                    image = image.to(device)
                    AE['train_AE'][k * len(label):k * len(label) + len(label), ...] = attacks[att](image, label)
                    AE['train_label'][k * len(label):k * len(label) + len(label)] = label
                    pbar.update(1)
            AE['train_AE'] = np.round(np.array(AE['train_AE'].detach()) * 255).astype(np.uint8)
            AE['train_label'] = list(np.array(AE['train_label']))

        if config.data_type == 'all' or config.data_type == 'test':
            with tqdm(total=len(test_loader), desc=f'{config.classifier_name}-{Amount + 1}-{att}-Test', ncols=100,
                      unit=' batches') as pbar:
                for k, (image, label) in enumerate(test_loader):
                    label = label.to(device)
                    image = image.to(device)
                    AE['test_AE'][k * len(label):k * len(label) + len(label), ...] = attacks[att](image, label)
                    AE['test_label'][k * len(label):k * len(label) + len(label)] = label
                    pbar.update(1)
            AE['test_AE'] = np.round(np.array(AE['test_AE'].detach()) * 255).astype(np.uint8)
            AE['test_label'] = list(np.array(AE['test_label']))

        file_path = os.path.join(str(os.path.join('AE', config.dataset_name)),
                                 config.classifier_name + '-' + att + '-AE-' + str(Amount + 1) + '.pkl')
        with open(file_path, 'wb') as file:
            pickle.dump(AE, file)


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default=r'Dataset', help='The path to the training dataset')
    parser.add_argument('--dataset_name', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'miniimagenet'], help='The name of the datasets')
    parser.add_argument('--data_amount', type=int, default=5, help='The amount of the examples')
    parser.add_argument('--batch_size', type=int, default=100, help='The batch size')
    parser.add_argument('--device', type=str, default='cuda:1', help='GPU number')
    parser.add_argument('--data_type', type=str, default='all', choices=['all', 'train', 'test'], help='Purpose of sample')
    parser.add_argument('--classifier_name', type=str, default='ResNet34', help='The name of the classifier')
    parser.add_argument('--classifier_path', type=str, default='', help='The name of the classifier')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    config = parse_args()
    device = torch.device(config.device)
    if config.dataset_name == 'cifar10':
        num_classes = 10
        trainSet = datasets.CIFAR10(root=config.data_path, train=True, download=True, transform=transform_cifar_create_train)
        testSet = datasets.CIFAR10(root=config.data_path, train=False, download=True, transform=transform_cifar_test)
    elif config.dataset_name == 'cifar100':
        num_classes = 100
        trainSet = datasets.CIFAR100(root=config.data_path, train=True, download=True, transform=transform_cifar_create_train)
        testSet = datasets.CIFAR100(root=config.data_path, train=False, download=True, transform=transform_cifar_test)
    else:
        num_classes = 100
        trainSet = MiniImageNet(root=config.data_path, train=True, pixel='64', transform=transform_miniimagenet_create_train)
        testSet = MiniImageNet(root=config.data_path, train=False, pixel='64', transform=transform_miniimagenet_test)

    train_loader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    test_loader = DataLoader(testSet, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    classifier = globals()[config.classifier_name](num_classes)
    classifier.load_state_dict(torch.load(config.classifier_path, map_location=device))
    classifier.to(device)
    classifier.eval()

    for am in range(config.data_amount):
        main(am)
