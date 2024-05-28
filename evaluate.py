import argparse
import threading
import torch.optim as optim
import torch.nn.functional as F
import torchattacks

from SAU_Net.sau_net import *
from torchvision import datasets
from torch.utils.data import DataLoader
from utils.utils import *
from tqdm import tqdm
from utils.classifiers_path import *
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


def test_model():
    clean_top1 = AverageMeter()
    clean_defense_top1 = AverageMeter()
    adv_top1 = [AverageMeter() for _ in config.attack_methods]
    adv_defense_top1 = [AverageMeter() for _ in config.attack_methods]
    with tqdm(total=len(test_loader), desc='Evaluation--Progress') as pbar:
        for k, (image, label) in enumerate(test_loader):
            image = torch.Tensor(image).to(device)
            label = torch.Tensor(label).to(device)

            clean_logit = classifier(image)
            clean_top1.update((clean_logit.max(1)[1] == label).sum().item(), len(label))

            image_pert = sau_net(image)
            image_pert = image_pert - torch.mean(image_pert,
                                                 dim=[1, 2, 3]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            clean_defense_logit = classifier(image - image_pert)
            clean_defense_top1.update((clean_defense_logit.max(1)[1] == label).sum().item(), len(label))

            for i in range(len(config.attack_methods)):
                image_adv = globals()[config.attack_methods[i]](image, label)

                adv_logit = classifier(image_adv)
                adv_top1[i].update((adv_logit.max(1)[1] == label).sum().item(), len(label))

                image_pert = sau_net(image_adv)
                image_pert = image_pert - torch.mean(image_pert,
                                                     dim=[1, 2, 3]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                adv_defense_logit = classifier(image_adv - image_pert)
                adv_defense_top1[i].update((adv_defense_logit.max(1)[1] == label).sum().item(), len(label))
            pbar.update(1)

    print(f'Clean Accuracy(%): {clean_top1.acc_rate * 100:.2f}%, '
          f'Clean Accuracy(%) After MAPE: {clean_defense_top1.acc_rate * 100:.2f}%;')
    for i in range(len(config.attack_methods)):
        print(f'{config.attack_methods[i]} Accuracy(%): {adv_top1[i].acc_rate * 100:.2f}%, '
              f'{config.attack_methods[i]} Accuracy(%) After MAPE: {adv_defense_top1[i].acc_rate * 100:.2f}%;')


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default=r'Dataset', help='The path to the training dataset')
    parser.add_argument('--attack_methods', type=str, default='fgsm_pgd', help='Attack methods')
    parser.add_argument('--sau_net_load_path', type=str, default=r'SAU_Net/cifar10/SAU_Net_MAPE_0.9594.pt', help='The loaded SAU-Net')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'miniimagenet'])
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU number')
    parser.add_argument('--classifier_name', type=str, default='MobileNetV2', help='The name of the classifier')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    config = parse_args()
    continue_event = threading.Event()
    device = torch.device(config.device)
    config.attack_methods = config.attack_methods.split('_')
    if config.dataset_name == 'cifar10':
        num_classes = 10
        testSet = datasets.CIFAR10(root=config.data_path, train=False, download=True, transform=transform_cifar_test)
        classifier_load_path = cifar10_classifier_load_path
    elif config.dataset_name == 'cifar100':
        num_classes = 100
        testSet = datasets.CIFAR100(root=config.data_path, train=False, download=True, transform=transform_cifar_test)
        classifier_load_path = cifar100_classifier_load_path
    else:
        num_classes = 100
        testSet = MiniImageNet(root=config.data_path, train=False, pixel='64', transform=transform_miniimagenet_test)
        classifier_load_path = miniimagenet_classifier_load_path
    classifier_name = config.classifier_name
    test_loader = DataLoader(testSet, batch_size=100, shuffle=False, pin_memory=True, num_workers=1)

    classifier = globals()[classifier_name](num_classes)
    classifier.load_state_dict(torch.load(classifier_load_path[classifier_name], map_location=device))
    classifier.to(device)
    classifier.eval()

    sau_net = SAU_Net()
    sau_net.load_state_dict(torch.load(config.sau_net_load_path, map_location=config.device))
    sau_net.to(device)
    sau_net.eval()
    print('Total params: %.2fM' % (sum(p.numel() for p in sau_net.parameters()) / 1000000.0))

    if 'fgsm' in config.attack_methods:
        fgsm = torchattacks.FGSM(classifier)
    if 'aa' in config.attack_methods:
        aa = torchattacks.AutoAttack(classifier)
    if 'bim' in config.attack_methods:
        bim = torchattacks.BIM(classifier, steps=20)
    if 'dim' in config.attack_methods:
        dim = torchattacks.DIFGSM(classifier, steps=20)
    if 'pgd' in config.attack_methods:
        pgd = torchattacks.PGD(classifier, steps=20)
    if 'upgd' in config.attack_methods:
        upgd = torchattacks.UPGD(classifier, steps=20)
    if 'vnim' in config.attack_methods:
        vnim = torchattacks.VNIFGSM(classifier, steps=20)

    test_model()

