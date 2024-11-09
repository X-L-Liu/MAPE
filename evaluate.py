import argparse
import torchattacks
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from model import *
from utils import *


def eval_model(Att):
    top1 = AverageMeter()
    with tqdm(total=len(test_loader), desc='Evaluation--Progress') as pbar:
        for k, (image, label) in enumerate(test_loader):
            image, label = torch.Tensor(image).to(device), label.to(device)
            image = attacks[Att](image, label)
            logit = classifier(image - cau_net(image))
            top1.update((logit.max(1)[1] == label).sum().item(), len(label))
            pbar.update(1)

    return top1.acc_rate


def parse_tuple(input_string):
    items = input_string.split(',')
    return tuple(item.strip() for item in items)


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default=r'dataset', help='The path to the training dataset')
    parser.add_argument('--attack_methods', type=parse_tuple, default=('FGSM', 'PGD_20'), help='Attack methods')
    parser.add_argument('--net_load_path', type=str, default=r'', help='The loaded CAU-Net')
    parser.add_argument('--dataset_name', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'miniimagenet'])
    parser.add_argument('--device', type=int, default=0, help='GPU number')
    parser.add_argument('--classifier_name', type=str, default='ResNet34', help='The name of the classifier')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    config = parse_args()
    device = torch.device(config.device)
    if config.dataset_name == 'cifar10':
        num_classes = 10
        testSet = datasets.CIFAR10(root=config.data_path, train=False, download=True, transform=transform_cifar_test)
    elif config.dataset_name == 'cifar100':
        num_classes = 100
        testSet = datasets.CIFAR100(root=config.data_path, train=False, download=True, transform=transform_cifar_test)
    else:
        num_classes = 100
        testSet = MiniImageNet(root=config.data_path, train=False, pixel='64', transform=transform_miniimagenet_test)
    test_loader = DataLoader(testSet, batch_size=100, shuffle=False, pin_memory=True, num_workers=1)

    cla_path_config = ClaPathConfig()
    assert config.classifier_name in cla_path_config.classifiers_path[config.dataset_name].keys(), \
        f'{config.classifier_name} has not been pre-trained!'

    classifier = globals()[config.classifier_name](num_classes)
    classifier.load_state_dict(torch.load(cla_path_config.classifiers_path[config.dataset_name][config.classifier_name],
                                          map_location=device))
    classifier.to(device)
    classifier.eval()
    attacks = {}
    for att in config.attack_methods:
        att_args = att.split('_')
        if len(att_args) == 1:
            attacks[att] = getattr(torchattacks, att_args[0])(classifier)
        else:
            attacks[att] = getattr(torchattacks, att_args[0])(classifier, steps=int(att_args[1]))

    cau_net = CAU_Net()
    cau_net.load_state_dict(torch.load(config.net_load_path, map_location=device))
    cau_net.to(device)
    cau_net.eval()
    print(f'Target Model: {config.classifier_name}')
    for att in config.attack_methods:
        acc = eval_model(att)
        print(f'Defend {att}: {acc*100:.2f}%')
