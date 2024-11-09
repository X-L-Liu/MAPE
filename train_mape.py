import argparse
import torch.optim as optim
import torchattacks
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from ppsa import PPSA
from utils import *
from model import *


def train_model():
    net.train()
    top1 = AverageMeter()
    with tqdm(total=len(train_loader), desc='Train-Progress') as pbar:
        for k, (image, label) in enumerate(train_loader):
            image, label = torch.Tensor(image).to(device), label.to(device)
            sel_cla = ppsa.determinate()
            for n in range(3):
                attacks[sel_cla][n].eps = random.uniform(4 / 255, 12 / 255)
                image[n::4] = attacks[sel_cla][n](image[n::4], label[n::4])
            logit = classifiers[sel_cla](image - net(image))
            loss = F.cross_entropy(logit, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1.update((logit.max(1)[1] == label).sum().item(), len(label))
            pbar.update(1)

    return top1.acc_rate


def test_model():
    net.eval()
    top1 = AverageMeter()
    with tqdm(total=len(test_loader), desc='Test--Progress') as pbar:
        for k, (image, label) in enumerate(test_loader):
            image, label = torch.Tensor(image).to(device), label.to(device)
            for n in range(3):
                image[n::4] = attacks[config.classifier_name[0]][n](image[n::4], label[n::4])
            logit = classifiers[config.classifier_name[0]](image - net(image))
            top1.update((logit.max(1)[1] == label).sum().item(), len(label))
            pbar.update(1)

    return top1.acc_rate


def main(Reload):
    global net_load_path
    for epoch in range(config.epochs):
        print(f'Reload: {Reload + 1}/{config.reload}  |  Epoch: {epoch + 1}/{config.epochs}')
        train_acc = train_model()
        test_acc = test_model()
        scheduler.step()
        if test_acc > config.best_acc:
            net_load_path = os.path.join(net_save_path, f'MAPE_CAU_Net_{test_acc*100:.2f}.pt')
            net_pre_path = net_load_path.replace(f'{test_acc*100:.2f}', f'{config.best_acc*100:.2f}')
            if os.path.exists(net_pre_path) and net_pre_path is not config.net_load_path:
                os.remove(net_pre_path)
            torch.save(net.state_dict(), net_load_path)
            config.best_acc = test_acc
        print(f'Train-Acc: {train_acc * 100:.2f}%  Test-Acc: {test_acc * 100:.2f}%  '
              f'Best-Test-Acc: {config.best_acc * 100:.2f}%')
        print('-' * 85)


def parse_tuple(input_string):
    items = input_string.split(',')
    return tuple(item.strip() for item in items)


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default=r'dataset', help='The dataset path')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--milestones', type=tuple, default=[50, 75, 100], help='Stage of adjusting learning rate')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=120, help='Total epochs')
    parser.add_argument('--reload', type=int, default=5, help='Reload model')
    parser.add_argument('--net_load_path', type=str, default=r'', help='The loaded CAU-Net')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'miniimagenet'])
    parser.add_argument('--device', type=int, default=0, help='GPU number')
    parser.add_argument('--best_acc', type=float, default=0, help='The best accuracy')
    parser.add_argument('--classifier_name', type=parse_tuple,
                        default=('ResNet34', 'GoogLeNet', 'MobileNetV2', 'VGG19'), help='The name of the classifier')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    config = parse_args()
    device = torch.device(f'cuda:{config.device}')
    if config.dataset_name == 'cifar10':
        num_classes = 10
        trainSet = datasets.CIFAR10(root=config.data_path, train=True, download=True, transform=transform_cifar_train)
        testSet = datasets.CIFAR10(root=config.data_path, train=False, download=True, transform=transform_cifar_test)
        net_save_path = os.path.join('CAU_Net', 'cifar10')
    elif config.dataset_name == 'cifar100':
        num_classes = 100
        trainSet = datasets.CIFAR100(root=config.data_path, train=True, download=True, transform=transform_cifar_train)
        testSet = datasets.CIFAR100(root=config.data_path, train=False, download=True, transform=transform_cifar_test)
        net_save_path = os.path.join('CAU_Net', 'cifar100')
    else:
        num_classes = 100
        trainSet = MiniImageNet(root=config.data_path, train=True, pixel='64', transform=transform_miniimagenet_train)
        testSet = MiniImageNet(root=config.data_path, train=False, pixel='64', transform=transform_miniimagenet_test)
        net_save_path = os.path.join('CAU_Net', 'miniimagenet')
    train_loader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(testSet, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    cla_path_config = ClaPathConfig()
    for cla_name in config.classifier_name:
        assert cla_name in cla_path_config.classifiers_path[config.dataset_name].keys(), \
            f'{cla_name} has not been pre-trained!'
    ppsa = PPSA(config.dataset_name, config.data_path, config.classifier_name, cla_path_config, device)

    classifiers = {}
    attacks = {}
    for cla_name in config.classifier_name:
        classifier = globals()[cla_name](num_classes)
        classifier.load_state_dict(torch.load(cla_path_config.classifiers_path[config.dataset_name][cla_name],
                                              map_location=device))
        classifier.to(device)
        classifier.eval()
        classifiers[cla_name] = classifier
        attacks[cla_name] = [
            torchattacks.FGSM(classifier),
            torchattacks.PGD(classifier),
            torchattacks.DIFGSM(classifier)
        ]

    if not os.path.exists(net_save_path):
        os.makedirs(net_save_path)
    net_load_path = config.net_load_path
    for reload in range(config.reload):
        net = CAU_Net()
        if net_load_path != '':
            net.load_state_dict(torch.load(net_load_path, map_location=device))
        net.to(device)
        print(f'Total params: {sum(p.numel() for p in net.parameters()) / 1000000.0:.2f}M')
        optimizer = optim.Adam(net.parameters(), lr=config.lr, weight_decay=0)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.milestones)
        main(reload)
