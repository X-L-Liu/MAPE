import argparse
import time
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from utils import *
from model import *


def train_epoch():
    classifier.train()
    top1 = AverageMeter()
    for k, (image, label) in enumerate(train_loader):
        image, label = torch.Tensor(image).to(device), label.to(device)
        logit = classifier(image)
        loss = F.cross_entropy(logit, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        top1.update((logit.max(1)[1] == label).sum().item(), len(label))

    return top1.acc_rate


def test_epoch():
    classifier.eval()
    with torch.no_grad():
        top1 = AverageMeter()
        for _, (image, label) in enumerate(test_loader):
            image, label = torch.Tensor(image).to(device), label.to(device)
            logit = classifier(image)
            top1.update((logit.max(1)[1] == label).sum().item(), len(label))

    return top1.acc_rate


def main(Reload):
    global classifier_load_path
    for epoch in range(config.epochs):
        start = time.time()
        train_acc = train_epoch()
        test_acc = test_epoch()
        scheduler.step()
        if test_acc > config.best_acc:
            classifier_load_path = os.path.join(classifier_save_path,
                                                config.classifier_name + f'_{test_acc*100:.2f}.pt')
            classifier_pre_path = classifier_load_path.replace(f'{test_acc*100:.2f}', f'{config.best_acc*100:.2f}')
            if os.path.exists(classifier_pre_path) and classifier_pre_path is not config.classifier_load_path:
                os.remove(classifier_pre_path)
            torch.save(classifier.state_dict(), classifier_load_path)
            cla_path_config.modify_config(config.dataset_name, config.classifier_name, classifier_load_path)
            config.best_acc = test_acc
        print(f'Reload: {Reload + 1}/{config.reload}  Epoch: {epoch + 1}/{config.epochs}  '
              f'Train-Top1: {train_acc * 100:.2f}%  Test-Top1: {test_acc * 100:.2f}%  '
              f'Best-Top1: {config.best_acc * 100:.2f}%  Time: {time.time() - start:.0f}s')


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default=r'dataset', help='The dataset path')
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum of the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0, help='The weight decay')
    parser.add_argument('--milestones', type=tuple, default=(50, 90, 120), help='Stage of adjusting learning rate')
    parser.add_argument('--epochs', type=int, default=150, help='Total epochs')
    parser.add_argument('--reload', type=int, default=5, help='Reload model')
    parser.add_argument('--dataset_name', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'miniimagenet'], help='The name of the datasets')
    parser.add_argument('--device', type=int, default=0, help='GPU number')
    parser.add_argument('--classifier_load_path', type=str, default='', help='The path to the best classifier')
    parser.add_argument('--best_acc', type=float, default=0, help='The best accuracy')
    parser.add_argument('--classifier_name', type=str, default='ResNet34', help='The name of the classifier')
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
        classifier_save_path = r'classifier/cifar10'
    elif config.dataset_name == 'cifar100':
        num_classes = 100
        trainSet = datasets.CIFAR100(root=config.data_path, train=True, download=True, transform=transform_cifar_train)
        testSet = datasets.CIFAR100(root=config.data_path, train=False, download=True, transform=transform_cifar_test)
        classifier_save_path = r'classifier/cifar100'
    else:
        num_classes = 100
        trainSet = MiniImageNet(root=config.data_path, train=True, pixel='64', transform=transform_miniimagenet_train)
        testSet = MiniImageNet(root=config.data_path, train=False, pixel='64', transform=transform_miniimagenet_test)
        classifier_save_path = r'classifier/miniimagenet'
    train_loader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(testSet, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    if not os.path.exists(classifier_save_path):
        os.makedirs(classifier_save_path)
    classifier_load_path = config.classifier_load_path

    cla_path_config = ClaPathConfig()

    for reload in range(config.reload):
        print('>' * 100)
        classifier = globals()[config.classifier_name](num_classes)
        if classifier_load_path != '':
            classifier.load_state_dict(torch.load(classifier_load_path, map_location=device))
        classifier.to(device)
        print(f'Total params: {sum(p.numel() for p in classifier.parameters()) / 1000000.0:.2f}M')
        optimizer = optim.SGD(classifier.parameters(), config.lr, config.momentum, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.milestones)
        main(reload)
