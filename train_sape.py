import argparse
import threading 
import torch.optim as optim
import torch.nn.functional as F
from SAU_Net.sau_net import *
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


def create_train_loader():
    train_AE_path = [os.path.join(data_path, classifier_name + '-' + att + '-AE-' +
                                  str(random.randint(1, config.data_amount)) + '.pkl')
                     for att in config.attack_methods]
    trainSet = AEDataset(train_AE_path, train=True)
    return DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=1)


def control_train_loader():
    global train_loader_1, train_loader_2, train_loader_1_status, train_loader_2_status, load_timer
    if train_loader_1_status == 'reclaimable':
        train_loader_1 = create_train_loader()
        train_loader_1_status = 'ready'
        continue_event.set()
    if train_loader_2_status == 'reclaimable':
        train_loader_2 = create_train_loader()
        train_loader_2_status = 'ready'
        continue_event.set()
    load_timer = threading.Timer(5, control_train_loader)
    load_timer.start()


def train_model(SAU_Net_, Optimizer):
    global train_loader_1, train_loader_2, train_loader_1_status, train_loader_2_status
    SAU_Net_.train()
    top1 = AverageMeter()
    if train_loader_1_status == 'ready':
        train_loader = train_loader_1
        train_loader_name = 'train_loader_1'
    else:
        train_loader = train_loader_2
        train_loader_name = 'train_loader_2'
    with tqdm(total=len(train_loader), desc='Train-Progress') as pbar:
        for k, (image, label) in enumerate(train_loader):
            image = torch.Tensor(image).to(device)
            label = torch.Tensor(label).to(device)
            image_ = image[:, 0, ...]
            for n in range(1, image.shape[1]):
                image_[n::image.shape[1], ...] = image[n::image.shape[1], n, ...]
            image_pert = SAU_Net_(image_)
            image_pert = image_pert - torch.mean(image_pert, dim=[1, 2, 3]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            logit = classifier(image_ - image_pert)
            loss = F.cross_entropy(logit, label)
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()
            top1.update((logit.max(1)[1] == label).sum().item(), len(label))
            pbar.update(1)
    if train_loader_name == 'train_loader_1':
        train_loader_1_status = 'reclaimable'
    else:
        train_loader_2_status = 'reclaimable'
    return top1.acc_rate


def test_model(SAU_Net_):
    SAU_Net_.eval()
    with torch.no_grad():
        top1 = AverageMeter()
        with tqdm(total=len(test_loader), desc='Test--Progress') as pbar:
            for k, (image, label) in enumerate(test_loader):
                image = torch.Tensor(image).to(device)
                label = torch.Tensor(label).to(device)
                image_ = image[:, 0, ...]
                for n in range(1, image.shape[1]):
                    image_[n::image.shape[1], ...] = image[n::image.shape[1], n, ...]
                image_pert = SAU_Net_(image_)
                image_pert = image_pert - torch.mean(image_pert,
                                                     dim=[1, 2, 3]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                logit = classifier(image_ - image_pert)
                top1.update((logit.max(1)[1] == label).sum().item(), len(label))
                pbar.update(1)
    return top1.acc_rate


def main(SAU_Net_, OptimizerAdam, OptimizerSgd, Reload):
    global sau_net_load_path
    for epoch in range(config.epochs):
        print(f'Reload: {Reload + 1}/{config.reload}  |  Epoch: {epoch + 1}/{config.epochs}')
        if epoch in config.milestones:
            OptimizerAdam.param_groups[0]['lr'] *= 0.1
            OptimizerSgd.param_groups[0]['lr'] *= 0.1
        if epoch > config.milestones[-1]:
            train_acc = train_model(SAU_Net_, OptimizerSgd)
        else:
            train_acc = train_model(SAU_Net_, OptimizerAdam)
        test_acc = test_model(SAU_Net_)
        if test_acc > config.best_acc:
            torch.save(SAU_Net_.state_dict(), os.path.join(net_save_path, f'SAU_Net_{classifier_name}_{test_acc}.pt'))
            sau_net_load_path = os.path.join(net_save_path, f'SAU_Net_{classifier_name}_{test_acc}.pt')
            if os.path.exists(os.path.join(net_save_path, f'SAU_Net_{classifier_name}_{config.best_acc}.pt')):
                os.remove(os.path.join(net_save_path, f'SAU_Net_{classifier_name}_{config.best_acc}.pt'))
            config.best_acc = test_acc
        print(f'Train-Acc: {train_acc * 100:.4f}%  Test-Acc: {test_acc * 100:.4f}%  '
              f'Best-Test-Acc: {config.best_acc * 100:.4f}%')
        if train_loader_1_status == 'reclaimable' and train_loader_2_status == 'reclaimable':
            continue_event.wait()
        print('-' * 85)


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--attack_methods', type=tuple, default=['Clean', 'AA', 'BIM', 'DIM', 'FGSM', 'UPGD', 'VNIM'])
    parser.add_argument('--batch_size', type=int, default=70)
    parser.add_argument('--milestones', type=tuple, default=[40, 60, 80, 100], help='Stage of adjusting learning rate')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=110, help='Total epochs')
    parser.add_argument('--reload', type=int, default=10, help='Reload model')
    parser.add_argument('--data_amount', type=int, default=5, help='The amount of dataset')
    parser.add_argument('--sau_net_load_path', type=str, default=r'', help='The loaded SAU-Net')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'miniimagenet'])
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU number')
    parser.add_argument('--best_acc', type=float, default=0, help='The best accuracy')
    parser.add_argument('--classifier_name', type=str, default='MobileNetV2', help='The name of the classifier')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    config = parse_args()
    continue_event = threading.Event()
    device = torch.device(config.device)
    if config.dataset_name == 'cifar10':
        num_classes = 10
        data_path = os.path.join('AE', 'cifar10')
        net_save_path = os.path.join('SAU_Net', 'cifar10')
        classifier_load_path = cifar10_classifier_load_path
    elif config.dataset_name == 'cifar100':
        num_classes = 100
        data_path = os.path.join('AE', 'cifar100')
        net_save_path = os.path.join('SAU_Net', 'cifar100')
        classifier_load_path = cifar100_classifier_load_path
    else:
        num_classes = 100
        data_path = os.path.join('AE', 'miniimagenet')
        net_save_path = os.path.join('SAU_Net', 'miniimagenet')
        classifier_load_path = miniimagenet_classifier_load_path
    classifier_name = config.classifier_name
    test_AE_path = [os.path.join(data_path, classifier_name + '-' + att + '-AE-1.pkl') for att in config.attack_methods]
    testSet = AEDataset(test_AE_path, train=False)
    test_loader = DataLoader(testSet, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=1)

    train_loader_1 = create_train_loader()
    train_loader_1_status = 'ready'
    train_loader_2 = DataLoader
    train_loader_2_status = 'reclaimable'

    load_timer = threading.Timer(1, control_train_loader)
    load_timer.daemon = True
    load_timer.start()

    classifier = globals()[classifier_name](num_classes)
    classifier.load_state_dict(torch.load(classifier_load_path[classifier_name], map_location=device))
    classifier.to(device)
    classifier.eval()

    sau_net_load_path = config.sau_net_load_path
    for reload in range(config.reload):
        sau_net = SAU_Net()
        if sau_net_load_path != '':
            sau_net.load_state_dict(torch.load(sau_net_load_path, map_location=config.device))
        sau_net.to(device)
        print('Total params: %.2fM' % (sum(p.numel() for p in sau_net.parameters()) / 1000000.0))
        optimizer_adam = optim.Adam(sau_net.parameters(), lr=config.lr, weight_decay=0)
        optimizer_sgd = optim.SGD(sau_net.parameters(), lr=config.lr, momentum=0.9, weight_decay=0)
        main(sau_net, optimizer_adam, optimizer_sgd, reload)
    load_timer.cancel()
