import copy
from torchvision import datasets
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import wasserstein_distance
from utils import *
from model import *


def get_logit(Loader, Num, Cla, Device):
    logit_total = torch.zeros([len(Loader.dataset), Num])
    with torch.no_grad():
        with tqdm(total=len(Loader), desc='Test--Progress') as pbar:
            for k, (image, label) in enumerate(Loader):
                image, label = torch.Tensor(image).to(Device), torch.Tensor(label).to(Device)
                logit = Cla(image)
                logit_total[k * Loader.batch_size: k * Loader.batch_size + len(image)] = logit
                pbar.update()

    return logit_total


def h(x):
    x = copy.deepcopy(x)
    dim = 1
    try:
        x[list(x.keys())[0]].keys()
        dim += 1
    except AttributeError:
        pass
    if dim == 1:
        x_sum = 0
        for k in list(x.keys()):
            x_sum += x[k]
        for k in list(x.keys()):
            x[k] /= x_sum
        return x
    else:
        for k in list(x.keys()):
            x[k] = h(x[k])
        return x


class PPSA:
    def __init__(self, dataset_name, data_path, classifier_name, cla_config, device):
        if dataset_name == 'cifar10':
            testSet = datasets.CIFAR10(data_path, False, download=True, transform=transforms.ToTensor())
            num_classes = 10
        elif dataset_name == 'cifar100':
            testSet = datasets.CIFAR100(data_path, False, download=True, transform=transforms.ToTensor())
            num_classes = 100
        else:
            testSet = datasets.CIFAR10(data_path, False, download=True, transform=transforms.ToTensor())
            num_classes = 100
        test_loader = DataLoader(testSet, batch_size=256, shuffle=False, pin_memory=True, num_workers=1)
        cla_logit = {}
        for cla_name in classifier_name:
            cla = globals()[cla_name](num_classes)
            cla.load_state_dict(torch.load(cla_config.classifiers_path[dataset_name][cla_name], map_location=device))
            cla.to(device)
            cla.eval()
            print(cla_name)
            temp = get_logit(test_loader, num_classes, cla, device)
            cla_logit[cla_name] = temp.view(-1).clone().detach().cpu().numpy()
            del cla

        self.mod_def_pro, self.mom_pro = {}, {}
        for cla_name_1 in classifier_name:
            temp1 = {}
            temp2 = {}
            for cla_name_2 in classifier_name:
                if cla_name_2 != cla_name_1:
                    temp1[cla_name_2] = math.log(wasserstein_distance(cla_logit[cla_name_1], cla_logit[cla_name_2]) + 1)
                    temp2[cla_name_2] = 1
            self.mod_def_pro[cla_name_1] = temp1
            self.mom_pro[cla_name_1] = temp2
        self.mod_def_pro = h(self.mod_def_pro)
        self.selected_cla_name = None
        self.classifier_name = classifier_name

    def determinate(self):
        if self.selected_cla_name is None:
            self.selected_cla_name = random.choice(self.classifier_name)
        else:
            last_model = self.selected_cla_name
            temp = h(self.mom_pro)
            for cla_name_1 in list(temp.keys()):
                for cla_name_2 in list(temp[cla_name_1].keys()):
                    temp[cla_name_1][cla_name_2] = 1 - temp[cla_name_1][cla_name_2]
            neg_mom_pro = h(temp)
            sch_pro = copy.deepcopy(self.mod_def_pro)
            for cla_name_1 in list(temp.keys()):
                for cla_name_2 in list(temp[cla_name_1].keys()):
                    sch_pro[cla_name_1][cla_name_2] *= neg_mom_pro[cla_name_1][cla_name_2]
            sch_pro = h(sch_pro)
            p = random.uniform(0, 1)
            pro_sum = 0
            sch_model = ''
            for cla_name in list(sch_pro[last_model].keys()):
                pro_sum += sch_pro[last_model][cla_name]
                if pro_sum >= p:
                    sch_model = cla_name
                    break
                sch_model = cla_name
            self.mom_pro[last_model][sch_model] += 1
            self.selected_cla_name = sch_model

        return self.selected_cla_name
