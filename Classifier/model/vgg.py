"""VGG11/13/16/19 in Pytorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _make_layers(Cfg):
    layers = []
    in_channels = 3
    for x in Cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes):
        super(VGG, self).__init__()
        self.features = _make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = F.adaptive_avg_pool2d(out, output_size=1)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def VGG11(num_classes=10):
    return VGG(vgg_name='VGG11', num_classes=num_classes)


def VGG13(num_classes=10):
    return VGG(vgg_name='VGG13', num_classes=num_classes)


def VGG16(num_classes=10):
    return VGG(vgg_name='VGG16', num_classes=num_classes)


def VGG19(num_classes=10):
    return VGG(vgg_name='VGG19', num_classes=num_classes)


if __name__ == '__main__':
    input1 = torch.rand([1, 3, 64, 64])
    model_x = VGG19()
    model_x(input1)
