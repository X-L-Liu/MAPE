import torch
import torch.nn as nn


class SplitAdd:

    def __call__(self, x):
        output_1 = torch.split(x, x.shape[1] // 2, dim=1)
        output_2 = torch.add(output_1[0], output_1[1])
        return output_2


class ChanAttn(nn.Module):
    def __init__(self, channels):
        super(ChanAttn, self).__init__()
        self.atte = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels, out_channels=max(channels // 4, 32), kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=max(channels // 4, 32), out_channels=channels, kernel_size=1, bias=True),
            nn.Mish(inplace=True)
        )

    def forward(self, x):
        output_1 = torch.mul(x, self.atte(x))
        output_2 = torch.split(output_1, output_1.shape[1] // 2, dim=1)
        output_3 = torch.add(output_2[0], output_2[1])
        return output_3


class SubMiddle(nn.Module):
    def __init__(self, channels, init_channels, highest, pre_subMiddle, chan_att):
        super(SubMiddle, self).__init__()
        self.subMiddle_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=init_channels if highest else channels,
                out_channels=channels,
                kernel_size=3,
                stride=1 if highest else 2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.Mish(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Mish(inplace=True)
        )

        self.subMiddle_2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.Mish(inplace=True),
            pre_subMiddle,
            nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Mish(inplace=True),
            nn.ConvTranspose2d(in_channels=channels, out_channels=channels // 2, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.Mish(inplace=True),
        )

        self.skip = ChanAttn(channels) if chan_att else SplitAdd()

        self.subMiddle_3 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(channels),
            nn.Mish(inplace=True)
        )

    def forward(self, x):
        output_1 = self.subMiddle_1(x)
        output_2 = torch.cat((self.subMiddle_2(output_1), self.skip(output_1)), dim=1)
        output_3 = self.subMiddle_3(output_2)
        return output_3


class CAU_Net(nn.Module):
    """
    Args:
        in_channels (int): Number of channels in the input image. Default: 3
        out_channels (int): Number of channels produced by this model_class. Default: 3
        init_conv_channels (int): Number of channels for the first convolution. Default: 32
        num_down (int): Number of down-sampling layers. Default: 5
        channel_attention (str, optional): True or False. Default: True

    """

    def __init__(self, in_channels=3, out_channels=3, init_conv_channels=32, num_down=5, channel_attention=True):
        super().__init__()
        max_channels = init_conv_channels * 2 ** (num_down - 1)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=max_channels, out_channels=max_channels, kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(max_channels),
            nn.Mish(inplace=True)
        )
        for n in range(num_down - 1)[::-1]:
            self.features = SubMiddle(
                channels=init_conv_channels * 2 ** n,
                init_channels=in_channels,
                highest=True if n == 0 else False,
                pre_subMiddle=self.features,
                chan_att=channel_attention
            )

        self.generation = nn.Sequential(
            nn.Conv2d(
                in_channels=init_conv_channels,
                out_channels=init_conv_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(init_conv_channels // 2),
            nn.Mish(inplace=True),
            nn.Conv2d(
                in_channels=init_conv_channels // 2,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        output_1 = self.features(x)
        output_2 = self.generation(output_1)
        return output_2
