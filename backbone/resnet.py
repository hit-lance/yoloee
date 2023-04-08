import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['resnet50']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):

    def __init__(self):
        super(ResNet50, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = nn.Sequential(
            self._make_layer(Bottleneck, 128, 3, stride=2),
            self._make_layer(Bottleneck, 128, 1, stride=2, ds=False))
        self.layer3 = nn.Sequential(
            self._make_layer(Bottleneck, 256, 2, stride=2),
            self._make_layer(Bottleneck, 256, 4, stride=2, ds=False))
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.load_state_dict(model_zoo.load_url(
            'https://download.pytorch.org/models/resnet50-19c8e357.pth'),
                             strict=False)

    def _make_layer(self, block, planes, blocks, stride=1, ds=True):
        layers = []
        if ds:
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
        else:
            layers.append(block(self.inplanes, planes))

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c1 = self.maxpool(c1)

        c2 = self.layer1(c1)
        
        c2 = self.layer2[0](c2)

        c3 = self.layer2[1](c2)
        c3 = self.layer3[0](c3)

        c4 = self.layer3[1](c3)
        
        c5 = self.layer4(c4)

        output = {'layer1': c3, 'layer2': c4, 'layer3': c5}

        return output
