import torch.nn as nn
import torch.utils.model_zoo as model_zoo


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

    def forward1(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out
    
    def forward2(self,inter):
        out = self.conv2(inter)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        return out

    def split_forward(self, x):
        inter = self.forward1(x)
        out = self.forward2(inter)
        return out, inter
    


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
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        state_dict = model_zoo.load_url(
            'https://download.pytorch.org/models/resnet50-19c8e357.pth')
        del state_dict['fc.weight'], state_dict['fc.bias']
        self.load_state_dict(state_dict, strict=True)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        inter1 = self.conv1(x)
        inter1 = self.bn1(inter1)
        inter1 = self.relu(inter1)
        inter1 = self.maxpool(inter1)

        inter1 = self.layer1(inter1)

        for i in range(3):
            inter1 = self.layer2[i](inter1)
        inter2, inter1 = self.layer2[3].split_forward(inter1)

        for i in range(3):
            inter2 = self.layer3[i](inter2)
        inter3, inter2 = self.layer3[3].split_forward(inter2)

        for i in range(4, 6):
            inter3 = self.layer3[i](inter3)
        
        inter3 = self.layer4[0](inter3)
        inter4, inter3 = self.layer4[1].split_forward(inter3)
        inter4 = self.layer4[2](inter4)
        
        return inter1, inter2, inter3, inter4


class DeviceResNet50(ResNet50):
    def __init__(self):
        super(DeviceResNet50, self).__init__()

    def forward(self, x, s):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)

        for i in range(3):
            out = self.layer2[i](out)
        
        inter1 = self.layer2[3].forward1(out)
        if s == 1:
            return inter1
        out = self.layer2[3].forward2(inter1)

        for i in range(3):
            out = self.layer3[i](out)
        inter2 = self.layer3[3].forward1(out)
        if s == 2:
            return inter2
        out = self.layer3[3].forward2(inter2)

        for i in range(4, 6):
            out = self.layer3[i](out)
        
        out = self.layer4[0](out)
        inter3 = self.layer4[1].forward1(out)
        if s == 3:
            return inter3
        out = self.layer4[1].forward2(inter3)

        return out

class CloudResNet50(ResNet50):
    def __init__(self):
        super(CloudResNet50, self).__init__()

    def forward(self, x, s):
        if s == 0:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

        if s==1:
            out = self.layer2[3].forward2(x)
            out = self.layer3(out)
            out = self.layer4(out)

        elif s==2:
            out = self.layer3[3].forward2(x)
            for i in range(4, 6):
                out = self.layer3[i](out)
            out = self.layer4(out)

        elif s==3:
            out = self.layer4[1].forward2(x)
            out = self.layer4[2](out)

        return out