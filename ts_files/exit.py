import torch
import torch.nn as nn
from common import CBL, reorg_layer


class EarlyExit(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        super(EarlyExit, self).__init__()
        layers = []
        if in_channels != 512:
            if in_channels == 128:
                layers.append(nn.Conv2d(128, 256, 3, 2, 2, 2))
                layers.append(nn.BatchNorm2d(256))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(256, 512, 3, 2, 1))
            layers.append(nn.BatchNorm2d(512))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(512, 512, 3, 1, 1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(512, 1024, 1, 1, 0))
        layers.append(nn.BatchNorm2d(1024))
        layers.append(nn.ReLU(inplace=True))

        self.features = nn.Sequential(*layers)

        # self.cbl1 = CBL(2048, 1024, kernel_size=1)
        self.cbl2 = CBL(1024, 1024, kernel_size=3, padding=1)

        self.pred = nn.Conv2d(1024, num_anchors * (1 + 4 + num_classes), 1)

    def forward(self, x):
        out = self.features(x)
        # out = self.cbl1(out)
        out = self.cbl2(out)
        out = self.pred(out)
        return out


class FinalExit(nn.Module):
    def __init__(self, num_classes, num_anchors):
        super(FinalExit, self).__init__()
        self.cbl1 = CBL(2048, 1024, kernel_size=1)
        self.cbl2 = CBL(1024, 1024, kernel_size=3, padding=1)
        self.cbl3 = CBL(1024, 1024, kernel_size=3, padding=1)
        # pred
        self.pred = nn.Conv2d(1024, num_anchors * (1 + 4 + num_classes), 1)

    def forward(self, y):
        y = self.cbl1(y)
        y = self.cbl2(y)
        out = self.cbl3(y)
        out = self.pred(out)
        return out