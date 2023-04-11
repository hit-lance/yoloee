import torch
import torch.nn as nn
from backbone.resnet import ResNet50
from utils.modules import CBL, reorg_layer


class EarlyExit(nn.Module):

    def __init__(self, in_channels, num_classes, num_anchors):
        super(EarlyExit, self).__init__()
        layers = []
        while in_channels != 1024:
            layers.append(
                CBL(in_channels=in_channels,
                    out_channels=in_channels * 2,
                    kernel_size=3,
                    stride=2,
                    padding=2,
                    dilation=2))
            in_channels *= 2

        layers.append(CBL(1024, 1024, kernel_size=1, stride=2))
        self.features = nn.Sequential(*layers)

        # self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pred = nn.Conv2d(1024, num_anchors * (1 + 4 + num_classes), 1)

    def forward(self, x):
        out = self.features(x)
        out = self.pred(out)
        return out


class FinalExit(nn.Module):

    def __init__(self, num_classes, num_anchors):
        super(FinalExit, self).__init__()
        self.cbl1 = CBL(2048, 1024, kernel_size=1)
        self.cbl2 = CBL(1024, 1024, kernel_size=3, padding=1)
        self.route_layer = CBL(1024, 128, kernel_size=1)
        self.reorg = reorg_layer(stride=2)
        self.cbl3 = CBL(1024 + 128 * 4, 1024, kernel_size=3, padding=1)
        # pred
        self.pred = nn.Conv2d(1024, num_anchors * (1 + 4 + num_classes), 1)

    def forward(self, x, y):
        x = self.route_layer(x)
        x = self.reorg(x)
        y = self.cbl1(y)
        y = self.cbl2(y)
        out = torch.cat([x, y], dim=1)
        out = self.cbl3(out)
        out = self.pred(out)
        return out


class YOLOv2R50(nn.Module):

    def __init__(self, num_classes=20, num_anchors=5):
        super(YOLOv2R50, self).__init__()

        # backbone
        self.backbone = ResNet50()

        self.exit1 = EarlyExit(256, num_classes, num_anchors)
        self.exit2 = EarlyExit(512, num_classes, num_anchors)
        self.exit3 = EarlyExit(1024, num_classes, num_anchors)
        self.exit4 = EarlyExit(1024, num_classes, num_anchors)
        self.exit5 = FinalExit(num_classes, num_anchors)

    def forward(self, x):
        # backbone
        inters = self.backbone(x)

        out1 = self.exit1(inters[0])
        out2 = self.exit2(inters[1])
        out3 = self.exit3(inters[2])
        out4 = self.exit4(inters[3])
        out5 = self.exit5(inters[3], inters[4])

        return out1, out2, out3, out4, out5
