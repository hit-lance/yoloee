import torch
import torch.nn as nn
from backbone.resnet import ResNet50
from utils.modules import CBL, reorg_layer


# class EarlyExit(nn.Module):

#     def __init__(self, in_channels ,num_classes, num_anchors):
#         super(EarlyExit, self).__init__()
#         self.downsample = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels*2, 1, 2, 0),
#             nn.BatchNorm2d(in_channels*2),
#         )
#         layers = []

#         layers.append(nn.Conv2d(in_channels, in_channels//2, 1, 1, 0))
#         layers.append(nn.BatchNorm2d(in_channels//2))
#         layers.append(nn.LeakyReLU(0.1, inplace=True))
#         layers.append(nn.Conv2d(in_channels//2, in_channels//2, 3, 2, 1))
#         layers.append(nn.BatchNorm2d(in_channels//2))
#         layers.append(nn.LeakyReLU(0.1, inplace=True))
#         layers.append(nn.Conv2d(in_channels//2, in_channels*2, 1, 1, 0))
#         layers.append(nn.BatchNorm2d(in_channels*2))
#         self.features = nn.Sequential(*layers)

#         self.relu = nn.LeakyReLU(0.1, inplace=True)
#         self.cbl1 = CBL(in_channels*2, 1024, kernel_size=1, stride=1024//in_channels)
#         self.cbl2 = CBL(1024, 1024, kernel_size=3, padding=1)

#         self.pred = nn.Conv2d(1024, num_anchors * (1 + 4 + num_classes), 1)
        
#         def forward(self, x):
#             identity = self.downsample(x)
#             out = self.features(x)
#             out += identity
#             out = self.relu(out)
#             out = self.cbl1(out)
#             out = self.cbl2(out)
#             out = self.pred(out)
#             return out


class EarlyExit(nn.Module):

    def __init__(self, in_channels, num_classes, num_anchors):
        super(EarlyExit, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(512, in_channels, 3, 2, 2, 2)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, 2048, 1, 2, 0),
            nn.BatchNorm2d(2048),
        )
        layers = []

        layers.append(nn.Conv2d(in_channels, 512, 1, 1, 0))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(nn.Conv2d(512, 512, 3, 2, 1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(nn.Conv2d(512, 2048, 1, 1, 0))
        layers.append(nn.BatchNorm2d(2048))
        self.features = nn.Sequential(*layers)

        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.cbl1 = CBL(2048, 1024, kernel_size=1)
        self.cbl2 = CBL(1024, 1024, kernel_size=3, padding=1)

        self.pred = nn.Conv2d(1024, num_anchors * (1 + 4 + num_classes), 1)

    def forward(self, x):
        if self.in_channels==512:     
            x = self.conv(x)
        identity = self.downsample(x)
        out = self.features(x)
        out += identity
        out = self.relu(out)
        out = self.cbl1(out)
        out = self.cbl2(out)
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

        self.exit1 = EarlyExit(512, num_classes, num_anchors)
        self.exit2 = EarlyExit(1024, num_classes, num_anchors)
        self.exit3 = EarlyExit(1024, num_classes, num_anchors)
        self.exit4 = FinalExit(num_classes, num_anchors)

    def forward(self, x):
        # backbone
        inters = self.backbone(x)

        out1 = self.exit1(inters[0])
        out2 = self.exit2(inters[1])
        out3 = self.exit3(inters[2])
        out4 = self.exit4(inters[2], inters[3])

        return out1, out2, out3, out4
