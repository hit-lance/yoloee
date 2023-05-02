import torch.nn as nn
from backbone import ResNet50
from exit import EarlyExit, FinalExit


class YOLOEE(nn.Module):
    def __init__(self, num_classes=20, num_anchors=5):
        super(YOLOEE, self).__init__()

        # backbone
        self.backbone = ResNet50()

        self.exit1 = EarlyExit(128, num_classes, num_anchors)
        self.exit2 = EarlyExit(256, num_classes, num_anchors)
        self.exit3 = EarlyExit(512, num_classes, num_anchors)
        self.exit4 = FinalExit(num_classes, num_anchors)

    def forward(self, x):
        inters = self.backbone(x)
        out1 = self.exit1(inters[0])
        out2 = self.exit2(inters[1])
        out3 = self.exit3(inters[2])
        out4 = self.exit4(inters[3])

        return out1, out2, out3, out4