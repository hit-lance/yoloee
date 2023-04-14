import torch.nn as nn
from models.backbone import DeviceResNet50
from models.yoloee import YOLOEE


class DeviceModel(YOLOEE):
    def __init__(self):
        super(DeviceModel, self).__init__()
        self.backbone = DeviceResNet50()

    def forward(self, x, s=4):
        # backbone
        inter = self.backbone(x, s)

        if s == 1:
            out = self.exit1(inter)
        elif s == 2:
            out = self.exit2(inter)
        elif s == 3:
            out = self.exit3(inter)
        elif s == 4:
            out = self.exit4(inter[0], inter[1])
            return None, out

        return inter, out
