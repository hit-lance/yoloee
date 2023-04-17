from backbone import CloudResNet50
from yoloee import YOLOEE


class CloudModel(YOLOEE):
    def __init__(self):
        super(CloudModel, self).__init__()
        self.backbone = CloudResNet50()

    def forward(self, x, s=0):
        # backbone
        inter = self.backbone(x, s)
        out = self.exit4(inter[0], inter[1])
        return out
