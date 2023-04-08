import torch
import torch.nn as nn
from backbone.resnet import ResNet50
from utils.modules import CBL, reorg_layer


class YOLOv2R50(nn.Module):

    def __init__(self, num_classes=20, num_anchors=5):
        super(YOLOv2R50, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # backbone
        self.backbone = ResNet50()

        # head
        self.convsets_1 = nn.Sequential(
            CBL(2048, 1024, kernel_size=1),
            CBL(1024, 1024, kernel_size=3, padding=1))

        # reorg
        self.route_layer = CBL(1024, 128, kernel_size=1)
        self.reorg = reorg_layer(stride=2)

        # head
        self.convsets_2 = CBL(1024 + 128 * 4, 1024, kernel_size=3, padding=1)

        # pred
        self.pred = nn.Conv2d(1024,
                              self.num_anchors * (1 + 4 + self.num_classes), 1)

        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.pred.bias[..., :self.num_anchors], bias_value)

    def forward(self, x):
        # backbone
        inters = self.backbone(x)

        # reorg layer
        p5 = self.convsets_1(inters[4])
        p4 = self.reorg(self.route_layer(inters[3]))
        p5 = torch.cat([p4, p5], dim=1)

        # head
        p5 = self.convsets_2(p5)

        # pred
        pred = self.pred(p5)

        B, abC, H, W = pred.size()

        # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
        pred = pred.permute(0, 2, 3, 1).contiguous().view(B, H * W, abC)

        # [B, H*W*num_anchor, 1]
        conf_pred = pred[:, :, :1 * self.num_anchors].contiguous().view(
            B, H * W * self.num_anchors, 1)
        # [B, H*W, num_anchor, num_cls]
        cls_pred = pred[:, :, 1 * self.num_anchors:(1 + self.num_classes) *
                        self.num_anchors].contiguous().view(
                            B, H * W * self.num_anchors, self.num_classes)
        # [B, H*W, num_anchor, 4]
        reg_pred = pred[:, :, (1 + self.num_classes) *
                        self.num_anchors:].contiguous()
        reg_pred = reg_pred.view(B, H * W, self.num_anchors, 4)

        return conf_pred, cls_pred, reg_pred
