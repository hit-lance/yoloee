import torch
import torch.nn as nn
from backbone.resnet import ResNet50
from utils.modules import Conv, reorg_layer


class YOLOv2R50(nn.Module):

    def __init__(self,
                 device,
                 input_size=None,
                 num_classes=20,
                 anchor_size=None,
                 ):
        super(YOLOv2R50, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.anchor_size = torch.tensor(anchor_size)
        self.num_anchors = len(anchor_size)
        self.set_grid(input_size)

        # backbone
        self.backbone = ResNet50()

        # head
        self.convsets_1 = nn.Sequential(Conv(2048, 1024, k=1),
                                        Conv(1024, 1024, k=3, p=1),
                                        Conv(1024, 1024, k=3, p=1))

        # reorg
        self.route_layer = Conv(1024, 128, k=1)
        self.reorg = reorg_layer(stride=2)

        # head
        self.convsets_2 = Conv(1024 + 128 * 4, 1024, k=3, p=1)

        # pred
        self.pred = nn.Conv2d(1024,
                              self.num_anchors * (1 + 4 + self.num_classes), 1)

        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.pred.bias[..., :self.num_anchors], bias_value)

    def set_grid(self, input_size):
        w, h = input_size, input_size
        # generate grid cells
        ws, hs = w // 32, h // 32
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        self.grid_cell = grid_xy.view(1, hs * ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        self.all_anchor_wh = self.anchor_size.repeat(
            hs * ws, 1, 1).unsqueeze(0).to(self.device)

    def decode_boxes(self, txtytwth_pred):
        """
            Input: \n
                txtytwth_pred : [B, H*W, anchor_n, 4] \n
            Output: \n
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] \n
        """
        # txtytwth -> cxcywh
        B, HW, ab_n, _ = txtytwth_pred.size()
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_cell
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.all_anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, -1, 4) * 32

        # cxcywh -> x1y1x2y2
        x1y1x2y2_pred = torch.zeros_like(xywh_pred)
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] * 0.5
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] * 0.5
        x1y1x2y2_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return x1y1x2y2_pred

    def forward(self, x):
        # backbone
        feats = self.backbone(x)

        # reorg layer
        p5 = self.convsets_1(feats['layer3'])
        p4 = self.reorg(self.route_layer(feats['layer2']))
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
        box_pred = self.decode_boxes(reg_pred)

        return conf_pred, cls_pred, reg_pred, box_pred
