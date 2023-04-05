import torch
import numpy as np


def decode_xywh(txtytwth_pred):
    """
        Input: \n
            txtytwth_pred : [B, H*W, anchor_n, 4] \n
        Output: \n
            xywh_pred : [B, H*W*anchor_n, 4] \n
    """
    B, HW, ab_n, _ = txtytwth_pred.size()
    # b_x = sigmoid(tx) + gride_x
    # b_y = sigmoid(ty) + gride_y
    xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_cell
    # b_w = anchor_w * exp(tw)
    # b_h = anchor_h * exp(th)
    wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.all_anchor_wh
    # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
    xywh_pred = torch.cat([xy_pred, wh_pred], -
                          1).view(B, -1, 4) * 32

    return xywh_pred


def decode_boxes(txtytwth_pred):
    """
        Input: \n
            txtytwth_pred : [B, H*W, anchor_n, 4] \n
        Output: \n
            x1y1x2y2_pred : [B, H*W*anchor_n, 4] \n
    """
    # txtytwth -> cxcywh
    xywh_pred = decode_xywh(txtytwth_pred)

    # cxcywh -> x1y1x2y2
    x1y1x2y2_pred = torch.zeros_like(xywh_pred)
    x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] * 0.5
    x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] * 0.5
    x1y1x2y2_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

    return x1y1x2y2_pred


def nms(dets, scores):
    nms_thresh = 0.6

    """"Pure Python NMS baseline."""
    x1 = dets[:, 0]  # xmin
    y1 = dets[:, 1]  # ymin
    x2 = dets[:, 2]  # xmax
    y2 = dets[:, 3]  # ymax

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h

        # Cross Area / (bbox + particular area - Cross Area)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # reserve all the boundingbox whose ovr less than thresh
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


def postprocess(bboxes, scores):
    """
    bboxes: (HxW, 4), bsize = 1
    scores: (HxW, num_classes), bsize = 1
    """

    cls_inds = np.argmax(scores, axis=1)
    scores = scores[(np.arange(scores.shape[0]), cls_inds)]

    # threshold
    conf_thresh = 0.001
    keep = np.where(scores >= conf_thresh)
    bboxes = bboxes[keep]
    scores = scores[keep]
    cls_inds = cls_inds[keep]

    # NMS
    keep = np.zeros(len(bboxes), dtype=int)
    num_classes = 20
    for i in range(num_classes):
        inds = np.where(cls_inds == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bboxes[inds]
        c_scores = scores[inds]
        c_keep = nms(c_bboxes, c_scores)
        keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)
    bboxes = bboxes[keep]
    scores = scores[keep]
    cls_inds = cls_inds[keep]

    return bboxes, scores, cls_inds
