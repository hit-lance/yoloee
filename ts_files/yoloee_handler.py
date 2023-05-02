# custom handler file

# model_handler.py
"""
ModelHandler defines a custom model handler.
"""

import math
import os
import numpy as np
import torch
import bitshuffle

from ts.torch_handler.base_handler import BaseHandler


def set_grid(input_size, anchor_size, device):
    w, h = input_size, input_size
    # generate grid cells
    ws, hs = w // 32, h // 32
    grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
    grid_cell = grid_xy.view(1, hs * ws, 1, 2).to(device)

    # generate anchor_wh tensor
    all_anchor_wh = torch.tensor(anchor_size).repeat(hs * ws, 1,
                                                     1).unsqueeze(0).to(device)

    return grid_cell, all_anchor_wh


def decode_boxes(txtytwth_pred, grid_cell, all_anchor_wh):
    """
        Input: \n
            txtytwth_pred : [B, H*W, anchor_n, 4] \n
        Output: \n
            x1y1x2y2_pred : [B, H*W*anchor_n, 4] \n
    """
    # txtytwth -> cxcywh
    B = txtytwth_pred.size()[0]
    txtytwth_pred = txtytwth_pred.view(B, -1, all_anchor_wh.size()[2], 4)
    xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + grid_cell
    wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * all_anchor_wh
    # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
    xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, -1, 4) * 32

    # cxcywh -> x1y1x2y2
    x1y1x2y2_pred = torch.zeros_like(xywh_pred)
    x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] * 0.5
    x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] * 0.5
    x1y1x2y2_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

    return x1y1x2y2_pred


def divide(pred, num_classes=20, num_anchors=5):
    B, abC, H, W = pred.size()

    # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
    pred = pred.permute(0, 2, 3, 1).contiguous().view(B, H * W, abC)

    # [B, H*W*num_anchor, 1]
    conf_pred = pred[:, :, :1 * num_anchors].contiguous().view(
        B, H * W * num_anchors, 1)
    # [B, H*W, num_anchor, num_cls]
    cls_pred = pred[:, :, 1 * num_anchors:(1 + num_classes) *
                    num_anchors].contiguous().view(B, H * W * num_anchors,
                                                   num_classes)
    # [B, H*W, num_anchor, 4]
    reg_pred = pred[:, :, (1 + num_classes) * num_anchors:].contiguous().view(
        B, H * W * num_anchors, 4)

    return conf_pred, cls_pred, reg_pred


def nms(dets, scores, nms_thresh=0.5):
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


def linear_dequantize(x_q, x_max, x_min):
    scale = (x_max - x_min) / 255
    x = x_min + scale * x_q
    return x


def uncompress(x_c, x_max, x_min, x_shape):
    x = bitshuffle.decompress_lz4(x_c, (math.prod(x_shape), ),
                                  np.dtype('uint8'))
    x = linear_dequantize(x, x_max, x_min)
    x = x.reshape(x_shape)
    return x


class YOLOEEHandler(BaseHandler):
    """
    A custom model handler implementation.
    """
    def __init__(self):
        super().__init__()
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.grid_cell = None
        self.all_anchor_wh = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.initialized = True

        properties = context.system_properties
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        model_file = self.manifest["model"].get("modelFile", "")
        serialized_file = self.manifest["model"]["serializedFile"]
        self.model_pt_path = os.path.join(model_dir, serialized_file)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.grid_cell, self.all_anchor_wh = set_grid(
            416, [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29],
                  [10.32, 10.65]], self.device)

        self.model = self._load_pickled_model(model_dir, model_file,
                                              self.model_pt_path)
        self.model.to(self.device)
        self.model.eval()
        #  load the model, refer 'custom handler class' above for details

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        data = data[0]

        s = int.from_bytes(data["split_point"], byteorder='big')

        if s == 0:
            model_input = np.frombuffer(data["model_input"],
                                        dtype=np.float32).reshape(
                                            1, 3, 416, 416)
        else:
            x_c = np.frombuffer(data["model_input"], dtype=np.uint8)
            x_max = np.frombuffer(data["x_max"], dtype=np.float32)
            x_min = np.frombuffer(data["x_min"], dtype=np.float32)

            if s == 1:
                x_shape = (1, 128, 52, 52)
            elif s == 2:
                x_shape = (1, 256, 26, 26)
            else:
                x_shape = (1, 512, 13, 13)

            model_input = uncompress(x_c, x_max, x_min, x_shape)

        model_input = torch.from_numpy(model_input).float()

        return model_input, s

    def inference(self, model_input, s):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        with torch.no_grad():
            model_input = model_input.to(self.device)
            model_output = self.model.forward(model_input, s)
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        num_classes = 20
        conf_thresh = 0.2
        nms_thresh = 0.5

        conf_pred, cls_pred, reg_pred = divide(inference_output)
        bbox_pred = decode_boxes(reg_pred, self.grid_cell, self.all_anchor_wh)

        # score
        scores = torch.sigmoid(conf_pred[0]) * torch.softmax(cls_pred[0],
                                                             dim=-1)

        # normalize bbox
        bboxes = torch.clamp(bbox_pred[0] / 416, 0., 1.)

        # to cpu
        scores = scores.to('cpu').numpy()
        bboxes = bboxes.to('cpu').numpy()

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]

        # threshold
        keep = np.where(scores >= conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=int)
        for i in range(num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = nms(c_bboxes, c_scores, nms_thresh)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep].reshape(-1, 1)
        cls_inds = cls_inds[keep].reshape(-1, 1).astype(np.float32)

        postprocess_output = np.hstack((bboxes, scores, cls_inds))
        return [postprocess_output.tobytes()]

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        inter, s = self.preprocess(data)
        model_output = self.inference(inter, s)
        return self.postprocess(model_output)
