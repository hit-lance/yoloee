from prometheus_client import start_http_server, Summary
from torch.autograd import Variable

import numpy as np
import requests
from compress import compress
import torch
from data import BaseTransform, config
from data.voc0712 import VOCDetection
import utils

from models.device_model import DeviceModel
from evaluate import postprocess

import time

if torch.cuda.is_available():
    print("use cuda...")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

DEVICE_INFER_TIME = Summary('device_infer', 'Device infer')
CLOUD_INFER_TIME = Summary('cloud_infer', 'Cloud infer(include network cost)')

cfg = config.yolov2_r50_cfg
dataset = VOCDetection(
    data_dir='data/VOCdevkit',
    image_sets=[('2007', 'test')],
    transform=BaseTransform(cfg['val_size']))
grid_cell, all_anchor_wh = utils.set_grid(416, cfg['anchor_size'], device)


class Session:
    def __init__(self, split_point, threshold=1, metrics_port=8003):

        self.device_model = DeviceModel().to(device).eval()
        self.device_model.load_state_dict(
            torch.load('yoloee.pth', map_location=device))
        print("model loaded")
        self.split_point = split_point
        self.threshold = threshold

        self.http_session = requests.Session()
        self.infer_url = "http://192.168.31.5:8080/predictions/yoloee"

        start_http_server(8082)

    @torch.no_grad()
    @DEVICE_INFER_TIME.time()
    def device_infer(self, x):
        inter, model_output = self.device_model(x, self.split_point)
        bboxes, scores, cls_inds = postprocess(model_output, grid_cell,
                                               all_anchor_wh)
        # torch.cuda.synchronize(), postprocess synchronize instead
        return inter, bboxes, scores, cls_inds

    @CLOUD_INFER_TIME.time()
    def cloud_infer(self, x):
        x = x.to('cpu').numpy()

        data = {"split_point": self.split_point.to_bytes(1, byteorder='big')}

        if self.split_point != 0:
            x, x_max, x_min = compress(x)
            data["x_max"] = x_max.tobytes()
            data["x_min"] = x_min.tobytes()

        data["model_input"] = x.tobytes()

        res = self.http_session.post(url=self.infer_url, data=data)

        model_output = np.frombuffer(
            res.content, dtype=np.float32).reshape(-1, 6)
        bboxes = model_output[:, :4]
        scores = model_output[:, 4]
        cls_inds = model_output[:, 5].astype(np.int64)
        return bboxes, scores, cls_inds

    def synergistic_infer(self, x):
        if self.split_point == 0:
            bboxes, scores, cls_inds = self.cloud_infer(x)
        elif self.split_point == 4:
            _, bboxes, scores, cls_inds = self.device_infer(x)
        else:
            inter, bboxes, scores, cls_inds = self.device_infer(x)
            if scores.size == 0 or np.mean(scores) < self.threshold:
                bboxes, scores, cls_inds = self.cloud_infer(inter)

        return bboxes, scores, cls_inds


if __name__ == "__main__":
    dataset = VOCDetection(
        data_dir='data/VOCdevkit',
        image_sets=[('2007', 'test')],
        transform=BaseTransform(416))
    num_images = len(dataset)

    session = Session(split_point=1)

    while True:
        for i in range(num_images):
            im, gt, h, w = dataset.pull_item(i)

            x = Variable(im.unsqueeze(0)).to(device)
            bboxes, scores, cls_inds = session.synergistic_infer(x)
