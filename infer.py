import os
import numpy as np
import torch
from torch.autograd import Variable

from data.voc0712 import VOC_CLASSES, VOCDetection
from evaluate import evaluate_detections, get_output_dir, postprocess

from models.cloud_model import CloudModel
from models.device_model import DeviceModel

import utils
from data import config, BaseTransform

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

score_threshold_list = [i / 10 for i in range(3, 10)]

device_model = DeviceModel().to(device)
device_model.load_state_dict(torch.load('yoloee.pth', map_location=device))
device_model.eval()

cloud_model = CloudModel().to(device)
cloud_model.load_state_dict(torch.load('yoloee.pth', map_location=device))
cloud_model.eval()

cfg = config.yolov2_r50_cfg
dataset = VOCDetection(data_dir='data/VOCdevkit',
                       image_sets=[('2007', 'test')],
                       transform=BaseTransform(cfg['val_size']))
grid_cell, all_anchor_wh = utils.set_grid(416, cfg['anchor_size'], device)
num_images = len(dataset)
labelmap = VOC_CLASSES
annopath = os.path.join('data/VOCdevkit', 'VOC2007', 'Annotations', '%s.xml')
imgsetpath = os.path.join('data/VOCdevkit', 'VOC2007', 'ImageSets', 'Main',
                          'test' + '.txt')

all_boxes = [[[] for _ in range(num_images)] for _ in range(len(labelmap))]

for s in range(1, 4):
    print("s={}".format(s))
    for score_threshold in score_threshold_list:
        with torch.no_grad():
            for i in range(num_images):
                im, gt, h, w = dataset.pull_item(i)

                x = Variable(im.unsqueeze(0)).to(device)
                # forward
                inter, pred = device_model(x, s)
                bboxes, scores, cls_inds = postprocess(pred, grid_cell,
                                                       all_anchor_wh)

                if scores.size == 0 or np.mean(scores) < score_threshold:
                    pred = cloud_model(inter, s)
                    bboxes, scores, cls_inds = postprocess(
                        pred, grid_cell, all_anchor_wh)

                scale = np.array([[w, h, w, h]])
                bboxes *= scale

                for j in range(len(labelmap)):
                    inds = np.where(cls_inds == j)[0]
                    if len(inds) == 0:
                        all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                        continue
                    c_bboxes = bboxes[inds]
                    c_scores = scores[inds]
                    c_dets = np.hstack(
                        (c_bboxes, c_scores[:, np.newaxis])).astype(np.float32,
                                                                    copy=False)
                    all_boxes[j][i] = c_dets

            mAP = evaluate_detections(all_boxes, labelmap, False, dataset,
                                      'data/VOCdevkitVOC',
                                      get_output_dir('voc_eval/', 'test'),
                                      'test', imgsetpath, annopath)
            print(mAP)
