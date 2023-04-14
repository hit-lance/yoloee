import argparse
import time
import torch
from torch.autograd import Variable

from data.voc0712 import VOC_CLASSES, VOCDetection
from evaluate import postprocess

from models.cloud_model import CloudModel
from models.device_model import DeviceModel

import utils
from data import config, BaseTransform

parser = argparse.ArgumentParser(description='Measurement')
parser.add_argument('--cloud',
                    action='store_true',
                    default=False,
                    help='measure cloud')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("use cuda...")
else:
    device = torch.device("cpu")

cloud_model = CloudModel().to(device)
cloud_model.load_state_dict(torch.load('yoloee.pth', map_location=device))
cloud_model.eval()

device_model = DeviceModel().to(device)
device_model.load_state_dict(torch.load('yoloee.pth', map_location=device))
device_model.eval()

cfg = config.yolov2_r50_cfg
dataset = VOCDetection(data_dir='data/VOCdevkit',
                       image_sets=[('2007', 'test')],
                       transform=BaseTransform(cfg['val_size']))
grid_cell, all_anchor_wh = utils.set_grid(416, cfg['anchor_size'], device)
num_images = len(dataset)

result = []
if args.cloud:
    infer_time = []
    with torch.no_grad():
        for i in range(num_images):
            im, gt, h, w = dataset.pull_item(i)

            x = Variable(im.unsqueeze(0)).to(device)
            # forward
            t0 = time.time()
            pred = cloud_model(x, 0)
            postprocess(pred, grid_cell, all_anchor_wh)
            t1 = time.time() - t0
            infer_time.append(t1)
    result.append(sum(infer_time) / len(infer_time))

for s in range(1, 5):
    infer_time = []
    with torch.no_grad():
        for i in range(num_images):
            im, gt, h, w = dataset.pull_item(i)

            x = Variable(im.unsqueeze(0)).to(device)
            # forward
            t0 = time.time()
            inter, pred = device_model(x, s)
            postprocess(pred, grid_cell, all_anchor_wh)
            t1 = time.time() - t0

            if args.cloud and s != 4:
                t0 = time.time()
                pred = cloud_model(inter, s)
                postprocess(pred, grid_cell, all_anchor_wh)
                t1 = time.time() - t0

            infer_time.append(t1)
    result.append(sum(infer_time) / len(infer_time))

print(result)