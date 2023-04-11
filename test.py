import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from data.voc0712 import VOC_CLASSES, VOCDetection
from data import config, BaseTransform
import numpy as np
import cv2
import time

from models.yolov2_r50 import YOLOv2R50
from utils import divide
from utils.anchor import decode_boxes, set_grid
from val import postprocess

parser = argparse.ArgumentParser(description='YOLO Detection')
# basic
parser.add_argument('-size',
                    '--input_size',
                    default=416,
                    type=int,
                    help='input_size')
parser.add_argument('--cuda',
                    action='store_true',
                    default=False,
                    help='use cuda.')
# model
parser.add_argument(
    '-v',
    '--version',
    default='yolo_v2',
    help='yolov2_d19, yolov2_r50, yolov2_slim, yolov3, yolov3_spp, yolov3_tiny'
)
parser.add_argument('--trained_model',
                    default='weight/',
                    type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--conf_thresh',
                    default=0.1,
                    type=float,
                    help='Confidence threshold')
parser.add_argument('--nms_thresh',
                    default=0.50,
                    type=float,
                    help='NMS threshold')
# dataset
parser.add_argument('-d', '--dataset', default='voc', help='voc or coco')
# visualize
parser.add_argument('-vs',
                    '--visual_threshold',
                    default=0.25,
                    type=float,
                    help='Final confidence threshold')
parser.add_argument('--show',
                    action='store_true',
                    default=True,
                    help='show the visulization results.')

args = parser.parse_args()


def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)

    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1 - t_size[1]),
                      (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img,
                    label, (int(x1), int(y1 - 5)),
                    0,
                    text_scale, (0, 0, 0),
                    1,
                    lineType=cv2.LINE_AA)

    return img


def visualize(img,
              bboxes,
              scores,
              cls_inds,
              vis_thresh,
              class_colors,
              class_names,
              class_indexs=None,
              dataset_name='voc'):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_id = int(cls_inds[i])
            if dataset_name == 'coco':
                cls_color = class_colors[cls_id]
                cls_id = class_indexs[cls_id]
            else:
                cls_color = class_colors[cls_id]

            if len(class_names) > 1:
                mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            else:
                cls_color = [255, 0, 0]
                mess = None
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)

    return img


def test(net,
         device,
         dataset,
         transform,
         vis_thresh,
         class_colors=None,
         class_names=None,
         class_indexs=None,
         dataset_name='voc'):

    net.eval()

    num_images = len(dataset)
    num_images = 1
    save_path = os.path.join('det_results/', args.dataset, args.version)
    os.makedirs(save_path, exist_ok=True)

    grid_cell, all_anchor_wh = set_grid(416, anchor_size, device)

    with torch.no_grad():
        for index in range(num_images):
            print('Testing image {:d}/{:d}....'.format(index + 1, num_images))
            image, _ = dataset.pull_image(index)
            h, w, _ = image.shape
            scale = np.array([[w, h, w, h]])
            bboxess = []
            scoress = []
            cls_indss = []

            # to tensor
            x = torch.from_numpy(transform(image)[0][:, :,
                                                     (2, 1,
                                                      0)]).permute(2, 0, 1)
            x = x.unsqueeze(0).to(device)

            t0 = time.time()
            # forward
            preds = net(x)
            for ii, pred in enumerate(preds):
                conf_pred, cls_pred, reg_pred = divide(pred)
                bbox_pred = decode_boxes(reg_pred, grid_cell, all_anchor_wh)

                # score
                scores = torch.sigmoid(conf_pred[0]) * torch.softmax(
                    cls_pred[0], dim=-1)

                # normalize bbox
                bboxes = torch.clamp(bbox_pred[0] / image.shape[-1], 0., 1.)

                # to cpu

                scores = scores.to('cpu').numpy()
                bboxes = bboxes.to('cpu').numpy()

                # post-process
                bboxes, scores, cls_inds = postprocess(bboxes, scores)
                bboxess.append(bboxes)
                scoress.append(scores)
                cls_indss.append(cls_inds)

            print("detection time used ", time.time() - t0, "s")
            bboxes, scores, cls_inds = bboxess[4], scoress[4], cls_indss[4]

            # rescale
            bboxes *= scale

            # vis detection
            img_processed = visualize(img=image,
                                      bboxes=bboxes,
                                      scores=scores,
                                      cls_inds=cls_inds,
                                      vis_thresh=vis_thresh,
                                      class_colors=class_colors,
                                      class_names=class_names,
                                      class_indexs=class_indexs,
                                      dataset_name=dataset_name)
            if args.show:
                cv2.imshow('detection', img_processed)
                cv2.waitKey(0)
            # save result
            cv2.imwrite(os.path.join(save_path,
                                     str(index).zfill(6) + '.jpg'),
                        img_processed)


if __name__ == '__main__':
    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # input size
    input_size = args.input_size

    # dataset
    print('test on voc ...')
    data_dir = 'data/VOCdevkit'
    class_names = VOC_CLASSES
    class_indexs = None
    num_classes = 20
    dataset = VOCDetection(data_dir=data_dir, image_sets=[('2007', 'test')])

    class_colors = [(np.random.randint(255), np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # model
    model_name = args.version
    print('Model: ', model_name)

    cfg = config.yolov2_r50_cfg

    # build model
    anchor_size = cfg['anchor_size'] if args.dataset == 'voc' else cfg[
        'anchor_size_coco']
    net = YOLOv2R50(num_classes=num_classes)

    # load weight
    net.load_state_dict(
        torch.load('yolov2_r50_epoch_250_75.48.pth', map_location=device))
    net.to(device).eval()
    print('Finished loading model!')

    # evaluation
    test(net=net,
         device=device,
         dataset=dataset,
         transform=BaseTransform(input_size),
         vis_thresh=args.visual_threshold,
         class_colors=class_colors,
         class_names=class_names,
         class_indexs=class_indexs,
         dataset_name=args.dataset)
