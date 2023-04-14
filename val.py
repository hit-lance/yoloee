from __future__ import division
import os
import pickle
import time
from torch.autograd import Variable
import numpy as np

import torch

from data.voc0712 import VOC_CLASSES, VOCDetection
from data import config
from data import BaseTransform

import utils
from models.yoloee import YOLOEE
import xml.etree.ElementTree as ET


def val(model, val_size, anchor_size, device):
    model.eval()

    data_dir = 'data/VOCdevkit'
    labelmap = VOC_CLASSES
    set_type = 'test'
    year = '2007'
    display = False
    grid_cell, all_anchor_wh = utils.set_grid(val_size, anchor_size, device)

    # path
    devkit_path = data_dir + 'VOC' + year
    annopath = os.path.join(data_dir, 'VOC2007', 'Annotations', '%s.xml')
    imgsetpath = os.path.join(data_dir, 'VOC2007', 'ImageSets', 'Main',
                              set_type + '.txt')

    output_dir = get_output_dir('voc_eval/', set_type)

    # dataset
    dataset = VOCDetection(data_dir=data_dir,
                           image_sets=[('2007', set_type)],
                           transform=BaseTransform(val_size))

    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    # all_boxes = [[[] for _ in range(num_images)] for _ in range(len(labelmap))]
    all_boxes_list = []
    for _ in range(4):
        all_boxes_list.append([[[] for _ in range(num_images)]
                               for _ in range(len(labelmap))])

    # timers
    det_file = os.path.join(output_dir, 'detections.pkl')

    with torch.no_grad():
        for i in range(num_images):
            im, gt, h, w = dataset.pull_item(i)

            x = Variable(im.unsqueeze(0)).to(device)
            t0 = time.time()
            # forward
            preds = model(x)

            for ii, pred in enumerate(preds):
                conf_pred, cls_pred, reg_pred = utils.divide(pred)
                bbox_pred = utils.decode_boxes(reg_pred, grid_cell,
                                               all_anchor_wh)

                # score
                scores = torch.sigmoid(conf_pred[0]) * torch.softmax(
                    cls_pred[0], dim=-1)

                # normalize bbox
                bboxes = torch.clamp(bbox_pred[0] / im.shape[-1], 0., 1.)

                # to cpu
                scores = scores.to('cpu').numpy()
                bboxes = bboxes.to('cpu').numpy()

                # post-process
                bboxes, scores, cls_inds = postprocess(bboxes,
                                                       scores,
                                                       conf_thresh=0.2,
                                                       nms_thresh=0.4)

                detect_time = time.time() - t0
                scale = np.array([[w, h, w, h]])
                bboxes *= scale

                for j in range(len(labelmap)):
                    inds = np.where(cls_inds == j)[0]
                    if len(inds) == 0:
                        all_boxes_list[ii][j][i] = np.empty([0, 5],
                                                            dtype=np.float32)
                        continue
                    c_bboxes = bboxes[inds]
                    c_scores = scores[inds]
                    c_dets = np.hstack(
                        (c_bboxes, c_scores[:, np.newaxis])).astype(np.float32,
                                                                    copy=False)
                    all_boxes_list[ii][j][i] = c_dets

                all_boxes_list[ii].append(all_boxes_list[ii][j][i])
                if i % 500 == 0:
                    print('im_detect: {:d}/{:d} {:.3f}s'.format(
                        i + 1, num_images, detect_time))

    print('Evaluating detections')
    mAPs = []
    for i, all_boxes in enumerate(all_boxes_list):
        os.path.join(output_dir, 'exit' + str(i + 1), 'detections.pkl')
        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        mAP = evaluate_detections(all_boxes, labelmap, display, dataset,
                                  devkit_path, output_dir, set_type,
                                  imgsetpath, annopath)
        mAPs.append(mAP)
        print('Mean AP: ', mAP)

    return mAPs


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a modelwork
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(cls, set_type, devkit_path):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + set_type + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, labelmap, display, dataset, set_type,
                           devkit_path):
    for cls_ind, cls in enumerate(labelmap):
        if display:
            print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(cls, set_type, devkit_path)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind][im_ind]
                if len(dets) == 0:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                        index[1], dets[k, -1], dets[k, 0] + 1, dets[k, 1] + 1,
                        dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(devkit_path,
                   output_dir,
                   labelmap,
                   display,
                   set_type,
                   imgsetpath,
                   annopath,
                   use_07=True):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(cls, set_type, devkit_path)
        rec, prec, ap = voc_eval(detpath=filename,
                                 classname=cls,
                                 cachedir=cachedir,
                                 imgsetpath=imgsetpath,
                                 display=display,
                                 annopath=annopath,
                                 ovthresh=0.5,
                                 use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    if display:
        mAP = np.mean(aps)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('--------------------------------------------------------------')
    else:
        mAP = np.mean(aps)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
    return mAP


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             classname,
             cachedir,
             imgsetpath,
             display,
             annopath,
             ovthresh=0.5,
             use_07_metric=True):
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imgsetpath, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0 and display:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        if display:
            print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {
            'bbox': bbox,
            'difficult': difficult,
            'det': det
        }

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) * (BBGT[:, 3] - BBGT[:, 1]) -
                       inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid utils.divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def evaluate_detections(box_list, labelmap, display, dataset, devkit_path,
                        output_dir, set_type, imgsetpath, annopath):
    write_voc_results_file(box_list, labelmap, display, dataset, set_type,
                           devkit_path)
    return do_python_eval(devkit_path, output_dir, labelmap, display, set_type,
                          imgsetpath, annopath)


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [
            int(bbox.find('xmin').text),
            int(bbox.find('ymin').text),
            int(bbox.find('xmax').text),
            int(bbox.find('ymax').text)
        ]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


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


def postprocess(bboxes,
                scores,
                num_classes=20,
                conf_thresh=0.2,
                nms_thresh=0.5):
    """
    bboxes: (HxW, 4), bsize = 1
    scores: (HxW, num_classes), bsize = 1
    """

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
    scores = scores[keep]
    cls_inds = cls_inds[keep]

    return bboxes, scores, cls_inds


if __name__ == '__main__':
    # build model
    device = torch.device("cpu")

    model = YOLOEE().to(device)
    model.load_state_dict(torch.load('yoloee.pth', map_location=device))

    cfg = config.yolov2_r50_cfg

    mAPs = val(model, cfg['val_size'], cfg['anchor_size'], device)
    print(mAPs)
