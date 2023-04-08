from __future__ import division

import os
import random
import argparse
import time
import cv2
import numpy as np

import torch
import torch.optim as optim

from data.voc0712 import VOCDetection
from data import config
from data import BaseTransform, detection_collate

import tools
from utils.anchor import decode_boxes, set_grid

from utils.augmentations import SSDAugmentation
from utils.vocapi_evaluator import VOCAPIEvaluator
from utils.modules import ModelEMA

from models.yolov2_r50 import YOLOv2R50


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    # basic
    parser.add_argument('--cuda',
                        action='store_true',
                        default=False,
                        help='use cuda.')
    parser.add_argument('-bs',
                        '--batch_size',
                        default=16,
                        type=int,
                        help='Batch size for training')
    parser.add_argument('--lr',
                        default=1e-3,
                        type=float,
                        help='initial learning rate')
    parser.add_argument('--wp_epoch',
                        type=int,
                        default=2,
                        help='The upper bound of warm-up')
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay',
                        default=5e-4,
                        type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--num_workers',
                        default=8,
                        type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--num_gpu',
                        default=1,
                        type=int,
                        help='Number of GPUs to train')
    parser.add_argument('--eval_epoch',
                        type=int,
                        default=10,
                        help='interval between evaluations')
    parser.add_argument('--tfboard',
                        action='store_true',
                        default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder',
                        default='weights/',
                        type=str,
                        help='Gamma update for SGD')
    parser.add_argument('--vis',
                        action='store_true',
                        default=False,
                        help='visualize target.')

    # model
    parser.add_argument('-v', '--version', default='yolov2_r50')

    # train trick
    parser.add_argument('--no_warmup',
                        action='store_true',
                        default=False,
                        help='do not use warmup')
    parser.add_argument('-ms',
                        '--multi_scale',
                        action='store_true',
                        default=False,
                        help='use multi-scale trick')
    parser.add_argument('--ema',
                        action='store_true',
                        default=False,
                        help='use ema training trick')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cfg = config.yolov2_r50_cfg

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # multi-scale
    if args.multi_scale:
        print('use the multi-scale trick ...')
        train_size = cfg['train_size']
        val_size = cfg['val_size']
    else:
        train_size = val_size = cfg['train_size']

    # Model ENA
    if args.ema:
        print('use EMA trick ...')

    # dataset and evaluator
    data_dir = 'data/VOCdevkit'
    num_classes = 20
    dataset = VOCDetection(data_dir=data_dir,
                           transform=SSDAugmentation(train_size))

    print('Training model on:', dataset.name)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    # build model
    anchor_size = cfg['anchor_size_voc']
    net = YOLOv2R50(num_classes=num_classes)
    model = net
    model = model.to(device).train()

    grid_cell, all_anchor_wh = set_grid(train_size, anchor_size, device)

    batch_size = args.batch_size

    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             shuffle=True,
                                             batch_size=batch_size,
                                             collate_fn=detection_collate,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             drop_last=True)

    # EMA
    ema = ModelEMA(model) if args.ema else None

    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',
                               time.localtime(time.time()))
        log_path = os.path.join('log/', c_time)
        os.makedirs(log_path, exist_ok=True)

        tblogger = SummaryWriter(log_path)

    # optimizer setup
    base_lr = (args.lr / 16) * batch_size
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(),
                          lr=base_lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    max_epoch = cfg['max_epoch']
    epoch_size = len(dataloader)
    best_map = -1.
    warmup = not args.no_warmup

    t0 = time.time()
    # start training loop
    for epoch in range(max_epoch):

        # use step lr
        if epoch in cfg['lr_epoch']:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)

        for iter_i, (images, targets) in enumerate(dataloader):
            # WarmUp strategy for learning rate
            ni = iter_i + epoch * epoch_size
            # warmup
            if epoch < args.wp_epoch and warmup:
                nw = args.wp_epoch * epoch_size
                tmp_lr = base_lr * pow(ni / nw, 4)
                set_lr(optimizer, tmp_lr)

            elif epoch == args.wp_epoch and iter_i == 0 and warmup:
                # warmup is over
                warmup = False
                tmp_lr = base_lr
                set_lr(optimizer, tmp_lr)

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                r = cfg['random_size_range']
                train_size = random.randint(r[0], r[1]) * 32
                grid_cell, all_anchor_wh = set_grid(train_size, anchor_size,
                                                    device)

            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images,
                                                         size=train_size,
                                                         mode='bilinear',
                                                         align_corners=False)

            targets = [label.tolist() for label in targets]
            # visualize labels
            if args.vis:
                vis_data(images, targets, train_size)
                continue

            # label assignment
            targets = tools.gt_creator(input_size=train_size,
                                       stride=32,
                                       label_lists=targets,
                                       anchor_size=anchor_size)

            # to device
            images = images.float().to(device)
            targets = torch.tensor(targets).float().to(device)

            # forward
            conf_pred, cls_pred, reg_pred = model(images)

            box_pred = decode_boxes(reg_pred, grid_cell, all_anchor_wh)

            x1y1x2y2_pred = (box_pred / train_size).view(-1, 4)
            x1y1x2y2_gt = targets[:, :, 7:].view(-1, 4)
            reg_pred = reg_pred.view(batch_size, -1, 4)

            # set conf target
            iou_pred = tools.iou_score(x1y1x2y2_pred,
                                       x1y1x2y2_gt).view(batch_size, -1, 1)
            gt_conf = iou_pred.clone().detach()

            # [obj, cls, txtytwth, x1y1x2y2] -> [conf, obj, cls, txtytwth]
            targets = torch.cat([gt_conf, targets[:, :, :7]], dim=2)
            conf_loss, cls_loss, box_loss, iou_loss = tools.loss(
                pred_conf=conf_pred,
                pred_cls=cls_pred,
                pred_txtytwth=reg_pred,
                pred_iou=iou_pred,
                label=targets)

            # compute loss
            total_loss = conf_loss + cls_loss + box_loss + iou_loss

            loss_dict = dict(conf_loss=conf_loss,
                             cls_loss=cls_loss,
                             box_loss=box_loss,
                             iou_loss=iou_loss,
                             total_loss=total_loss)

            loss_dict_reduced = loss_dict

            # backprop
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # ema
            if args.ema:
                ema.update(model)

            # display
            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    tblogger.add_scalar('conf loss',
                                        loss_dict_reduced['conf_loss'].item(),
                                        iter_i + epoch * epoch_size)
                    tblogger.add_scalar('cls loss',
                                        loss_dict_reduced['cls_loss'].item(),
                                        iter_i + epoch * epoch_size)
                    tblogger.add_scalar('box loss',
                                        loss_dict_reduced['box_loss'].item(),
                                        iter_i + epoch * epoch_size)
                    tblogger.add_scalar('iou loss',
                                        loss_dict_reduced['iou_loss'].item(),
                                        iter_i + epoch * epoch_size)

                t1 = time.time()
                cur_lr = [
                    param_group['lr'] for param_group in optimizer.param_groups
                ]
                # basic infor
                log = '[Epoch: {}/{}]'.format(epoch + 1, max_epoch)
                log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
                log += '[lr: {:.6f}]'.format(cur_lr[0])
                # loss infor
                for k in loss_dict_reduced.keys():
                    log += '[{}: {:.2f}]'.format(k, loss_dict[k])

                # other infor
                log += '[time: {:.2f}]'.format(t1 - t0)
                log += '[size: {}]'.format(train_size)

                # print log infor
                print(log, flush=True)

                t0 = time.time()

        # evaluation
        evaluator = VOCAPIEvaluator(data_root=data_dir,
                                    img_size=val_size,
                                    anchor_size=anchor_size,
                                    device=device,
                                    transform=BaseTransform(val_size))

        if (epoch % args.eval_epoch) == 0 or (epoch == max_epoch - 1):
            if args.ema:
                model_eval = ema.ema
            else:
                model_eval = model

            print('eval ...')

            # evaluate
            evaluator.evaluate(model_eval)

            cur_map = evaluator.map
            if cur_map > best_map:
                # update best-map
                best_map = cur_map
                # save model
                print('Saving state, epoch:', epoch + 1)
                weight_name = '{}_epoch_{}_{:.2f}.pth'.format(
                    args.version, epoch + 1, best_map * 100)
                checkpoint_path = os.path.join(path_to_save, weight_name)
                torch.save(model_eval.state_dict(), checkpoint_path)

            if args.tfboard:
                tblogger.add_scalar('07test/mAP', evaluator.map, epoch)

            model_eval.train()

    if args.tfboard:
        tblogger.close()


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def vis_data(images, targets, input_size):
    # vis data
    mean = (0.406, 0.456, 0.485)
    std = (0.225, 0.224, 0.229)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    img = images[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    img = ((img * std + mean) * 255).astype(np.uint8)
    img = img.copy()

    for box in targets[0]:
        xmin, ymin, xmax, ymax = box[:-1]
        # print(xmin, ymin, xmax, ymax)
        xmin *= input_size
        ymin *= input_size
        xmax *= input_size
        ymax *= input_size
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                      (0, 0, 255), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    train()
