# config.py

# YOLOv2 with resnet-50
yolov2_r50_cfg = {
    # network
    'backbone': 'r50',
    # for multi-scale trick
    'train_size': 640,
    'val_size': 416,
    'random_size_range': [10, 19],
    # anchor size
    'anchor_size_voc': [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]],
    # train
    'lr_epoch': (150, 200),
    'max_epoch': 250,
    'ignore_thresh': 0.5
}
