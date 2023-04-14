import numpy as np
from data import *
import torch.nn as nn

import cv2
import numpy as np
from numpy import random


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image /= 255.
        image -= self.mean
        image /= self.std

        return image, boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=416):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            sample_id = np.random.randint(len(self.sample_options))
            mode = self.sample_options[sample_id]
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        # self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return im, boxes, labels
        # return self.rand_light_noise(im, boxes, labels)


class SSDAugmentation(object):
    def __init__(self, size=416):
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(size),
            Normalize(mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229))
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


class MSEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSEWithLogitsLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets, mask):
        inputs = torch.sigmoid(logits)

        # We ignore those whose tarhets == -1.0.
        pos_id = (mask == 1.0).float()
        neg_id = (mask == 0.0).float()
        pos_loss = pos_id * (inputs - targets)**2
        neg_loss = neg_id * (inputs)**2
        loss = 5.0 * pos_loss + 1.0 * neg_loss

        if self.reduction == 'mean':
            batch_size = logits.size(0)
            loss = torch.sum(loss) / batch_size

            return loss

        else:
            return loss


def compute_iou(anchor_boxes, gt_box):
    """
    Input:
        anchor_boxes : ndarray -> [[c_x_s, c_y_s, anchor_w, anchor_h], ..., [c_x_s, c_y_s, anchor_w, anchor_h]].
        gt_box : ndarray -> [c_x_s, c_y_s, anchor_w, anchor_h].
    Output:
        iou : ndarray -> [iou_1, iou_2, ..., iou_m], and m is equal to the number of anchor boxes.
    """
    # compute the iou between anchor box and gt box
    # First, change [c_x_s, c_y_s, anchor_w, anchor_h] ->  [xmin, ymin, xmax, ymax]
    # anchor box :
    ab_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    ab_x1y1_x2y2[:, 0] = anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2  # xmin
    ab_x1y1_x2y2[:, 1] = anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2  # ymin
    ab_x1y1_x2y2[:, 2] = anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2  # xmax
    ab_x1y1_x2y2[:, 3] = anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2  # ymax
    w_ab, h_ab = anchor_boxes[:, 2], anchor_boxes[:, 3]

    # gt_box :
    # We need to expand gt_box(ndarray) to the shape of anchor_boxes(ndarray), in order to compute IoU easily.
    gt_box_expand = np.repeat(gt_box, len(anchor_boxes), axis=0)

    gb_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    gb_x1y1_x2y2[:, 0] = gt_box_expand[:, 0] - gt_box_expand[:, 2] / 2  # xmin
    gb_x1y1_x2y2[:, 1] = gt_box_expand[:, 1] - gt_box_expand[:, 3] / 2  # ymin
    gb_x1y1_x2y2[:, 2] = gt_box_expand[:, 0] + gt_box_expand[:, 2] / 2  # xmax
    gb_x1y1_x2y2[:, 3] = gt_box_expand[:, 1] + gt_box_expand[:, 3] / 2  # ymin
    w_gt, h_gt = gt_box_expand[:, 2], gt_box_expand[:, 3]

    # Then we compute IoU between anchor_box and gt_box
    S_gt = w_gt * h_gt
    S_ab = w_ab * h_ab
    I_w = np.minimum(gb_x1y1_x2y2[:, 2], ab_x1y1_x2y2[:, 2]) - np.maximum(
        gb_x1y1_x2y2[:, 0], ab_x1y1_x2y2[:, 0])
    I_h = np.minimum(gb_x1y1_x2y2[:, 3], ab_x1y1_x2y2[:, 3]) - np.maximum(
        gb_x1y1_x2y2[:, 1], ab_x1y1_x2y2[:, 1])
    S_I = I_h * I_w
    U = S_gt + S_ab - S_I + 1e-20
    IoU = S_I / U

    return IoU


def set_anchors(anchor_size):
    """
    Input:
        anchor_size : list -> [[h_1, w_1], [h_2, w_2], ..., [h_n, w_n]].
    Output:
        anchor_boxes : ndarray -> [[0, 0, anchor_w, anchor_h],
                                   [0, 0, anchor_w, anchor_h],
                                   ...
                                   [0, 0, anchor_w, anchor_h]].
    """
    anchor_number = len(anchor_size)
    anchor_boxes = np.zeros([anchor_number, 4])
    for index, size in enumerate(anchor_size):
        anchor_w, anchor_h = size
        anchor_boxes[index] = np.array([0, 0, anchor_w, anchor_h])

    return anchor_boxes


def generate_txtytwth(gt_label, w, h, s, all_anchor_size):
    xmin, ymin, xmax, ymax = gt_label[:-1]
    # compute the center, width and height
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h

    if box_w < 1. or box_h < 1.:
        # print('A dirty data !!!')
        return False

    # map the center, width and height to the feature map size
    c_x_s = c_x / s
    c_y_s = c_y / s
    box_ws = box_w / s
    box_hs = box_h / s

    # the grid cell location
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    # generate anchor boxes
    anchor_boxes = set_anchors(all_anchor_size)
    gt_box = np.array([[0, 0, box_ws, box_hs]])
    # compute the IoU
    iou = compute_iou(anchor_boxes, gt_box)
    # We consider those anchor boxes whose IoU is more than ignore thresh,
    iou_mask = (iou > 0.5)

    result = []
    if iou_mask.sum() == 0:
        # We assign the anchor box with highest IoU score.
        index = np.argmax(iou)
        p_w, p_h = all_anchor_size[index]
        tx = c_x_s - grid_x
        ty = c_y_s - grid_y
        tw = np.log(box_ws / p_w)
        th = np.log(box_hs / p_h)
        weight = 2.0 - (box_w / w) * (box_h / h)

        result.append([
            index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax,
            ymax
        ])

        return result

    else:
        # There are more than one anchor boxes whose IoU are higher than ignore thresh.
        # But we only assign only one anchor box whose IoU is the best(objectness target is 1) and ignore other
        # anchor boxes whose(we set their objectness as -1 which means we will ignore them during computing obj loss )
        # iou_ = iou * iou_mask

        # We get the index of the best IoU
        best_index = np.argmax(iou)
        for index, iou_m in enumerate(iou_mask):
            if iou_m:
                if index == best_index:
                    p_w, p_h = all_anchor_size[index]
                    tx = c_x_s - grid_x
                    ty = c_y_s - grid_y
                    tw = np.log(box_ws / p_w)
                    th = np.log(box_hs / p_h)
                    weight = 2.0 - (box_w / w) * (box_h / h)

                    result.append([
                        index, grid_x, grid_y, tx, ty, tw, th, weight, xmin,
                        ymin, xmax, ymax
                    ])
                else:
                    # we ignore other anchor boxes even if their iou scores all higher than ignore thresh
                    result.append([
                        index, grid_x, grid_y, 0., 0., 0., 0., -1.0, 0., 0.,
                        0., 0.
                    ])

        return result


def gt_creator(input_size, stride, label_lists, anchor_size):
    """
    Input:
        input_size : list -> the size of image in the training stage.
        stride : int or list -> the downSample of the CNN, such as 32, 64 and so on.
        label_list : list -> [[[xmin, ymin, xmax, ymax, cls_ind], ... ], [[xmin, ymin, xmax, ymax, cls_ind], ... ]],  
                        and len(label_list) = batch_size;
                            len(label_list[i]) = the number of class instance in a image;
                            (xmin, ymin, xmax, ymax) : the coords of a bbox whose valus is between 0 and 1;
                            cls_ind : the corresponding class label.
    Output:
        gt_tensor : ndarray -> shape = [batch_size, anchor_number, 1+1+4, grid_cell number ]
    """

    # prepare the all empty gt datas
    batch_size = len(label_lists)
    h = w = input_size

    # We  make gt labels by anchor-free method and anchor-based method.
    ws = w // stride
    hs = h // stride
    s = stride

    # We use anchor boxes to build training target.
    all_anchor_size = anchor_size
    anchor_number = len(all_anchor_size)

    gt_tensor = np.zeros(
        [batch_size, hs, ws, anchor_number, 1 + 1 + 4 + 1 + 4])

    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            # get a bbox coords
            gt_class = int(gt_label[-1])
            results = generate_txtytwth(gt_label, w, h, s, all_anchor_size)
            if results:
                for result in results:
                    index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax = result
                    if weight > 0.:
                        if grid_y < gt_tensor.shape[
                                1] and grid_x < gt_tensor.shape[2]:
                            gt_tensor[batch_index, grid_y, grid_x, index,
                                      0] = 1.0
                            gt_tensor[batch_index, grid_y, grid_x, index,
                                      1] = gt_class
                            gt_tensor[batch_index, grid_y, grid_x, index,
                                      2:6] = np.array([tx, ty, tw, th])
                            gt_tensor[batch_index, grid_y, grid_x, index,
                                      6] = weight
                            gt_tensor[batch_index, grid_y, grid_x, index,
                                      7:] = np.array([xmin, ymin, xmax, ymax])
                    else:
                        gt_tensor[batch_index, grid_y, grid_x, index, 0] = -1.0
                        gt_tensor[batch_index, grid_y, grid_x, index, 6] = -1.0

    gt_tensor = gt_tensor.reshape(batch_size, hs * ws * anchor_number,
                                  1 + 1 + 4 + 1 + 4)

    return gt_tensor


def iou_score(bboxes_a, bboxes_b):
    """
        bbox_1 : [B*N, 4] = [x1, y1, x2, y2]
        bbox_2 : [B*N, 4] = [x1, y1, x2, y2]
    """
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en  # * ((tl < br).all())
    return area_i / (area_a + area_b - area_i + 1e-14)


def loss(pred_conf, pred_cls, pred_txtytwth, pred_iou, label):
    # loss func
    conf_loss_function = MSEWithLogitsLoss(reduction='mean')
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')
    iou_loss_function = nn.SmoothL1Loss(reduction='none')

    # pred
    pred_conf = pred_conf[:, :, 0]
    pred_cls = pred_cls.permute(0, 2, 1)
    pred_txty = pred_txtytwth[:, :, :2]
    pred_twth = pred_txtytwth[:, :, 2:]
    pred_iou = pred_iou[:, :, 0]

    # gt
    gt_conf = label[:, :, 0].float()
    gt_obj = label[:, :, 1].float()
    gt_cls = label[:, :, 2].long()
    gt_txty = label[:, :, 3:5].float()
    gt_twth = label[:, :, 5:7].float()
    gt_box_scale_weight = label[:, :, 7].float()
    gt_iou = (gt_box_scale_weight > 0.).float()
    gt_mask = (gt_box_scale_weight > 0.).float()

    batch_size = pred_conf.size(0)
    # objectness loss
    conf_loss = conf_loss_function(pred_conf, gt_conf, gt_obj)

    # class loss
    cls_loss = torch.sum(
        cls_loss_function(pred_cls, gt_cls) * gt_mask) / batch_size

    # box loss
    txty_loss = torch.sum(
        torch.sum(txty_loss_function(pred_txty, gt_txty), dim=-1) *
        gt_box_scale_weight * gt_mask) / batch_size
    twth_loss = torch.sum(
        torch.sum(twth_loss_function(pred_twth, gt_twth), dim=-1) *
        gt_box_scale_weight * gt_mask) / batch_size
    bbox_loss = txty_loss + twth_loss

    # iou loss
    iou_loss = torch.sum(
        iou_loss_function(pred_iou, gt_iou) * gt_mask) / batch_size

    return conf_loss, cls_loss, bbox_loss, iou_loss


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
        B, H * W*num_anchors, 4)

    return conf_pred, cls_pred, reg_pred
