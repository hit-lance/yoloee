import torch
import torch.backends.cudnn as cudnn
from data.voc0712 import VOC_CLASSES, VOCDetection
from data import config, BaseTransform
import numpy as np
import cv2

from models.yolov2_r50 import YOLOv2R50
from utils import divide
from utils.anchor import decode_boxes, set_grid
from val import postprocess


def draw_bbox_labels(image, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), cls_color, 2)

    if label is not None:
        # plot title bbox
        cv2.rectangle(image, (x1, y2 - t_size[1] // 2),
                      (int(x1 + t_size[0] * text_scale), y2), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(image,
                    label, (int(x1), int(y2)),
                    0,
                    text_scale, (0, 0, 0),
                    1,
                    lineType=cv2.LINE_AA)

    return image


def draw_bbox(image, bboxes, scores, cls_inds, vis_thresh, class_colors,
              class_names):
    image = image.copy()
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_id = int(cls_inds[i])
            cls_color = class_colors[cls_id]
            if len(class_names) > 1:
                mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            else:
                cls_color = [255, 0, 0]
                mess = None
            image = draw_bbox_labels(image,
                                     bbox,
                                     mess,
                                     cls_color,
                                     text_scale=ts)

    return image


def visualize(model,
              device,
              dataset,
              transform,
              vis_thresh,
              class_colors=None,
              class_names=None):

    model.eval()

    num_images = len(dataset)
    # num_images = 5

    grid_cell, all_anchor_wh = set_grid(416, anchor_size, device)

    border_size = 40
    border_color = (255, 255, 255)  # white
    vertical_border_size = 20
    vertical_border_color = (255, 255, 255)  # white

    titles = ["Exit 1", "Exit 2", "Exit 3", "Exit 4"]

    with torch.no_grad():
        for index in range(num_images):
            print('Testing image {:d}/{:d}....'.format(index + 1, num_images))
            image, _ = dataset.pull_image(index)
            h, w, _ = image.shape
            scale = np.array([[w, h, w, h]])

            # to tensor
            x = torch.from_numpy(transform(image)[0][:, :,
                                                     (2, 1,
                                                      0)]).permute(2, 0, 1)
            x = x.unsqueeze(0).to(device)

            # forward
            preds = model(x)

            img_processed_list = []

            for i, pred in enumerate(preds):
                conf_pred, cls_pred, reg_pred = divide(pred)
                bbox_pred = decode_boxes(reg_pred, grid_cell, all_anchor_wh)

                # score
                scores = torch.sigmoid(conf_pred[0]) * torch.softmax(
                    cls_pred[0], dim=-1)

                # normalize bbox
                bboxes = torch.clamp(bbox_pred[0] / 416, 0., 1.)

                # to cpu
                scores = scores.to('cpu').numpy()
                bboxes = bboxes.to('cpu').numpy()

                # post-process
                bboxes, scores, cls_inds = postprocess(bboxes,
                                                       scores,
                                                       conf_thresh=vis_thresh)

                # rescale
                bboxes *= scale

                # vis detection
                img_processed = draw_bbox(image=image,
                                          bboxes=bboxes,
                                          scores=scores,
                                          cls_inds=cls_inds,
                                          vis_thresh=vis_thresh,
                                          class_colors=class_colors,
                                          class_names=class_names)

                img_processed = cv2.copyMakeBorder(img_processed,
                                                   0,
                                                   border_size,
                                                   border_size,
                                                   border_size,
                                                   cv2.BORDER_CONSTANT,
                                                   value=border_color)

                # Define the font properties
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.8
                color = (0, 0, 0)
                thickness = 2

                # Get the size of the text
                (textWidth,
                 textHeight), _ = cv2.getTextSize(titles[i], font, fontScale,
                                                  thickness)

                # Calculate the position of the text
                x = int((img_processed.shape[1] - textWidth) / 2)
                y = img_processed.shape[
                    0] + textHeight - 30  # Set the y-coordinate to be the height of the image plus the height of the text plus some padding

                # Add the text to the image
                cv2.putText(img_processed, titles[i], (x, y), font, fontScale,
                            color, thickness)

                img_processed_list.append(img_processed)

            horizontal1 = np.concatenate(
                (img_processed_list[0], img_processed_list[1]), axis=1)
            horizontal2 = np.concatenate(
                (img_processed_list[2], img_processed_list[3]), axis=1)

            # Add a border between the two rows of images
            vertical_border = np.full(
                (vertical_border_size, horizontal1.shape[1], 3),
                vertical_border_color,
                dtype=np.uint8)
            final_image = np.concatenate(
                (horizontal1, vertical_border, horizontal2), axis=0)

            # Display the final image
            cv2.imshow("Final Image", final_image)
            cv2.waitKey(0)


if __name__ == '__main__':
    # cuda
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # input size
    input_size = 416
    visual_threshold = 0.2
    # dataset
    print('test on voc ...')
    data_dir = 'data/VOCdevkit'
    class_names = VOC_CLASSES
    class_indexs = None
    num_classes = 20
    dataset = VOCDetection(data_dir=data_dir, image_sets=[('2007', 'test')])

    class_colors = [(np.random.randint(255), np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    cfg = config.yolov2_r50_cfg

    # load model
    anchor_size = cfg['anchor_size']
    model = YOLOv2R50(num_classes=num_classes)
    model.load_state_dict(torch.load('yoloee.pth', map_location=device))
    model.to(device)

    # evaluation
    visualize(model=model,
              device=device,
              dataset=dataset,
              transform=BaseTransform(input_size),
              vis_thresh=visual_threshold,
              class_colors=class_colors,
              class_names=class_names)
