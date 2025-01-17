import time
import torch
import numpy as np
import torchvision


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def py_nms(bboxes, iou_thres, score_thres):
    """
    reference: https://github.com/610265158/Peppa_Pig_Face_Landmark/blob/master/Skps/core/api/face_detector.py#L95
    """

    upper_thres = np.where(bboxes[:, 4] > score_thres)[0]

    bboxes = bboxes[upper_thres]

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    order = np.argsort(bboxes[:, 4])[::-1]

    keep = []

    while order.shape[0] > 0:
        cur = order[0]

        keep.append(cur)

        area = (bboxes[cur, 2] - bboxes[cur, 0]) * (bboxes[cur, 3] - bboxes[cur, 1])

        x1_reain = x1[order[1:]]
        y1_reain = y1[order[1:]]
        x2_reain = x2[order[1:]]
        y2_reain = y2[order[1:]]

        xx1 = np.maximum(bboxes[cur, 0], x1_reain)
        yy1 = np.maximum(bboxes[cur, 1], y1_reain)
        xx2 = np.minimum(bboxes[cur, 2], x2_reain)
        yy2 = np.minimum(bboxes[cur, 3], y2_reain)

        intersection = np.maximum(0, yy2 - yy1) * np.maximum(0, xx2 - xx1)

        iou = intersection / (area + (y2_reain - y1_reain) * (x2_reain - x1_reain) - intersection)

        ##keep the low iou
        low_iou_position = np.where(iou < iou_thres)[0]

        order = order[low_iou_position + 1]

    return bboxes[keep]