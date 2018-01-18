import keras
import numpy as np
from pycocotools.coco import COCO
from random import shuffle
import os
import cv2
from random import uniform
import tensorflow
from skimage.filters import gaussian

sigma = 1
width = 640
height = 512
out_width = width / 8
out_height = height / 8
pad_val = [103.939, 116.779, 123.68]
interpolation = cv2.INTER_CUBIC


def gaussian_multi_person(inp):
    '''
    :param inp: Multi person ground truth heatmap input (17 ch) Each channel contains multiple joints.
    :return: out: Gaussian augmented output. Values are between 0. and 1.
    '''
    h,w,ch = inp.shape
    out = np.zeros_like(inp)
    for i in range(ch):
        layer = inp[:,:,i]
        ind = np.argwhere(layer == 1)
        b = []
        if len(ind) > 0:
            for j in ind:
                t = np.zeros((h, w))
                t[j[0], j[1]] = 1
                t = gaussian(t, sigma=sigma, mode='constant')
                t = t * (1 / t.max())
                b.append(t)

            out[:, :, i] = np.maximum.reduce(b)
        else:
            out[:, :, i] = np.zeros((h, w))
    return out


def resize_image(img, anns, width=640, height=512):
    tr_x, tr_y, tr_p_x, tr_p_y, tr_p_x1, tr_p_y1 = 0, 0, 0, 0, 0, 0
    scale = 1.

    (h, w, _) = img.shape

    h = float(h)
    w = float(w)
    height = float(height)
    width = float(width)
    ratio = height / width

    if h / w == ratio:
        scale = width / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)
    elif h / w < ratio:
        scale = width / w
        new_h = round(h * scale)
        h_pad = height - new_h

        if h_pad % 2 == 0:
            top = h_pad / 2
            bottom = h_pad / 2
            left = 0
            right = 0
        else:
            top = h_pad // 2 + 1
            bottom = h_pad // 2
            left = 0
            right = 0

        tr_y = top

        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)
        img = cv2.copyMakeBorder(img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,
                                 value=pad_val)

    elif h / w > ratio:
        person_bbox = get_person_bbox(anns, w, h)

        p_width, p_height = person_bbox[2] - person_bbox[0], person_bbox[3] - person_bbox[1]
        p_width = float(p_width)
        p_height = float(p_height)
        p_ratio = p_height / p_width

        if p_ratio == ratio:
            tr_person = person_bbox[1]
            img = img[person_bbox[1]:person_bbox[3], person_bbox[0]:person_bbox[2], :]

        elif p_ratio < ratio:
            scale = width / p_width
            new_h = round(p_height * scale)
            h_pad = height - new_h

            if h_pad % 2 == 0:
                top = h_pad / 2
                bottom = h_pad / 2
                left = 0
                right = 0
            else:
                top = h_pad // 2 + 1
                bottom = h_pad // 2
                left = 0
                right = 0

            tr_y = top  # - person_bbox[0]
            tr_p_x = person_bbox[0]
            tr_p_y = person_bbox[1]
            tr_p_x1 = person_bbox[2]
            tr_p_y1 = person_bbox[3]
            img = img[person_bbox[1]:person_bbox[3], person_bbox[0]:person_bbox[2], :]
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)
            img = cv2.copyMakeBorder(img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,
                                     value=pad_val)

        elif p_ratio > ratio:
            scale = height / p_height
            new_w = round(p_width * scale)
            w_pad = width - new_w

            if w_pad % 2 == 0:
                top = 0
                bottom = 0
                left = w_pad / 2
                right = w_pad / 2
            else:
                top = 0
                bottom = 0
                left = w_pad // 2 + 1
                right = w_pad // 2

            tr_x = left
            tr_p_x = person_bbox[0]
            tr_p_y = person_bbox[1]
            tr_p_x1 = person_bbox[2]
            tr_p_y1 = person_bbox[3]
            img = img[person_bbox[1]:person_bbox[3], person_bbox[0]:person_bbox[2], :]
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)
            img = cv2.copyMakeBorder(img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,
                                     value=pad_val)

    return img, [scale, tr_x, tr_y, tr_p_x, tr_p_y, tr_p_x1, tr_p_y1]


def transform_anns(anns, transformation):
    scale, tr_x, tr_y, tr_p_x, tr_p_y, _, _ = transformation

    anns[:, 0] = (anns[:, 0] - tr_p_x) * scale + tr_x
    anns[:, 2] = (anns[:, 2] - tr_p_x) * scale + tr_x
    anns[:, 1] = (anns[:, 1] - tr_p_y) * scale + tr_y
    anns[:, 3] = (anns[:, 3] - tr_p_y) * scale + tr_y

    # Rescale kp annotations to fixed size
    kp = anns[:, 5:]  # *= image_scale
    kp[:, 0::3] = (kp[:, 0::3] - tr_p_x) * scale + tr_x
    kp[:, 1::3] = (kp[:, 1::3] - tr_p_y) * scale + tr_y
    anns[:, 5:] = kp
    return anns


def get_person_bbox(anns, width, height, extend=.1):
    # get bboxes in x1, y1, x2, y2
    bboxes = np.array([(a[0], a[1], a[0] + a[2], a[1] + a[3]) for a in anns])
    w, h = (bboxes[:, 2].max() - bboxes[:, 0].min()), (bboxes[:, 3].max() - bboxes[:, 1].min())

    # extend bbox
    bbox = [bboxes[:, 0].min() - w * extend, bboxes[:, 1].min() - h * extend,
            bboxes[:, 2].max() + w * extend, bboxes[:, 3].max() + h * extend]

    if bbox[0] < 0: bbox[0] = 0
    if bbox[1] < 0: bbox[1] = 0
    if bbox[2] >= width: bbox[2] = width - 1
    if bbox[3] >= height: bbox[3] = height - 1

    bbox = [int(round(b)) for b in bbox]
    return bbox

def transform_mask(mask, transformation):
    scale, tr_x, tr_y, tr_p_x, tr_p_y, tr_p_x1, tr_p_y1 = transformation

    print transformation # (tr_p_x, tr_p_y, tr_p_x1, tr_p_y1)

    if (tr_p_x, tr_p_y, tr_p_x1, tr_p_y1) != (0, 0, 0, 0):
        mask = mask[tr_p_y:tr_p_y1,tr_p_x:tr_p_x1]

    mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=interpolation)

    top = tr_y
    bottom = height - top - mask.shape[0]

    left = tr_x
    right = width - left - mask.shape[1]

    mask = cv2.copyMakeBorder(mask, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,
                             value=[0])
    mask = cv2.resize(mask, (out_width, out_height), interpolation=interpolation)

    return mask
