"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import random
import threading
import time
import warnings

import keras

from ..utils.image import preprocess_image, resize_image, random_transform
from ..utils.anchors import anchor_targets_bbox
from ..utils.kp_utils import gaussian_multi_person, resize_image, transform_anns, height, width, out_height, out_width, \
    transform_mask


class GeneratorKeypoint(object):
    def __init__(
        self,
        image_data_generator,
        batch_size=1,
        group_method='ratio',  # one of 'none', 'random', 'ratio'
        shuffle_groups=True,
        seed=None
    ):
        self.image_data_generator = image_data_generator
        self.batch_size           = int(batch_size)
        self.group_method         = group_method
        self.shuffle_groups       = shuffle_groups
        if seed is None:
            seed = np.uint32((time.time() % 1)) * 1000
        np.random.seed(seed)

        self.group_index = 0
        self.lock        = threading.Lock()

        self.group_images()

    def size(self):
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        raise NotImplementedError('num_classes method not implemented')

    def name_to_label(self, name):
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_group(self, group):
        return [self.load_annotations(image_index) for image_index in group]

    def filter_annotations(self, image_group, annotations_group, group):
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            assert(isinstance(annotations, np.ndarray)), '\'load_annotations\' should return a list of numpy arrays, received: {}'.format(type(annotations))

            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations[:, 2] <= annotations[:, 0]) |
                (annotations[:, 3] <= annotations[:, 1]) |
                (annotations[:, 0] < 0) |
                (annotations[:, 1] < 0) |
                (annotations[:, 2] > image.shape[1]) |
                (annotations[:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    [annotations[invalid_index, :] for invalid_index in invalid_indices]
                ))
                annotations_group[index] = np.delete(annotations, invalid_indices, axis=0)

        return image_group, annotations_group

    def load_image_group(self, group):
        return [self.load_image(image_index) for image_index in group]

    def resize_image(self, image):
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_image(self, image):
        return preprocess_image(image)

    def preprocess_group(self, image_group, annotations_group, masks_group):
        _masks_group = np.zeros((len(masks_group),out_height, out_width, 17))
        for index, (image, annotations, masks) in enumerate(zip(image_group, annotations_group, masks_group)):
            # preprocess the image (subtract imagenet mean)
            # image = self.preprocess_image(image) # TODO: URGEEEEEEEEEEEEEEEEEEEEEEENT
            image, transformation = resize_image(img=image, anns=annotations)
            # apply resizing to annotations too
            annotations = transform_anns(annotations, transformation)

            mask_array = []
            for mask in masks:
                mask = transform_mask(mask, transformation)
                mask_array.append(mask)

            mask = np.maximum.reduce(mask_array)
            mask = np.expand_dims(mask, axis=-1)
            mask = np.repeat(mask, 17, axis=-1)


            # copy processed data back to group
            image_group[index]       = image
            annotations_group[index] = annotations
            _masks_group[index]   = mask
        return image_group, annotations_group, _masks_group

    def group_images(self):
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group, mask_group):
        # get the max image shape
        max_shape = (height,width,3)

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())
        mask_batch  = np.zeros((self.batch_size, out_height, out_width, 17), dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index] = image

        for mask_index, mask in enumerate(mask_group):
            mask_batch[mask_index] = mask

        return [image_batch, mask_batch]

    def anchor_targets(
        self,
        image_shape,
        boxes,
        num_classes,
        mask_shape=None,
        negative_overlap=0.4,
        positive_overlap=0.5,
        **kwargs
    ):
        return anchor_targets_bbox(image_shape, boxes, num_classes, mask_shape, negative_overlap, positive_overlap, **kwargs)

    def compute_targets(self, image_group, annotations_group):
        # get the max image shape
        max_shape = (height,width,3)

        # compute labels and regression targets
        labels_group     = [None] * self.batch_size
        regression_group = [None] * self.batch_size
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            labels_group[index], regression_group[index] = self.anchor_targets(max_shape, annotations, self.num_classes(), mask_shape=image.shape)

            # append anchor states to regression targets (necessary for filtering 'ignore', 'positive' and 'negative' anchors)
            anchor_states           = np.max(labels_group[index], axis=1, keepdims=True)
            regression_group[index] = np.append(regression_group[index], anchor_states, axis=1)

        labels_batch     = np.zeros((self.batch_size,) + labels_group[0].shape, dtype=keras.backend.floatx())
        regression_batch = np.zeros((self.batch_size,) + regression_group[0].shape, dtype=keras.backend.floatx())

        # copy all labels and regression values to the batch blob
        for index, (labels, regression) in enumerate(zip(labels_group, regression_group)):
            labels_batch[index, ...]     = labels
            regression_batch[index, ...] = regression

        # compute keypoint heatmap targets
        # keypoints_batch = []
        # keypoints_batch = np.zeros((self.batch_size, max_shape[0], max_shape[1], 17), dtype=keras.backend.floatx())
        keypoints_batch = np.zeros((self.batch_size, out_height, out_width, 17), dtype=keras.backend.floatx())
        for index, (image,annotations) in enumerate(zip(image_group, annotations_group)):
            x_scale = float(out_width) / width
            y_scale = float(out_height) / height
            heatmap = np.zeros((out_height,out_width,17), keras.backend.floatx())
            kp = annotations[:, 5:]
            kp_x = kp[:, 0::3]
            kp_y = kp[:, 1::3]
            kp_z = kp[:, 2::3]

            for i in range(annotations.shape[0]):
                kp_x = kp[i, 0::3]
                kp_y = kp[i, 1::3]
                kp_z = kp[i, 2::3]
                for joint_id, k in enumerate(zip(kp_x, kp_y, kp_z)):
                    if k[2] > 0:
                        x0 = int(round(k[0] * x_scale))
                        y0 = int(round(k[1] * y_scale))

                        if x0 >= out_width and y0 >= out_height:
                            heatmap[out_height - 1, out_width - 1, joint_id] = 1.
                        elif x0 >= out_width:
                            heatmap[y0, out_width - 1, joint_id] = 1.
                        elif y0 >= out_height:
                            heatmap[out_height - 1, x0, joint_id] = 1.
                        elif x0 < 0 and y0 < 0:
                            heatmap[0, 0, joint_id] = 1.
                        elif x0 < 0:
                            heatmap[y0, 0, joint_id] = 1.
                        elif y0 < 0:
                            heatmap[0, x0, joint_id] = 1.
                        else:
                            heatmap[y0, x0, joint_id] = 1.

            # Put gaussian around each keypoint
            heatmap = gaussian_multi_person(heatmap)
            # keypoints_batch.append(heatmap)
            keypoints_batch[index] = heatmap
        keypoints_batch = np.array(keypoints_batch, dtype=keras.backend.floatx())
        return [regression_batch, labels_batch, keypoints_batch]

    def compute_input_output(self, group):
        # load images and annotations
        image_group       = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)
        masks_group = [a[1] for a in annotations_group]
        annotations_group = [a[0] for a in annotations_group]
        for m in masks_group:
            print len(m)

        # annotations_group = [a[0] for a in annotations_group]
        # mask_group = a[1] for a in annotations_group]

        # check validity of annotations
        # image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # perform preprocessing steps
        image_group, annotations_group, masks_group = self.preprocess_group(image_group, annotations_group, masks_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group, masks_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def __next__(self):
        return self.next()

    def next(self):
        # advance the group index
        with self.lock:
            if self.group_index == 0 and self.shuffle_groups:
                # shuffle groups at start of epoch
                random.shuffle(self.groups)
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        return self.compute_input_output(group)
