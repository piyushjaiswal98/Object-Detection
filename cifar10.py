# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:46:58 2018

@author: Piyushjaiswal
"""

import numpy as np
import pickle
import os
from one import one_hot_encoded


img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
num_classes = 10
_num_files_train = 5
_images_per_file = 10000
_num_images_train = _num_files_train * _images_per_file


def _get_file_path(filename=""):

    return os.path.join("cifar-10-batches-py/", filename)


def _unpickle(filename):
    file_path = _get_file_path(filename)
    print("Loading data: " + file_path)
    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    return data


def _convert_images(raw):
     raw_float = np.array(raw, dtype=float) / 255.0
     images = raw_float.reshape([-1, num_channels, img_size, img_size])
     images = images.transpose([0, 2, 3, 1])

     return images


def _load_data(filename):

    data = _unpickle(filename)
    raw_images = data[b'data']
    cls = np.array(data[b'labels'])
    images = _convert_images(raw_images)

    return images, cls


def load_class_names():
   
    raw = _unpickle(filename="batches.meta")[b'label_names']
    names = [x.decode('utf-8') for x in raw]
    return names


def load_training_data():

    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    begin = 0

    for i in range(_num_files_train):
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))
        num_images = len(images_batch)
        end = begin + num_images
        images[begin:end, :] = images_batch
        cls[begin:end] = cls_batch
        begin = end

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)


def load_test_data():

    images, cls = _load_data(filename="test_batch")
    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)
