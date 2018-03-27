# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:49:13 2018

@author: Piyushjaiswal
"""

import numpy as np


def one_hot_encoded(class_numbers, num_classes=None):

    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]



