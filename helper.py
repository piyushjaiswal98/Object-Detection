# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:40:50 2018

@author: Piyushjaiswal
"""

import tensorflow as tf
from cifar10 import num_channels

img_size_cropped = 24

def pre_process_image(image, training):
    
    if training:
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
        
    else: # For Testing

        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)

    return image

# Loops Over the previous function for all the images
    
def pre_process(images, training):
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)

    return images