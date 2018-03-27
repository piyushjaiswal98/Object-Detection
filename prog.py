# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:58:02 2018

@author: Piyushjaiswal
"""

import cifar10
from cifar10 import img_size, num_channels, num_classes
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import prettytensor as pt
from image_formation import plot_images

from helper import pre_process


class_names = cifar10.load_class_names()
images_train, cls_train,labels_train = cifar10.load_training_data()
images_test, cls_test,labels_test = cifar10.load_test_data()

print("size of:")
print (" - Training set:\t\t{}",format(len(images_train)))
print (" - Test set:\t\t{}",format(len(images_test)))


# Testing for sample images from the dataset by plotting them

images = images_test[36:45]
cls_true = cls_test[36:45]
plot_images(images = images,cls_true = cls_true, smooth = True)



# Setting the Tensorflow placeholder variables to serve as inputs for tensorflow computational graphs

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)



# For Distorted images

distorted_images = pre_process(images=x, training=True)



# Forming the Neural Network

def main_network(images, training):

    x_pretty = pt.wrap(images)
    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer

    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty.\
            conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=64, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=256, name='layer_fc1').\
            fully_connected(size=128, name='layer_fc2').\
            softmax_classifier(num_classes=num_classes, labels=y_true)

    return y_pred, loss


def create_network(training):

    with tf.variable_scope('network', reuse= not training):

        images = x
        images = pre_process(images=images, training=training)
        y_pred, loss = main_network(images=images, training=training)

    return y_pred, loss


# Setting Variable, Loss Function and The Optimizer
    
global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)

_, loss = create_network(training=True)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)



# Creating Network for Testing Phase

y_pred, _ = create_network(training=False)
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# Saving the variables to be used in the future

saver = tf.train.Saver()



# Getting the weights

def get_weights(layer_name):

    with tf.variable_scope("network/" + layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable

weights_conv1 = get_weights(layer_name='layer_conv1')
weights_conv2 = get_weights(layer_name='layer_conv2')



# Getting layer outputs

def get_layer_output(layer_name):
    
    tensor_name = "network/" + layer_name + "/Relu:0"
    tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)

    return tensor

output_conv1 = get_layer_output(layer_name='layer_conv1')
output_conv2 = get_layer_output(layer_name='layer_conv2')


# Session creation

session = tf.Session()

save_dir = 'checkpoints/'   

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
save_path = os.path.join(save_dir, 'cifar10_cnn')

try:
    print("Trying to restore last checkpoint ...")
    last_check_point = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
    saver.restore(session, save_path=last_check_point)
    print("Restored checkpoint from:", last_check_point)
    
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    session.run(tf.global_variables_initializer())
    
    
    
# Creating random training batches
    
train_batch_size = 64

def random_batch():
    
    num_images = len(images_train)
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch



# Performing Optimization
    

def optimize(num_iterations):

    start_time = time.time()

    for i in range(num_iterations):
        x_batch, y_true_batch = random_batch()
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        
        iteration, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        if (iteration % 100 == 0) or (i == num_iterations - 1):
            batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)

            msg = "Iterations: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(iteration, batch_acc))

        if (iteration % 1000 == 0) or (i == num_iterations - 1):
            saver.save(session,
                       save_path=save_path,
                       global_step=global_step)

            print("Saved checkpoint.")

    end_time = time.time()
    time_dif = end_time - start_time

    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    
# Example errors
    
def plot_example_errors(cls_pred, correct):

    incorrect = (correct == False)
    images = images_test[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = cls_test[incorrect]
    
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])
    
    
    
# Confusion matrix

def plot_confusion_matrix(cls_pred):

    cm = confusion_matrix(y_true=cls_test,  # True class.
                          y_pred=cls_pred)  # Predicted class.

    for i in range(num_classes):
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))



# Calculating Classifications

batch_size = 256

def predict_cls(images, labels, cls_true):
    num_images = len(images)

    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    i = 0

    while i < num_images:
        j = min(i + batch_size, num_images)
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}
        
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        
        i = j

    correct = (cls_true == cls_pred)

    return correct, cls_pred


# Calculating the predicted class for the test set
    
def predict_cls_test():
    return predict_cls(images = images_test,
                       labels = labels_test,
                       cls_true = cls_test)


# Classification Accuracy
    
def classification_accuracy(correct):

    return correct.mean(), correct.sum()


# Showing Performance

def print_test_accuracy():

    correct, cls_pred = predict_cls(images = images_test,
                       labels = labels_test,
                       cls_true = cls_test)

    acc, num_correct = classification_accuracy(correct)
    num_images = len(correct)
    print("\n\n")
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

     # Plot the confusion matrix,
    print("\n\nConfusion Matrix:\n")
    plot_confusion_matrix(cls_pred=cls_pred)
    
    
    print("\n\nDo you want to plot some example errors\n")
    print(" Y : yes or N: no")
    ex = str(input())
    if (ex == "Y" or ex == "y"):
            print("\nExample errors:\n")
            plot_example_errors(cls_pred=cls_pred, correct=correct)




# Plotting convolutional weights
        
def plot_conv_weights(weights, input_channel=0):

    w = session.run(weights)

    print("Min:  {0:.5f}, Max:   {1:.5f}".format(w.min(), w.max()))
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))
    
    w_min = np.min(w)
    w_max = np.max(w)
    abs_max = max(abs(w_min), abs(w_max))

    num_filters = w.shape[3]

    num_grids = math.ceil(math.sqrt(num_filters))

    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = w[:, :, input_channel, i]
            ax.imshow(img, vmin=-abs_max, vmax=abs_max,
                      interpolation='nearest', cmap='seismic')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()
    
    

# Plotting output of convolutional layers

def plot_layer_output(layer_output, image):

    feed_dict = {x: [image]}
    
    values = session.run(layer_output, feed_dict=feed_dict)

    values_min = np.min(values)
    values_max = np.max(values)

    num_images = values.shape[3]

    num_grids = math.ceil(math.sqrt(num_images))
    
    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i<num_images:
            img = values[0, :, :, i]
            ax.imshow(img, vmin=values_min, vmax=values_max,
                      interpolation='nearest', cmap='binary')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()
    

# Examples of Distorted Input Images
    
def plot_distorted_image(image, cls_true):
    
    image_duplicates = np.repeat(image[np.newaxis, :, :, :], 9, axis=0)
    feed_dict = {x: image_duplicates}
    result = session.run(distorted_images, feed_dict=feed_dict)
    plot_images(images=result, cls_true=np.repeat(cls_true, 9))


def get_test_image(i):
    return images_test[i, :, :, :], cls_test[i]

img, cls = get_test_image(44)

print("\n\n\n An example image from the dataset with various Distortions carried out on it\n")
plot_distorted_image(img, cls)



# Performing Optimization

optimize(num_iterations=100)
    

# Printing Accuracy
    
print_test_accuracy()


# Convolutional Weights
print("Plotting the convolutions Weights:\n")
print("Convolutional Layer-1\n")
plot_conv_weights(weights=weights_conv1, input_channel=0)
print("\nConvolutional Layer-2\n")
plot_conv_weights(weights=weights_conv2, input_channel=1)


# Output of Convolutional Layer

def plot_image(image):
    fig, axes = plt.subplots(1, 2)
    ax0 = axes.flat[0]
    ax1 = axes.flat[1]
    ax0.imshow(image, interpolation='nearest')
    ax1.imshow(image, interpolation='spline16')
    ax0.set_xlabel('Raw')
    ax1.set_xlabel('Smooth')
    plt.show()
    
img, cls = get_test_image(44)
print("\n An Example Image on which Convolutional is applied\n")
plot_image(img)


print("\nConvolutional Layer-1 output on the shown image\n")
plot_layer_output(output_conv1, image=img)
print("\nConvolutional Layer-2 output on the shown image\n")
plot_layer_output(output_conv2, image=img)

print("Class Labels for the Given image:\n")
label_pred, cls_pred = session.run([y_pred, y_pred_cls],
                                   feed_dict={x: [img]})

np.set_printoptions(precision=3, suppress=True)# Setting the Precision

print(label_pred[0]) # Printing the Labels

session.close() # Closing Session