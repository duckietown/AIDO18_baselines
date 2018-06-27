#!/usr/env/python
import cv2
import numpy as np
import tensorflow as tf

def fun_img_preprocessing(image, image_final_height, image_final_width):

    # crop the 1/3 upper part of the image
    new_img = image[480/3:, :, :]

    # transform the color image to grayscale
    new_img = cv2.cvtColor( new_img[:, :, :], cv2.COLOR_RGB2GRAY )

    # resize the image from 320x640 to 48x96
    new_img = cv2.resize( new_img, ( image_final_width, image_final_height ) ) # returns image 32 x 64 and not 64 x 32

    # normalize images to range [0, 1] (devide each pixel by 255)
    # first transform the array of int to array of float else the division with 255 will return an array of 0s
    new_img = new_img.astype(float)
    new_img = new_img / 255

    # new_part
    new_img = np.reshape(new_img, (1, -1))

    return new_img


def prediction(logs_path, image):

    ############################
    #  Build Tensorflow graph  #
    ############################

    # define the fc layers if they will be 2 or 3 depending on which model we restore (at the moment we use 2 fc layers)

    # define the height and width of each image inserted to training model
    img_height_size = 48
    img_width_size = 96

    # define the number of predicted outputs
    num_pred = 2

    # define placeholder variable for input images
    x = tf.placeholder(tf.float16, shape=[1, img_height_size * img_width_size], name='x')

    # define the 4-d tensor expected by tensorflow
    # [1: num of images, img_height, img_width, num_channels]
    x_img = tf.reshape(x, [1, img_height_size, img_width_size, 1])

    # define 1st convolutional layer
    hl_conv_1 = tf.layers.conv2d(x_img, kernel_size=5, filters=8, padding="valid", activation=tf.nn.relu, name="conv_layer_1")

    # define 1st max pooling
    max_pool_1 = tf.layers.max_pooling2d(hl_conv_1, pool_size=2, strides=2)

    # define 2nd convolutional layer
    hl_conv_2 = tf.layers.conv2d(max_pool_1, kernel_size=5, filters=24, padding="valid", activation=tf.nn.relu, name="conv_layer_2")

        # define 2nd max pooling
    max_pool_2 = tf.layers.max_pooling2d(hl_conv_2, pool_size=2, strides=2)

    # define 3rd convolutional layer
    hl_conv_3 = tf.layers.conv2d(max_pool_2, kernel_size=5, filters=32, padding="valid",
                                 activation=tf.nn.relu, name="conv_layer_3")

    # define 2nd max pooling
    max_pool_3 = tf.layers.max_pooling2d(hl_conv_3, pool_size=2, strides=2)

    # flatten the ouput of max_pool_2 in order to add the fully connected layers
    conv_flat = tf.layers.flatten(max_pool_3)

    # add 1st fully connected layers to the neural network
    hl_fc_1 = tf.layers.dense(inputs=conv_flat, units=64, activation=tf.nn.relu, name="fc_layer_1")

    # add 2nd fully connected layers to predict the driving commands
    hl_fc_2 = tf.layers.dense(inputs=hl_fc_1, units=num_pred, name="fc_layer_2")

    # define the predicted outputs of the NN
    y_pred = hl_fc_2


    ################################
    # Restore model and prediction #
    ################################

    # Operation to save and restore all variables
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # Restore model
        saver.restore(sess, logs_path)

        # predict wheer driver commands
        pred_1, pred_2 = sess.run([y_pred[0, 0], y_pred[0, 1]], feed_dict={x: image })

    return pred_1, pred_2
