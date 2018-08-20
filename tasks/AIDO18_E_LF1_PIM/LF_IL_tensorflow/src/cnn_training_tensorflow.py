#!/usr/bin/env python

import tensorflow as tf
import pandas as pd
import time
import os
import numpy as np
from sklearn.model_selection import train_test_split

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def training(batch_size, lr, optimizer, epochs, images, velocities, model_name):

    # split dataset in training and validation datasets
    train_x_data, val_x_data, train_y_data, val_y_data = train_test_split(images, velocities, test_size=0.1)

    # calculate size of training and validation sets
    training_num = train_x_data.shape[0]
    validation_num = val_x_data.shape[0]

    # define placeholder variable for input images
    # [None:tensor may hold arbitrary images, 4608:size of each image as a vector (4608 = 48x96)]
    x = tf.placeholder(tf.float16, shape=[None, 4608], name='x')

    # define the 4-d tensor expected by tensorflow
    # [-1: arbitrary num of images, img_height, img_width, num_channels]
    x_img = tf.reshape(x, [-1, 48, 96, 1])

    # define placeholder for the true predictions of omega velocities
    # [None: tensor may hold arbitrary num of velocities, number of omega predictions for each image]
    y_true = tf.placeholder(tf.float16, shape=[None, 1], name="y_true_commands")

    # define 1st convolutional layer
    hl_conv_1 = tf.layers.conv2d(x_img, kernel_size=5, filters=2, padding="valid",
                                 activation=tf.nn.relu, name="conv_layer_1")

    max_pool_1 = tf.layers.max_pooling2d(hl_conv_1, pool_size=2, strides=2)

    # define 2nd convolutional layer
    hl_conv_2 = tf.layers.conv2d(max_pool_1, kernel_size=5, filters=8, padding="valid",
                                 activation=tf.nn.relu, name="conv_layer_2")

    max_pool_2 = tf.layers.max_pooling2d(hl_conv_2, pool_size=2, strides=2)

    # flatten tensor to connect it with the fully connected layers
    conv_flat = tf.layers.flatten(max_pool_2)

    # add 1st fully connected layers to the neural network
    hl_fc_1 = tf.layers.dense(inputs=conv_flat, units=64, activation=tf.nn.relu, name="fc_layer_1")

    # add 2nd fully connected layers to predict the driving commands
    hl_fc_2 = tf.layers.dense(inputs=hl_fc_1, units=1, name="fc_layer_2")

    # define the predicted outputs of the CNN
    y_pred = hl_fc_2

    # define loss function and encapsulate its scope
    with tf.name_scope("Loss"):
        loss = tf.reduce_sum( tf.square(y_pred - y_true) )

    # define the optimizer
    with tf.name_scope("Optimizer"):
        if optimizer == "Adam":
            opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        elif optimizer == "GDS":
            opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    ############################
    #       Run session        #
    ############################

    # define paths to save the tensorflow logs
    model_path = os.path.join(os.getcwd(), 'tensorflow_logs', model_name)
    logs_train_path = os.path.join(model_path, 'train')
    logs_val_path = os.path.join(model_path, 'validation')
    graph_path = os.path.join(model_path, 'graph')

    # Operation to save and restore all variables
    saver = tf.train.Saver()

    # initialize variables
    init = tf.global_variables_initializer()

    # manual scalar summaries for model error tracking
    man_loss = None
    man_loss_summary = tf.Summary()
    man_loss_summary.value.add(tag='Loss', simple_value = man_loss)

    # auxiliary function to run training and validation
    def auxiliary_fun(data_size, x_data, y_data):

        pred_loss = 0
        i = 0
        while i <= data_size - 1:

            # extract batch
            if i + batch_size <= data_size - 1:
                train_x = x_data[i: i + batch_size]
                train_y = y_data[i: i + batch_size]
            else:
                train_x = x_data[i:]
                train_y = y_data[i:]

            # train using the batch and calculate the loss
            _, c = sess.run([opt, loss], feed_dict={x: train_x, y_true: train_y})

            pred_loss += c
            i += batch_size

        # for each epoch calculate the average loss of the model
        avg_loss = pred_loss / data_size
        return avg_loss

    with tf.Session() as sess:

        # run initializer
        sess.run(init)

        # operation to write logs for Tensorboard
        tf_graph = sess.graph
        val_writer = tf.summary.FileWriter(logs_val_path, graph=tf.get_default_graph() )
        val_writer.add_graph(tf_graph)

        train_writer = tf.summary.FileWriter(logs_train_path, graph=tf.get_default_graph() )
        train_writer.add_graph(tf_graph)

        tf.train.write_graph(tf_graph.as_graph_def(), graph_path, 'graph.pbtxt', as_text= True)
        tf.train.write_graph(tf_graph.as_graph_def(), graph_path, 'graph.pb', as_text= False)

        for epoch in range(epochs):

            # run training cycle
            avg_train_loss = auxiliary_fun(training_num, train_x_data, train_y_data)

            # save the training loss using the manual summaries
            man_loss_summary.value[0].simple_value = avg_train_loss
            train_writer.add_summary(man_loss_summary, epoch)

            # run validation cycle
            avg_validation_loss = auxiliary_fun(validation_num, val_x_data, val_y_data)

            # save the validation errors using the manual summaries
            man_loss_summary.value[0].simple_value = avg_validation_loss
            val_writer.add_summary(man_loss_summary, epoch)

            # print train and validation loss to monitor progress during training every 50 epochs
            if epoch % 50 == 0:
                print("Epoch: {:04d} , train_loss = {:.6f} , val_loss = {:.6f}".format(epoch+1, avg_train_loss, avg_validation_loss))

            # save weights every 100 epochs
            if epoch % 100 == 0:
                saver.save(sess, logs_train_path, epoch)

    # close summary writer
    train_writer.close()
    val_writer.close()


def form_model_name(batch_size, lr, optimizer, epochs):

    return "batch={},lr={},optimizer={},epochs={}".format(batch_size, lr, optimizer, epochs)


def main():

    # define path for training dataset
    file_path = os.path.join(os.getcwd(), 'data', 'train', 'train_set.h5')

    # define batch_size (e.g 50, 100)
    batch_size = 100

    # define which optimizer you want to use (e.g "Adam", "GDS"). For "Adam" and "GDS" this script will take care the rest.
    # ATTENTION !! If you want to choose a different optimizer from these two, you will have to add it in the training function above.
    optimizer = "GDS"

    # define learning rate (e.g 1E-3, 1E-4, 1E-5):
    lr = 1E-4

    # define total epochs (e.g 1000, 5000, 10000)
    epochs = 1000

    # keep track of training time
    start_time = time.time()

    # read data and preprocess images
    print('Reading training dataset')

    # read train dataframe
    df_data = pd.read_hdf(file_path, key='data')
    df_img = pd.read_hdf(file_path, key='images', encoding='utf-8')

    # extract omega velocities from training dataset
    velocities = df_data['vel_omega'].values
    velocities = np.reshape(velocities, (-1, 1))

    # extract images from training dataset
    images = df_img['img'][0]

    print('The dataset is loaded: {} images and {} omega velocities.'.format(images.shape[0], velocities.shape[0]))

    if not images.shape[0] == velocities.shape[0]:
        raise ValueError("The number of images and velocities must be the same.")

    # construct model name based on hyper parameters
    model_name = form_model_name(batch_size, lr, optimizer, epochs)

    print('Starting training for {} model.'.format(model_name))

    # Train the model
    training(batch_size, lr, optimizer, epochs, images, velocities, model_name)

    # calculate total training time in minutes
    training_time = (time.time() - start_time) / 60

    print('Finished training of {} in {} minutes.'.format(model_name, training_time))

if __name__ == '__main__':
    main()
