#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import pandas as pd
import os

def load_data(file_path):

    # read dataframes
    df_data = pd.read_hdf(file_path, key='data')
    df_img = pd.read_hdf(file_path, key='images', encoding='utf-8')

    # extract omega velocities from dataset
    velocities = df_data['vel_omega'].values
    velocities = np.reshape(velocities, (-1, 1))

    # extract images from dataset
    images = df_img['img'][0]

    print('The dataset is loaded: {} images and {} omega velocities.'.format(images.shape[0], velocities.shape[0]))

    if not images.shape[0] == velocities.shape[0]:
        raise ValueError("The number of images and velocities must be the same.")

    return velocities, images


def form_model_name(batch_size, lr, optimizer, epochs):
    return "batch={},lr={},optimizer={},epochs={}".format(batch_size, lr, optimizer, epochs)


class CNN_training:

    def __init__(self, batch, epochs, learning_rate, optimizer):

        self.batch_size = batch
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer

    def backpropagation(self):

        # define the optimizer
        with tf.name_scope("Optimizer"):
            if self.optimizer == "Adam":
                return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer == "GDS":
                return tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def loss_function(self):

        # define loss function and encapsulate its scope
        with tf.name_scope("Loss"):
            return tf.reduce_mean( tf.square(self.vel_pred - self.vel_true) )

    def model(self, x):

        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):

            # define the 4-d tensor expected by TensorFlow
            # [-1: arbitrary num of images, img_height, img_width, num_channels]
            x_img = tf.reshape(x, [-1, 48, 96, 1])

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

            return hl_fc_2

    def epoch_iteration(self, data_size, x_data, y_data, mode):

        pred_loss = 0
        i = 0
        while i <= data_size - 1:

            # extract batch
            if i + self.batch_size <= data_size - 1:
                train_x = x_data[i: i + self.batch_size]
                train_y = y_data[i: i + self.batch_size]
            else:
                train_x = x_data[i:]
                train_y = y_data[i:]

            if mode == 'train':
                # train using the batch and calculate the loss
                _, c = self.sess.run([self.opt, self.loss], feed_dict={self.x: train_x, self.vel_true: train_y})

            elif mode == 'test':
                # train using the batch and calculate the loss
                c = self.sess.run(self.loss, feed_dict={self.x: train_x, self.vel_true: train_y})

            pred_loss += c
            i += self.batch_size

        return pred_loss

    def training(self, model_name, train_velocities, train_images, test_velocities, test_images):

        # define paths to save the TensorFlow logs
        model_path = os.path.join(os.getcwd(), 'tensorflow_logs', model_name)
        logs_train_path = os.path.join(model_path, 'train')
        logs_test_path = os.path.join(model_path, 'test')
        graph_path = os.path.join(model_path, 'graph')


        # manual scalar summaries for loss tracking
        man_loss_summary = tf.Summary()
        man_loss_summary.value.add(tag='Loss', simple_value=None)

        # define placeholder variable for input images (each images is a row vector [1, 4608 = 48x96x1])
        self.x = tf.placeholder(tf.float16, shape=[None, 48 * 96], name='x')

        # define placeholder for the true omega velocities
        # [None: tensor may hold arbitrary num of velocities, number of omega predictions for each image]
        self.vel_true = tf.placeholder(tf.float16, shape=[None, 1], name="vel_true")
        self.vel_pred = self.model(self.x)

        self.loss = self.loss_function()
        self.opt = self.backpropagation()

        # initialize variables
        init = tf.global_variables_initializer()

        # Operation to save and restore all variables
        saver = tf.train.Saver()

        with tf.Session() as self.sess:

            # run initializer
            self.sess.run(init)

            # operation to write logs for Tensorboard
            tf_graph = self.sess.graph
            test_writer = tf.summary.FileWriter(logs_test_path, graph=tf.get_default_graph() )
            test_writer.add_graph(tf_graph)

            train_writer = tf.summary.FileWriter(logs_train_path, graph=tf.get_default_graph() )
            train_writer.add_graph(tf_graph)

            tf.train.write_graph(tf_graph.as_graph_def(), graph_path, 'graph.pbtxt', as_text= True)
            tf.train.write_graph(tf_graph.as_graph_def(), graph_path, 'graph.pb', as_text= False)

            for epoch in range(self.epochs):

                # run train cycle
                avg_train_loss = self.epoch_iteration(train_velocities.shape[0], train_images, train_velocities, 'train')

                # save the training loss using the manual summaries
                man_loss_summary.value[0].simple_value = avg_train_loss
                train_writer.add_summary(man_loss_summary, epoch)

                # run test cycle
                avg_test_loss = self.epoch_iteration(test_velocities.shape[0], test_images, test_velocities, 'test')

                # save the test errors using the manual summaries
                man_loss_summary.value[0].simple_value = avg_test_loss
                test_writer.add_summary(man_loss_summary, epoch)

                # print train and test loss to monitor progress during training every 50 epochs
                if epoch % 50 == 0:
                    print("Epoch: {:04d} , train_loss = {:.6f} , test_loss = {:.6f}".format(epoch+1, avg_train_loss, avg_test_loss))

                # save weights every 100 epochs
                if epoch % 100 == 0:
                    saver.save(self.sess, logs_train_path, epoch)

        # close summary writer
        train_writer.close()
        test_writer.close()





