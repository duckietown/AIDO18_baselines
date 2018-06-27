#!/usr/env/python


import tensorflow as tf
import pandas as pd
import time
from image_preprocessing import fun_img_preprocessing
import os
import glob
import numpy as np


def model(optim, learning_rate, dropout, use_three_fc, model_name, velocities, images_preprocessed,
          img_height_size, img_width_size, batch_size):

    # define size of training, validation and test set
    training_num = int(velocities.shape[0] * 0.85)
    validation_num = int(velocities.shape[0] * 0.1)
    # test_num = int(velocities.shape[0] * 0.05)  # not used here just to present the number of test data

    train_x_data = images_preprocessed[0: training_num]
    train_y_data = velocities[0: training_num]

    val_x_data = images_preprocessed[training_num: training_num + validation_num]
    val_y_data = velocities[training_num: training_num + validation_num]

    # test_x_data = images_preprocessed[-test_num:] # not used here, just to present how test data are extracted
    # test_y_data = velocities[-test_num:]  # not used here, just to present how test data are extracted

    # define the number of predicted outputs
    num_pred = 2

    # define placeholder variable for input images
    # [None:tensor may hold arbitrary images, img_size*img_size:size of each image as a vector]
    x = tf.placeholder(tf.float16, shape=[None, img_height_size * img_width_size], name='x')

    # define the 4-d tensor expected by tensorflow
    # [-1: arbitrary num of images, img_height, img_width, num_channels]
    x_img = tf.reshape(x, [-1, img_height_size, img_width_size, 1])

    # define placeholder for the true predictions of commands as taken from bag files
    y_true = tf.placeholder(tf.float16, shape=[None, num_pred], name="y_true_commands")

    # define 1st convolutional layer
    hl_conv_1 = tf.layers.conv2d(x_img, kernel_size=5, filters=8, padding="valid",
                                 activation=tf.nn.relu, name="conv_layer_1")
    # define 1st max pooling
    max_pool_1 = tf.layers.max_pooling2d(hl_conv_1, pool_size=2, strides=2)

    # define 2nd convolutional layer
    hl_conv_2 = tf.layers.conv2d(max_pool_1, kernel_size=5, filters=24, padding="valid",
                                 activation=tf.nn.relu, name="conv_layer_2")
    # define 2nd max pooling
    max_pool_2 = tf.layers.max_pooling2d(hl_conv_2, pool_size=2, strides=2)


    # define 3rd convolutional layer
    hl_conv_3 = tf.layers.conv2d(max_pool_2, kernel_size=5, filters=32, padding="valid",
                                 activation=tf.nn.relu, name="conv_layer_3")

    # define 2nd max pooling
    max_pool_3 = tf.layers.max_pooling2d(hl_conv_3, pool_size=2, strides=2)

    # flatten the ouput of max_pool_2 in order to add the fully connected layers
    conv_flat = tf.layers.flatten(max_pool_3)

    if use_three_fc:

        # add 1st fully connected layers to the neural network
        hl_fc_1 = tf.layers.dense(inputs=conv_flat, units=1024, activation=tf.nn.relu, name="fc_layer_1")
        # add dropout
        drop_out = tf.nn.dropout(hl_fc_1, keep_prob=dropout)
        # add 2nd fully connected layers to the neural network
        hl_fc_2 = tf.layers.dense(inputs=drop_out, units=64, activation=tf.nn.relu, name="fc_layer_2")
        # add dropout
        drop_out = tf.nn.dropout(hl_fc_2, keep_prob=dropout)
        # add 3rd fully connected layers to predict the driving commands
        hl_fc_3 = tf.layers.dense(inputs=drop_out, units=num_pred, name="fc_layer_3")

        # define the predicted outputs of the NN
        y_pred = hl_fc_3

    else:

        # add 1st fully connected layers to the neural network
        hl_fc_1 = tf.layers.dense(inputs=conv_flat, units=64, activation=tf.nn.relu, name="fc_layer_1")
        # add dropout
        drop_out = tf.nn.dropout(hl_fc_1, keep_prob = dropout)
        # add 2nd fully connected layers to predict the driving commands
        hl_fc_2 = tf.layers.dense(inputs=drop_out, units=num_pred, name="fc_layer_2")

        # define the predicted outputs of the NN
        y_pred = hl_fc_2


    # define loss function and encapsulate its scope
    with tf.name_scope("Loss"):
        loss = tf.reduce_sum( tf.square(y_pred - y_true) )

    # define the optimizer
    if optim == "Adam":
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    elif optim == "GDS":
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.name_scope("Loss_v"):
        loss_v = tf.reduce_sum( tf.subtract( y_pred[:, 0], y_true[:, 0] ) )

    with tf.name_scope("Loss_w"):
        loss_w = tf.reduce_sum( tf.subtract( y_pred[:, 1], y_true[:, 1] ) )


    ############################
    #       Run session        #
    ############################

    # define path to save the logs
    logs_train_path = os.getcwd() + '/tensorflow_logs/' + model_name + '/train'
    logs_val_path = os.getcwd() + '/tensorflow_logs/' + model_name + '/validation'

    # Operation to save and restore all variables
    saver = tf.train.Saver()

    # initialize variables
    init = tf.global_variables_initializer()

    # define total number or epochs
    total_epochs = 1000

    # manual scalar summaries
    man_loss = None
    man_loss_summary = tf.Summary()
    man_loss_summary.value.add(tag='Loss', simple_value = man_loss)

    man_loss_v = None
    man_loss_v_summary = tf.Summary()
    man_loss_v_summary.value.add(tag='Loss_v', simple_value = man_loss_v)

    man_loss_w = None
    man_loss_w_summary = tf.Summary()
    man_loss_w_summary.value.add(tag='Loss_w', simple_value = man_loss_w)

    with tf.Session() as sess:

        # run initializer
        sess.run(init)

        # op to write logs to Tensorboard
        val_writer = tf.summary.FileWriter(logs_val_path, graph=tf.get_default_graph() )
        val_writer.add_graph(sess.graph)

        train_writer = tf.summary.FileWriter(logs_train_path, graph=tf.get_default_graph() )
        train_writer.add_graph(sess.graph)

        for epoch in range(total_epochs):

            train_loss = 0
            train_loss_v = 0
            train_loss_w = 0

            validation_loss = 0
            validation_loss_v = 0
            validation_loss_w = 0

            i = 0

            while i <= training_num - 1:

                if i + batch_size <= training_num - 1:
                    train_x = train_x_data[i : i + batch_size]
                    train_y = train_y_data[i : i + batch_size]

                else:
                    train_x = train_x_data[i :]
                    train_y = train_y_data[i :]

                _, c = sess.run( [opt, loss], feed_dict={x: train_x, y_true: train_y} )

                c_v, c_w = sess.run( [loss_v, loss_w], feed_dict={x: train_x, y_true: train_y} )

                train_loss += c
                train_loss_v += c_v
                train_loss_w += c_w

                i += batch_size

            avg_train_loss = train_loss / training_num
            avg_train_v = train_loss_v / training_num
            avg_train_w = train_loss_w / training_num

            # manual summaries
            man_loss_summary.value[0].simple_value = avg_train_loss
            train_writer.add_summary(man_loss_summary, epoch)

            man_loss_v_summary.value[0].simple_value = avg_train_v
            train_writer.add_summary(man_loss_v_summary, epoch)

            man_loss_w_summary.value[0].simple_value = avg_train_w
            train_writer.add_summary(man_loss_w_summary, epoch)

            i = 0

            while i <= validation_num - 1:

                if i + batch_size <= validation_num - 1:
                    val_x = val_x_data[i: i + batch_size]
                    val_y = val_y_data[i : i + batch_size]
                else:
                    val_x = val_x_data[i:]
                    val_y = val_y_data[i :]

                _, c_val = sess.run( [opt, loss], feed_dict={x: val_x, y_true: val_y} )

                validation_loss += c_val

                c_val_v, c_val_w = sess.run( [loss_v, loss_w], feed_dict={x: val_x, y_true: val_y} )

                validation_loss_v += c_val_v
                validation_loss_w += c_val_w

                i += batch_size

            avg_validation_loss = validation_loss / validation_num
            avg_validation_loss_v = validation_loss_v / validation_num
            avg_validation_loss_w = validation_loss_w / validation_num

            # manual summaries
            man_loss_summary.value[0].simple_value = avg_validation_loss
            val_writer.add_summary(man_loss_summary, epoch)

            man_loss_v_summary.value[0].simple_value = avg_validation_loss_v
            val_writer.add_summary(man_loss_v_summary, epoch)

            man_loss_w_summary.value[0].simple_value = avg_validation_loss_w
            val_writer.add_summary(man_loss_w_summary, epoch)


            if epoch % 50 == 0:
                print("Epoch : ", '%04d' % (epoch+1), "train_loss = ", "{:.6f}".format(avg_train_loss), "val_loss = ", "{:.6f}".format(avg_validation_loss),
                      "\nval_loss_v = ", "{:.6f}".format(avg_validation_loss_v), "val_loss_w = ", "{:.6f}".format(avg_validation_loss_w))

            if epoch % 100 == 0:
                saver.save(sess, logs_train_path, epoch)

    # close summary writer
    train_writer.close()
    val_writer.close()


def form_model_name(learning_rate, dropout, use_three_fc, optim, img_height_size, img_width_size, batch_size):

    if use_three_fc:
        fc_param = "fc=3"
    else:
        fc_param = "fc=2"

    return "opt=%s,lr=%.0E,%s,drop=%s,img=%sx%s,batch=%s" % (optim, learning_rate, fc_param, dropout,
                                                    img_height_size, img_width_size, batch_size)

def main():

    start_time = time.time()

    # read data and preprocess images
    print('Reading files')

    # read and append all data to df dataframe
    files = glob.glob('*.h5')
    df = pd.read_hdf(files[0], 'table')

    for i in range(1, len(files)):

        temp_df = pd.read_hdf(files[i], 'table')
        df = df.append(temp_df, ignore_index=True)

    velocities = df[['vel_v', 'vel_omega']].values

    images = df['rgb'].values

    # define the height and width of each image inserted to the training model
    img_height_size = 48
    img_width_size = 96

    # image preprocessing
    images_preprocessed = fun_img_preprocessing(images[0], img_height_size, img_width_size)

    for img in images[1:]:
        images_preprocessed = np.append(images_preprocessed, fun_img_preprocessing(img, img_height_size,
                                                                                   img_width_size), axis=0)

    print('Preprocessing finished')

    # for batch_size in [50, 100]:
    for batch_size in [100]:

        # for optim in ["Adam", "GDS"]:
        for optim in ["GDS"]:

            # for learning_rate in [1E-3, 1E-4]:
            for learning_rate in [1E-5]:

                # for dropout in [0.2, 0.5, 0.8, 1]:  !! actually this is the keep probability argument of the dropout
                for dropout in [0.5]:

                    # for use_three_fc in [True, False]:
                    for use_three_fc in [False]:

                        model_name = form_model_name(learning_rate, dropout, use_three_fc, optim,
                                                     img_height_size, img_width_size, batch_size)

                        print('Starting training for %s' % model_name)

                        model(optim, learning_rate, dropout, use_three_fc, model_name, velocities, images_preprocessed,
                              img_height_size, img_width_size, batch_size)

                        print('Finished training for %s' % model_name)


    training_time = time.time() - start_time

    print("Total training time for all models: " + str(training_time/60) + " minutes")


if __name__ == '__main__':
    main()
