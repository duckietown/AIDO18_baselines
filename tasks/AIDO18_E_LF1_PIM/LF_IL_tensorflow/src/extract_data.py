#!/usr/bin/env python

import cv2
import numpy as np
import pandas as pd
import os
import collections
import rosbag
import cv_bridge
from copy import copy
from extract_data_functions import image_preprocessing, synchronize_data

# A collection of ros messages coming from a single topic.
MessageCollection = collections.namedtuple("MessageCollection", ["topic", "type", "messages"])

def extract_messages(path, requested_topics):

    # check if path is string and requested_topics a list
    assert isinstance(path, str)
    assert isinstance(requested_topics, list)

    bag = rosbag.Bag(path)

    _, available_topics = bag.get_type_and_topic_info()

    # check if the requested topics exist in bag's topics and if yes extract the messages only for them
    extracted_messages = {}
    for topic in requested_topics:
        if topic not in available_topics:
            raise ValueError("Could not find the requested topic (%s) in the bag %s" % (topic, path))
        extracted_messages[topic] = MessageCollection(topic=topic, type=available_topics[topic].msg_type, messages=[])

    for msg in bag.read_messages():
        topic = msg.topic
        if topic not in requested_topics:
            continue
        extracted_messages[topic].messages.append(msg)
    bag.close()

    return extracted_messages

def main():

    # define the list of topics that you want to extract
    ros_topics = [
                # the duckiebot name can change from one bag file to the other, so define
                # the topics WITHOUT the duckiebot name in the beginning
                "/camera_node/image/compressed",
                "/lane_controller_node/car_cmd"
                ]

    # define the bags_directory in order to extract the data
    bags_directory = os.path.join(os.getcwd(), "bag_files")

    # define data_directory
    data_directory = 'data'
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # define train and test directories inside the data directory
    test_dir = os.path.join(data_directory, "test")
    train_dir = os.path.join(data_directory, "train")
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    cvbridge = cv_bridge.CvBridge()

    # create a dataframe to store the data for all bag files
    # df_all = pd.DataFrame()

    first_time = True

    for file in os.listdir(bags_directory):
        if not file.endswith(".bag"):
            continue

        # extract bag_ID to include it in the data for potential future use (Useful in case of weird data distributions
        # or final results, since you will be able to associate the data with the bag files)
        bag_ID = file.partition(".bag")[0]

        # extract the duckiebot name to complete the definition of the nodes
        duckiebot_name = file.partition("_")[2].partition(".bag")[0]

        # complete the topics names with the duckiebot name in the beginning
        ros_topics_temp = copy(ros_topics)
        for num, topic in enumerate(ros_topics_temp):
            ros_topics_temp[num] = "/" + duckiebot_name + topic

        # define absolute path of the bag_file
        abs_path = os.path.abspath(os.path.join(bags_directory, file))

        print("Extract data for {} file.".format(file))
        try:
            msgs = extract_messages(abs_path, ros_topics_temp)
        except rosbag.bag.ROSBagException:
            print("Failed to open {}".format(abs_path))
            continue

                         ######## This following part is implementation specific ########

        # The composition of the ros messages is different (e.g. different names in the messages) and also different
        # tools are used to handle the different extracted data (e.g. cvbridge for images). As a result, the following
        # part of the script can be used as a basis to extract the data, but IT HAS TO BE MODIFIED based on your topics.

        # easy way to find the structure of your ros messages : print dir(msgs[name_of_topic])


        # extract the images and car_cmds messages
        ext_images = msgs["/" + duckiebot_name + "/camera_node/image/compressed"].messages
        ext_car_cmds = msgs["/" + duckiebot_name + "/lane_controller_node/car_cmd"].messages

        # create dataframe with the images and the images' timestamps
        for num, img in enumerate(ext_images):

            # get the rgb image
            img = cvbridge.compressed_imgmsg_to_cv2(img.message)
            img = image_preprocessing(img)  # -> each image is of dimensions (1, 48x96=4608)

            # hack to get the timestamp of each image in <float 'secs.nsecs'> format instead of <int 'rospy.rostime.Time'>
            temp_timestamp = ext_images[num].timestamp
            img_timestamp = temp_timestamp.secs + temp_timestamp.nsecs *10 ** -len(str(temp_timestamp.nsecs))

            temp_df = pd.DataFrame({
                'img': [img],
                'img_timestamp': [img_timestamp]
            })

            if num == 0:
                df_imgs = temp_df.copy()
            else:
                df_imgs = df_imgs.append(temp_df, ignore_index=True)


        # create dataframe with the car_cmds and the car_cmds' timestamps
        for num, cmd in enumerate(ext_car_cmds):

            # read wheel commands messages
            cmd_msg = cmd.message

            # hack to get the timestamp of each image in <float 'secs.nsecs'> format instead of <int 'rospy.rostime.Time'>
            temp_timestamp = ext_car_cmds[num].timestamp
            vel_timestamp = temp_timestamp.secs + temp_timestamp.nsecs * 10 ** -len(str(temp_timestamp.nsecs))

            temp_df = pd.DataFrame({
                'vel_timestamp': [vel_timestamp],
                'vel_omega': [cmd_msg.omega],
                'vel_v': [cmd_msg.v]
            })

            if num == 0:
                df_cmds = temp_df.copy()
            else:
                df_cmds = df_cmds.append(temp_df, ignore_index=True)

        # synchronize data
        print("Starting synchronization of data for {} file.".format(file))

        temp_synch_data, temp_synch_imgs = synchronize_data(df_imgs, df_cmds, bag_ID)
        # print temp_synch_data.shape, temp_synch_imgs.shape

        if first_time:
            synch_data = copy(temp_synch_data)
            synch_imgs = copy(temp_synch_imgs)
            first_time = False

        else:
            synch_data = np.vstack((synch_data, temp_synch_data))
            synch_imgs = np.vstack((synch_imgs, temp_synch_imgs))
        
        print("\nShape of total data: {} , shape of total images: {}\n".format(synch_data.shape, synch_imgs.shape))

    print("Synchronization of all data is finished.\n")

    # define size of train dataset
    train_size = int(0.9 * synch_data.shape[0])

    # create train dataframe
    df_data_train = pd.DataFrame({
        'img_timestamp': synch_data[:train_size, 0],
        'vel_timestamp': synch_data[:train_size, 1],
        'vel_v': synch_data[:train_size, 2],
        'vel_omega': synch_data[:train_size, 3],
        'bag_ID': synch_data[:train_size, 4],
    })

    # create train dataframe for images in order to save them in the same .h5 file with the rest train data
    df_img_train = pd.DataFrame({
        'img': [ synch_imgs[:train_size, :] ]
    })

    # create test dataframe
    df_data_test = pd.DataFrame({
        'img_timestamp': synch_data[train_size:, 0],
        'vel_timestamp': synch_data[train_size:, 1],
        'vel_v': synch_data[train_size:, 2],
        'vel_omega': synch_data[train_size:, 3],
        'bag_ID': synch_data[train_size:, 4],
    })

    # create test dataframe for images in order to save them in the same .h5 file with the rest test data
    df_img_test = pd.DataFrame({
        'img': [ synch_imgs[train_size:, :] ]
    })


    # save train and test datasets to .h5 files

    # ATTENTION 1 !!
    #  If the datasets become too large, you could face memory errors on laptops.
    # If you face memory errors while saving the following files, split the data to multiple .h5 files.

    # ATTENTION 2 !!
    # The .h5 files are tricky and require special attention. In these files you save compressed objects and you can
    # have more than one objects saved in the same file. If for example we have two different dataframes df1 and df2,
    # then df1.to_hdf('file.h5', key='df1') and df2.to_hdf('file.h5', key='df2') will result to both df1, df2 to be
    # saved in 'file.h5' file but with different key for each dataframe. However, if we save the same dataframe to the
    # same .h5 file with the same key, then in this file you will have the same information twice as different objects
    # and thus the size of the .h5 file will be double for no reason and without any warning. As a result, here since
    # the key does not change, we will check if the .h5 file exists before saving the new data, and if it exists we will
    # first remove the previous file ad then save the new data.

    # define the names of the train and test .h5 files
    train_set_name = os.path.join(train_dir, 'train_set.h5')
    test_set_name = os.path.join(test_dir, 'test_set.h5')

    # check if these two files exist in the data directory and if yes remove them before saving the new files
    if os.path.isfile(train_set_name):
        os.remove(train_set_name)

    if os.path.isfile(test_set_name):
        os.remove(test_set_name)

    # df_all_train.to_hdf(train_set_name, 'table')
    df_data_train.to_hdf(train_set_name, key='data')
    df_img_train.to_hdf(train_set_name, key='images')


    # df_all_test.to_hdf(test_set_name, 'table')
    df_data_test.to_hdf(test_set_name, key='data')
    df_img_test.to_hdf(test_set_name, key='images')

    print("\nThe total {} data were split into {} training and {} test datasets and saved in {} "
          "directory.".format(synch_data.shape[0], df_data_train.shape[0], df_data_test.shape[0], data_directory))

if __name__ == "__main__":
    main()