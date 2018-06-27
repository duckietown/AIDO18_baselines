#!/usr/bin/env python

import tensorflow as tf
import cv2

import rospy
import cv_bridge
import sensor_msgs.msg

import duckietown_msgs.msg

from cnn_lane_following.cnn_predictions import fun_img_preprocessing
from cnn_lane_following.cnn_predictions import predictions


class CNNController:

    def __init__(self):

        self.cnn = predictions
        self.cvbridge = cv_bridge.CvBridge()
        self.pub = rospy.Publisher("~car_cmd", duckietown_msgs.msg.Twist2DStamped)
        rospy.Subscriber("~compressed", sensor_msgs.msg.CompressedImage, self.receive_img)

    def receive_img(self, img_msg):

        rospy.loginfo("received img")
        img = self.cvbridge.compressed_imgmsg_to_cv2(img_msg)
        img_height_size = 48
        img_width_size = 96
        img = fun_img_preprocessing(img, img_height_size, img_width_size)  # returns image of shape [1, img_height_size x img_width_size]

        logs_path = './tensorflow_logs/opt=GDS,lr=1E-05,fc=2,drop=0.5,img=48x96,batch=100/train-900'
        prediction = self.cnn(logs_path, image=img)
        car_control_msg = duckietown_msgs.msg.Twist2DStamped()
        car_control_msg.header = img_msg.header
        # car_control_msg.v = out[0]
        car_control_msg.v = 0.386400014162
        car_control_msg.omega = prediction[1]

        rospy.loginfo( "publishing wheel_cmd: {}, {}".format(prediction[0], prediction[1]) )
        self.pub.publish(car_control_msg)

def main():

    rospy.init_node("cnn_lanecontrol")
    controller = CNNController()
    rospy.spin()

if __name__ == "__main__":
    main()
