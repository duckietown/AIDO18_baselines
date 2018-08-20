#!/usr/bin/env python
"""
ATTENTION: because this file needs to access rosbag utilites, it must be run with python 2.7.
"""
import os
import argparse
import collections
import random

import rosbag
import cv_bridge
import cv2


# A collection of ros messages coming from a single topic.
MessageCollection = collections.namedtuple("MessageCollection", ["topic", "type", "messages"])


def extract_messages(path, requested_topics):
    """
    Sorts the content of the bag after types.

    :param path: str -> path to rosbag
    :param requested_topics: List[str] -> topics for which the messages should be extraced
    :return:
    """
    assert isinstance(path, str)
    assert isinstance(requested_topics, list)

    bag = rosbag.Bag(path)

    _, available_topics = bag.get_type_and_topic_info()

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


def match_imgs_and_cmds(imgs, cmds):
    img_idx = 1
    cmd_idx = 0

    cmd_indexes = [-1] * len(imgs)
    t_min = imgs[0].timestamp
    while img_idx < len(imgs) and cmd_idx < len(cmds):
        t_img = imgs[img_idx].timestamp
        t_cmd = cmds[cmd_idx].timestamp

        if t_min > t_cmd:
            cmd_idx += 1
        elif t_img < t_cmd:
            img_idx += 1
        else:
            cmd_indexes[img_idx - 1] = cmd_idx
            cmd_idx += 1
            img_idx += 1
            t_min = t_img
    return cmd_indexes


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src", "-s", required=True, help="folder containing the ros bags")
    parser.add_argument("--tgt", "-t", required=True, help="folder to store images")

    args = parser.parse_args()

    test_dir = os.path.join(args.tgt, "test")
    train_dir = os.path.join(args.tgt, "train")
    if not os.path.exists(args.tgt):
        os.mkdir(args.tgt)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    test_idx = 0
    train_idx = 0
    test_percentage = 0.3

    cvbridge = cv_bridge.CvBridge()
    for pth in os.listdir(args.src):
        if not pth.endswith(".bag"):
            continue
        abs_path = os.path.abspath(os.path.join(args.src, pth))

        print("Extracting images for {}".format(abs_path))
        try:
            msgs = extract_messages(abs_path, ["/a313/camera_node/image/compressed",
                                               "/a313/wheels_driver_node/wheels_cmd_executed"])
        except rosbag.bag.ROSBagException:
            print("failed to open {}".format(abs_path))
            continue

        imgs = msgs["/a313/camera_node/image/compressed"].messages
        cmds = msgs["/a313/wheels_driver_node/wheels_cmd_executed"].messages

        cmd_indexes = match_imgs_and_cmds(imgs, cmds)

        for i, cmd_idx in enumerate(cmd_indexes):
            if cmd_idx == -1:
                continue

            if random.random() < test_percentage:
                img_pth = os.path.join(test_dir, "{0:06d}.jpg".format(test_idx))
                lbl_pth = os.path.join(test_dir, "{0:06d}.txt".format(test_idx))
                test_idx += 1
            else:
                img_pth = os.path.join(train_dir, "{0:06d}.jpg".format(train_idx))
                lbl_pth = os.path.join(train_dir, "{0:06d}.txt".format(train_idx))
                train_idx += 1

            img = cvbridge.compressed_imgmsg_to_cv2(imgs[i].message)
            cv2.imwrite(img_pth, img)
            with open(lbl_pth, "w") as fid:
                cmd_msg = cmds[cmd_idx].message
                fid.write("{} {}".format(cmd_msg.vel_right, cmd_msg.vel_left))


if __name__ == "__main__":
    main()
