#!/usr/bin/env python3
import argparse
import pathlib

import torch

import networks


def run(model_path: pathlib.Path):
    net = networks.InitialNet()
    net.load_state_dict(torch.load(model_path.as_posix()))

    while True:
        # Spin up a process that listens for images and compute the steering commands:
        # img = sub.recv(zmq.DONT_WAIT)
        # if img is not None:
        #     cmd = net(img)
        #     cmd = serialize(cmd)
        #     pub.send(cmd)
        # else:
        #     time.sleep(0.01)
        raise NotImplemented()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", "-m", required=True, help="the path to the model weights")

    args = parser.parse_args()

    run(pathlib.Path(args.model_path))


if __name__ == "__main__":
    main()
