#!/usr/bin/env python3

import torch
import torch.utils.data
import networks
import dataset
import pathlib
import argparse


def main(train_path: pathlib.Path, input_img):
    """
    Runs a pretrained pytorch network
    :param input_img: Input to the network
    :return: Output of network
    """
    ## load
    # ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", "-s", required=True, help="directory containing data")
    parser.add_argument("--tgt", "-t", required=True, help="where to store the models")
    train_set = dataset.DataSet(train_path)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=200, shuffle=True)

    model = networks.InitialNet()  # the model should be defined with the same code you used to create the trained model
    state_dict = torch.load("./../modeldir/checkpoint_2000.pth")
    model.load_state_dict(state_dict)

    # -----------------
    for lbls, imgs in train_loader:
        output = model(imgs)
        print("Output:", output)

    return 0

if __name__ == "__main__":
    y_predict = main()  # outputs prediction of trained network