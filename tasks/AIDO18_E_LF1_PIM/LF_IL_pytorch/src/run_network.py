#!/usr/bin/env python3

import torch
import torch.utils.data
import networks


def main(input_img):
    """
    Runs a pretrained pytorch network
    :param input_img: Input to the network
    :return: Output of network
    """
    ## load
    model = networks.InitialNet()  # the model should be defined with the same code you used to create the trained model
    state_dict = torch.load("./../modeldir/checkpoint_2000.pth")
    model.load_state_dict(state_dict)
    output = model(input_img)
    return output

if __name__ == "__main__":
    y_predict = main()  # outputs prediction of trained network