#!/usr/bin/env python3
import os
import requests
import PIL.Image
import pathlib

import torch
import torchvision.transforms as transforms


def download_logs():  # -> None:
    """
    Downloads logs. Downloads logs specified in urls dict below.
    """
    urls = {
        "20180108135529_a313": "https://gateway.ipfs.io/ipfs/QmbUTAU5xZc6YPdvmp9eMi8tWsXniooeL1mkewmBD83ViK",
        "20180108141006_a313": "https://gateway.ipfs.io/ipfs/QmP1todoYHyXkMENw3kVth4h4FUeVH4J1sL7NPDqfnVr9X",
        # "20180108141155_a313": "https://gateway.ipfs.io/ipfs/QmcEWuGZYUeKfYKQTsdSqaLzXm2brtqz7eGjjFEGrfHRL1",
        "20180108141448_a313": "https://gateway.ipfs.io/ipfs/QmV3EGcd7aJSzQkZkdYHSLgK1Pp1THKSj5y3fJ64nNf3RA",
        "20180108141719_a313": "https://gateway.ipfs.io/ipfs/QmPzNbtuuyUPwrW9U5ZESgtLr9y5YoAMK9rGkHo2qaVrjV"
    }
    directory = os.path.join(os.getcwd(), "logdir")
    if not os.path.exists(directory):
        os.makedirs(directory)

    for url in urls:

        # extract bag_ID from url
        bag_ID = url

        # extract link of the bag file
        link = urls[url]

        # check that a file exists on the defined url
        response = requests.head(link)
        if response.status_code != 200:
            raise RuntimeError("Cannot find the file {} at the link {}".format(bag_ID, link))

        # define bag_name but also prevent errors for bag_ID misunderstanding (bag_ID should be without .bag extension)
        if ".bag" in bag_ID:
            bag_name = os.path.join(directory, bag_ID)
        else:
            bag_name = os.path.join(directory, bag_ID + ".bag")

        if not os.path.isfile(bag_name):

            # download file and save it to a bag file
            r = requests.get(link, allow_redirects=True)
            open(bag_name, 'wb').write(r.content)

        # print which bag files have been downloaded so far
        if ".bag" in bag_ID:
            print("The {} file is downloaded.".format(bag_ID))
        else:
            print("The {}.bag file is downloaded.".format(bag_ID))


class DataSet(torch.utils.data.Dataset):
    """
    A simple data set to use when you have a single folder containing the images and for every image a .txt file in the
    same directory with the same name containing a single line with space separated values as labels.
    """

    def __init__(self, data_dir: pathlib.Path):
        self.images = [path for path in data_dir.iterdir() if path.suffix == ".jpg"]
        self.labels = []
        for image in self.images:
            lbl_file = image.parent / "{}.txt".format(image.stem)
            if not lbl_file.is_file():
                raise IOError("Could not find the label file {}".format(lbl_file))
            with lbl_file.open("r") as fid:
                self.labels.append(list(map(float, fid.read().strip().split(" "))))

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((80, 160)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = PIL.Image.open(self.images[item])
        if img is None:
            raise IOError("Could not read the image {}".format(self.images[item]))
        return torch.Tensor(self.labels[item]), self.transform(img)


def main():
    # download bag files
    # path = 'bag_files'
    download_logs()

if __name__ == "__main__":
    main()