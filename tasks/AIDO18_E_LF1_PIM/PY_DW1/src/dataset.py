#!/usr/bin/env python3
import sys
import pathlib
import requests
import PIL.Image

import torch
import torchvision.transforms as transforms


def download_logs(path: pathlib.Path) -> None:
    """
    Downloads logs. To be replaced by a API call or similar (depending on specification).
    
    :param path: where to store the logs
    """
    urls = {
        "20180108135529_a313": "https://gateway.ipfs.io/ipfs/QmbUTAU5xZc6YPdvmp9eMi8tWsXniooeL1mkewmBD83ViK",
        "20180108141006_a313": "https://gateway.ipfs.io/ipfs/QmP1todoYHyXkMENw3kVth4h4FUeVH4J1sL7NPDqfnVr9X",
        "20180108141155_a313": "https://gateway.ipfs.io/ipfs/QmcEWuGZYUeKfYKQTsdSqaLzXm2brtqz7eGjjFEGrfHRL1",
        "20180108141448_a313": "https://gateway.ipfs.io/ipfs/QmV3EGcd7aJSzQkZkdYHSLgK1Pp1THKSj5y3fJ64nNf3RA",
        "20180108141719_a313": "https://gateway.ipfs.io/ipfs/QmPzNbtuuyUPwrW9U5ZESgtLr9y5YoAMK9rGkHo2qaVrjV"
    }
    path.mkdir(exist_ok=True, parents=True)
    for name, url in urls.items():
        file_path = path / "{}.bag".format(name)
        if file_path.is_file():
            print("{}: already downloaded".format(name))
            continue

        with file_path.open("wb") as f:
            print("{}: downloading...".format(name))
            response = requests.get(url, stream=True)

            if response.status_code != 200:
                error_msg = response.raw if len(response.raw) < 200 else response.raw[:199]
                raise RuntimeError(
                    "Failed to download log at {}\nCODE: {}\nMSG: {}".format(url, response.status_code, error_msg))

            total_length = response.headers.get('content-length')
            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=1024):
                    dl += len(data)
                    f.write(data)
                    done = int(100 * dl / total_length)
                    sys.stdout.write("\r[{}{}] {}%".format('=' * (done // 2), ' ' * (50 - done // 2), done))
                    sys.stdout.flush()
        sys.stdout.write("\n")


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

