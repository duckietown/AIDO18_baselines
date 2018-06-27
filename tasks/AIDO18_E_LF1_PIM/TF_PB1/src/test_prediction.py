import tensorflow as tf
import numpy as np
import os
from prediction import fun_img_preprocessing
from prediction import prediction
import matplotlib.pyplot as plt


def main():


    image = np.load('test_images_1.npy')

    image_final_height =48
    image_final_width =96

    test_image = fun_img_preprocessing(image[0], image_final_height, image_final_width)

    logs_path = os.getcwd() + '/tensorflow_logs/opt=GDS,lr=1E-05,fc=2,drop=0.5,img=48x96,batch=100/' + 'train-900'

    pred= prediction(logs_path, test_image)

    print pred[0], pred[1]


if __name__ == '__main__':
    main()
