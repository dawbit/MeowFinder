import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

def label_img(image_name):
    """ Create an one-hot encoded vector from image name """
    word_label = image_name.split('.')[0]
    if word_label == 'cat': return np.array([1, 0])
    elif word_label == 'dog': return np.array([0, 1])


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data