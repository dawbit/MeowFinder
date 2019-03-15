import neural_network
import test_data
import train_data
import plot_data

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

TRAIN_DIR = 'train'
TEST_DIR = 'test'
IMG_SIZE = 125
LR = 1e-3

MODEL_NAME = 'meowfinder-{}-{}'.format(LR, 'basic')

if os.path.isfile('./train_data.npy'):
    # If you have already created the dataset:
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')
else:
    train_data = create_train_data()
    test_data = create_test_data()

train = train_data[:-500]
test = train_data[-500:]