#import matplotlib.pyplot as plt
#import cv2
#import numpy as np
#import os
#from random import shuffle
#from tqdm import tqdm
#import tensorflow as tf
#import tflearn
#from tflearn.layers.conv import conv_2d, max_pool_2d
#from tflearn.layers.core import input_data, dropout, fully_connected
#from tflearn.layers.estimator import regression

import train_data
import test_data
import neural_network1
import plot_data

TRAIN_DIR = 'train'
TEST_DIR = 'test'
IMG_SIZE = 125
LR = 1e-3
MODEL_NAME = 'meowfinder-{}-{}'.format(LR, 'basic')

# If dataset is not created:
train_data = train_data.create_train_data()
test_data = test_data.create_test_data()

# If you have already created the dataset:


neural_network1.network1()
plot_data.plt_dat()