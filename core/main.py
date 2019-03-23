#import matplotlib.pyplot as plt
#import cv2
import numpy as np
import os
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

if os.path.isfile('train_data.npy'):
    train_data = np.load('train_data.npy')
    if os.path.isfile('test_data.npy'):
        test_data = np.load('test_data.npy')
    else:
        test_data.create_test_data()
else:
    print("Nie istnieje")
    train_data = train_data.create_train_data()
    test_data = test_data.create_test_data()

model = neural_network1.network1(train_data, test_data)
plot_data.plt_dat(model, test_data)
