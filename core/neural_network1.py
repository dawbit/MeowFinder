from random import shuffle

import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import settings as s


def network1(train, test, train_amount):
    global model

    train_amount = int(train_amount * 4.8 / 6)

    x_train = np.array([i[0] for i in train[:train_amount]]).reshape(-1, s.IMG_SIZE, s.IMG_SIZE, 1)  # oryginalna wersja#
    y_train = [i[1] for i in train[:train_amount]]

    x_validation = np.array([i[0] for i in train[train_amount:]]).reshape(-1, s.IMG_SIZE, s.IMG_SIZE, 1)  # oryginalna wersja#
    y_validation = [i[1] for i in train[train_amount:]]

    # x_test = np.array([i[0] for i in test]).reshape(-1, s.IMG_SIZE, s.IMG_SIZE, 1)  # oryginalna wersja#
    # y_test = [i[1] for i in train]

    # # Make sure the data is normalized
    # img_prep = ImagePreprocessing()
    # img_prep.add_featurewise_zero_center()
    # img_prep.add_featurewise_stdnorm()
    #
    # # Create extra synthetic training data by flipping, rotating and blurring the
    # # images on our data set.
    # img_aug = ImageAugmentation()
    # img_aug.add_random_flip_leftright()
    # img_aug.add_random_rotation(max_angle=25.)
    # img_aug.add_random_blur(sigma_max=3.)
    # data_preprocessing=img_prep,
    #                          data_augmentation=img_aug,

    network = input_data(shape=[None, s.IMG_SIZE, s.IMG_SIZE, 1], name='input')

    # Step 1: Convolution
    network = conv_2d(network, 32, 5, activation='relu', regularizer='L2')
    # Step 2: Max pooling
    network = max_pool_2d(network, 5)

    # Step 3: Convolution again
    network = conv_2d(network, 64, 5, activation='relu', regularizer='L2')
    network = max_pool_2d(network, 5)
    # Step 4: Convolution yet again
    network = conv_2d(network, 128, 5, activation='relu', regularizer='L2')
    # Step 5: Max pooling again
    network = max_pool_2d(network, 5)

    network = conv_2d(network, 64, 5, activation='relu', regularizer='L2')
    network = max_pool_2d(network, 5)

    network = conv_2d(network, 32, 5, activation='relu', regularizer='L2')
    network = max_pool_2d(network, 5)

    # Step 6: Fully-connected 1024 node neural network
    network = fully_connected(network, 1024, activation='relu', regularizer='L2')

    # Step 7: Dropout – throw away some data randomly during training to prevent over-fitting
    network = dropout(network, 0.8)

    # Step 8: Fully-connected neural network with two outputs (0=isn’t a bird, 1=is a bird) to make the final prediction
    network = fully_connected(network, 12, activation='softmax')

    # Tell tflearn how we want to train the network
    network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=s.LR, name='targets')

    # Wrap the network in a model object
    model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='log')

    # Train it! We’ll do 100 training passes and monitor it as it goes.
    model.fit(x_train, y_train, n_epoch=12,
              validation_set=({'input': x_validation}, {'targets': y_validation}),
              shuffle=True,
              snapshot_epoch=True,
              show_metric=True,
              batch_size=40,
              run_id=s.MODEL_NAME)

    # model.fit(X_train, y_train, n_epoch=100, shuffle=True, validation_set=(X_test, y_test),
    #           show_metric=True, batch_size=96,
    #           snapshot_epoch=True,
    #           run_id='meow-finder')

    # Save model when training is complete to a file
    model.save(s.MODEL_NAME)
    print('Network trained and saved as {0}'.format(s.MODEL_NAME))

    return model
