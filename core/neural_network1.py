import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import settings as s


def network1(train, train_amount):
    global model

    # region SPLIT DATA FOR TRAIN/VALIDATION
    train_amount = int(train_amount * 4.8 / 6)

    x_train = np.array([i[0] for i in train[:train_amount]]).reshape(-1, s.IMG_SIZE, s.IMG_SIZE, 1)
    y_train = [i[1] for i in train[:train_amount]]

    x_validation = np.array([i[0] for i in train[train_amount:]]).reshape(-1, s.IMG_SIZE, s.IMG_SIZE, 1)
    y_validation = [i[1] for i in train[train_amount:]]
    # endregion

    # region NETWORK
    network = input_data(shape=[None, s.IMG_SIZE, s.IMG_SIZE, 1], name='input')

    network = conv_2d(network, 32, 5, activation='relu', regularizer='L2')
    network = max_pool_2d(network, 5)

    network = conv_2d(network, 64, 5, activation='relu', regularizer='L2')
    network = max_pool_2d(network, 5)

    network = conv_2d(network, 128, 5, activation='relu', regularizer='L2')
    network = max_pool_2d(network, 5)

    network = conv_2d(network, 64, 5, activation='relu', regularizer='L2')
    network = max_pool_2d(network, 5)

    network = conv_2d(network, 32, 5, activation='relu', regularizer='L2')
    network = max_pool_2d(network, 5)

    network = fully_connected(network, 512, activation='relu', regularizer='L2')
    network = dropout(network, 0.7)

    network = fully_connected(network, 12, activation='softmax')

    network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=s.LR, name='targets')

    model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='log')
    # endregion

    # region TRAIN
    model.fit(x_train, y_train, n_epoch=12,
              validation_set=({'input': x_validation}, {'targets': y_validation}),
              shuffle=True,
              snapshot_epoch=True,
              show_metric=True,
              batch_size=40,
              run_id=s.MODEL_NAME)
    # endregion

    # region SAVE
    model.save(s.MODEL_NAME)
    print('Network trained and saved as {0}'.format(s.MODEL_NAME))
    # endregion
