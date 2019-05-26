import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
import settings as s
from tflearn.data_utils import image_preloader


tflearn.init_graph(num_cores=4, gpu_memory_fraction=0.5)


def network1(train, train_amount):
    global model
    # region SPLIT DATA FOR TRAIN/VALIDATION
    train_amount = int(train_amount * 5.5 / 6)

    x_train = np.array([i[0] for i in train[:train_amount]]).reshape(-1, s.IMG_SIZE, s.IMG_SIZE, 1)
    x_train = x_train / 255.0
    y_train = [i[1] for i in train[:train_amount]]

    x_validation = np.array([i[0] for i in train[train_amount:]]).reshape(-1, s.IMG_SIZE, s.IMG_SIZE, 1)
    x_validation = x_validation / 255.0
    y_validation = [i[1] for i in train[train_amount:]]
    # endregion

    # region NETWORK
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center(mean=[0.47938])

    network = input_data(shape=[None, s.IMG_SIZE, s.IMG_SIZE, 1], name='input', data_preprocessing=img_prep)

    network = conv_2d(network, 32, 3, activation='relu', scope='conv1_1')
    network = conv_2d(network, 64, 3, activation='relu', scope='conv1_2')
    network = max_pool_2d(network, 2, strides=2, name='maxpool_1')

    network = conv_2d(network, 128, 3, activation='relu', scope='conv2_1')
    network = max_pool_2d(network, 2, strides=2, name='maxpool_2')

    network = conv_2d(network, 128, 3, activation='relu', scope='conv3_1')
    network = max_pool_2d(network, 2, strides=2, name='maxpool_3')

    network = conv_2d(network, 256, 3, activation='relu', scope='conv4_1')
    network = max_pool_2d(network, 2, strides=2, name='maxpool_4')

    network = fully_connected(network, 1024, activation='relu', scope='fc5')
    network = dropout(network, 0.5, name='dropout_1')

    network = fully_connected(network, 1024, activation='relu', scope='fc6')
    network = dropout(network, 0.5, name='dropout_2')

    network = fully_connected(network, s.len_animals, activation='softmax', scope='fc7')

    network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=s.LR, name='targets')

    model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='log', checkpoint_path='model.tfl.ckpt')
    # endregion

    # region TRAIN
    model.fit(x_train, y_train, n_epoch=15,
              validation_set=({'input': x_validation}, {'targets': y_validation}),
              shuffle=True,
              snapshot_epoch=True,
              show_metric=True,
              batch_size=100,
              run_id=s.MODEL_NAME)
    # endregion

    # region SAVE
    model.save(s.MODEL_NAME)
    print('Network trained and saved as {0}'.format(s.MODEL_NAME))
    # endregion

