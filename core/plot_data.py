import matplotlib.pyplot as plt
import numpy as np
import settings as s
import tflearn

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing

tflearn.init_graph(num_cores=4, gpu_memory_fraction=0.5)

# region NETWORK
def cnn():
    # region NETWORK
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center(mean=[0.4735053442384178])

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

    model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='log')

    model.load(s.MODEL_NAME)

    return model
# endregion


def plt_dat(test_data):
    model = cnn()
    for num in range(len(test_data)):
        d = test_data[num]
        img_data, img_num = d

        data = img_data.reshape(s.IMG_SIZE, s.IMG_SIZE, 1)
        data = data / 255.0
        prediction = model.predict([data])[0]

        s.num_animals[np.argmax(prediction)] += 1

    fig = plt.figure(figsize=(12, 8))

    for num, data in enumerate(test_data[:20]):
        img_data = data[0]

        y = fig.add_subplot(4, 5, num + 1)
        orig = img_data
        data = img_data.reshape(s.IMG_SIZE, s.IMG_SIZE, 1)
        data = data / 255.0

        model_out = model.predict([data])[0]
        str_label = '{} {:.2f}%'.format(s.animals[np.argmax(model_out)], max(model_out)*100)

        y.imshow(orig, cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)

    for i in range(len(s.num_animals)):
        print(s.animals[i] + " : " + str(s.num_animals[i]))

    plt.show()
