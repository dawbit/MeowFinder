import matplotlib.pyplot as plt
import numpy as np
import settings as s
import tflearn

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


# region NETWORK
def cnn():
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

    model.load(s.MODEL_NAME)

    return model
# endregion


def plt_dat(test_data):
    model = cnn()
    for num in range(len(test_data)):
        d = test_data[num]
        img_data, img_num = d

        data = img_data.reshape(s.IMG_SIZE, s.IMG_SIZE, 1)
        prediction = model.predict([data])[0]

        s.num_animals[np.argmax(prediction)] += 1

    fig = plt.figure(figsize=(16, 12))

    for num, data in enumerate(test_data[:64]):
        img_data = data[0]

        y = fig.add_subplot(8, 8, num + 1)
        orig = img_data
        data = img_data.reshape(s.IMG_SIZE, s.IMG_SIZE, 1)
        model_out = model.predict([data])[0]

        str_label = s.animals[np.argmax(model_out)]

        y.imshow(orig, cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)

    for i in range(len(s.num_animals)):
        print(s.animals[i] + " : " + str(s.num_animals[i]))

    plt.show()