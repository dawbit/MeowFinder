import matplotlib.pyplot as plt
import numpy as np

#-------IMPORTOWANIE TEGO NIE DZIALA, HGW--------#

TRAIN_DIR = 'train'
TEST_DIR = 'test'
IMG_SIZE = 150
LR = 1e-3
MODEL_NAME = 'meowfinder-{}-{}'.format(LR, 'basic')


#-------IMPORTOWANIE TEGO NIE DZIALA, HGW--------#


def plt_dat(model, test_data):
    # for num in range(len(test_data)):
        # d = test_data[num]
        # img_data, img_num = d

        # data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        # prediction = model.predict([data])[0]

        # fig = plt.figure(figsize=(6, 6))
        # ax = fig.add_subplot(111)
        # ax.imshow(img_data, cmap="gray")
        # print(f"cat: {prediction[0]}, dog: {prediction[1]}")

    fig = plt.figure(figsize=(16, 12))

    cats = 0
    dogs = 0

    for num, data in enumerate(test_data[:64]):

        # img_num = data[1] # not used anywhere
        img_data = data[0]

        y = fig.add_subplot(8, 8, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]

        if np.argmax(model_out) == 1:
            str_label = 'Dog'
            dogs += 1
        else:
            str_label = 'Cat'
            cats += 1

        y.imshow(orig, cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)

    print("Dogs:", dogs, "\tCats:", cats)
    plt.show()
