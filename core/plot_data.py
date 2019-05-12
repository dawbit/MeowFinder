import matplotlib.pyplot as plt
import numpy as np
import settings as s


def plt_dat(model, test_data):
    for num in range(len(test_data)):
        d = test_data[num]
        img_data, img_num = d

        data = img_data.reshape(s.IMG_SIZE, s.IMG_SIZE, 1)
        prediction = model.predict([data])[0]

        s.num_animals[np.argmax(prediction)] += 1

    fig = plt.figure(figsize=(16, 12))

    for num, data in enumerate(test_data[:64]):
        # img_num = data[1] # not used anywhere
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
