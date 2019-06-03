import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
# from PIL import Image
import settings as s


# def label_img(image_name):
#     for i in range(s.len_animals):
#         if s.animals[i] in image_name:
#             class_label = np.zeros(s.len_animals)
#             class_label[i] = 1
#             return class_label

def create_train_data():
    training_data = []
    for animal in s.animals:
        print('Current animal:', animal, '\n')
        for img in tqdm(os.listdir(s.TRAIN_DIR + '/' + animal)):
            label = np.zeros(s.len_animals)
            label[s.animals.index(animal)] = 1
            path = os.path.join(s.TRAIN_DIR + '/' + animal, img)

            # region PNG TO JPG
            # pil_img = Image.open(path)
            # if pil_img.mode == 'P':
            #     pil_img = pil_img.convert('RGB')
            # np_image = np.array(pil_img)
            #
            # img = cv2.cvtColor(np_image, cv2.IMREAD_GRAYSCALE)
            # endregion

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (s.IMG_SIZE, s.IMG_SIZE))
            training_data.append([np.array(img), np.array(label)])

            # img = Image.open(path)
            # img_arr = np.array(img)
            #
            # img = cv2.resize(img_arr, (125, 125))
            # training_data.append([[np.array(img)], np.array(label)])

    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data
