import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import settings as s
from PIL import Image
from io import BytesIO


def label_img(image_name):
    for i in range(s.len_animals):
        if s.animals[i] in image_name:
            class_label = np.zeros(s.len_animals)
            class_label[i] = 1
            return class_label


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(s.TRAIN_DIR)):
        # label = label_img(img)
        # path = os.path.join(s.TRAIN_DIR, img)
        #
        # with open(path, 'rb') as img_bin:
        #     buff = BytesIO()
        #     buff.write(img_bin.read())
        #     buff.seek(0)
        #     temp_img = np.array(Image.open(buff), dtype=np.uint8)
        #     img = cv2.cvtColor(temp_img, cv2.IMREAD_GRAYSCALE)

        label = label_img(img)
        path = os.path.join(s.TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (s.IMG_SIZE, s.IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])

    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data
