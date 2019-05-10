import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import settings as s


def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(s.TEST_DIR)):
        path = os.path.join(s.TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img, (s.IMG_SIZE, s.IMG_SIZE))
        testing_data.append([np.array(img_data), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

