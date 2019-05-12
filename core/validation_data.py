import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import settings as s


def create_validation_data():
    validation_data = []
    for img in tqdm(os.listdir(s.VALIDATION_DIR)):
        path = os.path.join(s.VALIDATION_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img, (s.IMG_SIZE, s.IMG_SIZE))
        validation_data.append([np.array(img_data), img_num])

    shuffle(validation_data)
    np.save('validation_data.npy', validation_data)
    return validation_data
