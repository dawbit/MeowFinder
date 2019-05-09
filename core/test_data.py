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
        kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(img, -1, kernel_sharpening)
        blurred = cv2.GaussianBlur(sharpened, (3, 3), 0)
        img_data = cv2.Laplacian(blurred, cv2.CV_64F, ksize=7)
        img_data = cv2.resize(img_data, (s.IMG_SIZE, s.IMG_SIZE))
        testing_data.append([np.array(img_data), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data
