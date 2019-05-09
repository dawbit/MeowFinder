import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import settings as s


def label_img(image_name):
    # word_label = image_name.split('.')[0]  # not used anywhere
    #if word_label == 'cat': return np.array([1, 0])
    if 'cat' in image_name: return np.array([1, 0])
    #elif word_label == 'dog': return np.array([0, 1])
    elif 'dog' in image_name: return np.array([0, 1])
    else: exit(10)


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(s.TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(s.TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(img, -1, kernel_sharpening)
        blurred = cv2.GaussianBlur(sharpened, (3, 3), 0)
        img = cv2.Laplacian(blurred, cv2.CV_64F, ksize=7)
        img = cv2.resize(img, (s.IMG_SIZE, s.IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data, len(tqdm(os.listdir(s.TRAIN_DIR)))
