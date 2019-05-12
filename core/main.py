import numpy as np
import os
import train_data
import test_data
import validation_data
import neural_network1
import plot_data

if os.path.isfile('train_data.npy'):
    train_data = np.load('train_data.npy')
    train_amount = len(train_data)
    if os.path.isfile('test_data.npy'):
        test_data = np.load('test_data.npy')
    else:
        test_data.create_test_data()
else:
    print("Nie istnieje")
    train_data, train_amount = train_data.create_train_data()
    test_data = test_data.create_test_data()

model = neural_network1.network1(train_data, test_data, train_amount)
validation = validation_data.create_validation_data()
plot_data.plt_dat(model, test_data)
