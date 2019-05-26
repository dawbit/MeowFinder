import numpy as np
import os
import sys
import train_data
import test_data
import neural_network1
import plot_data


# if os.path.isfile('train_data.npy'):
#     train_data = np.load('train_data.npy')
#     train_amount = len(train_data)
#     if os.path.isfile('test_data.npy'):
#         test_data = np.load('test_data.npy')
#     else:
#         test_data.create_test_data()
# else:
#     print("Nie istnieje")
#     train_data, train_amount = train_data.create_train_data()
#     test_data = test_data.create_test_data()
#
# model = neural_network1.network1(train_data, train_amount)
# plot_data.plt_dat(model, test_data)

def decision(argument):
    if argument.isnumeric() is False:
        sys.exit(1)

    if argument == "1":
        createModel()
    elif argument == "2":
        makePredictions()
    else:
        both()


def createModel():
    if os.path.isfile('train_data.npy'):
        inp = input("Train data exist. Want to remove?")
        if inp == "yes" or inp == "y" or inp == "true":
            os.remove("train_data.npy")
            tra_data = train_data.create_train_data()
        else:
            tra_data = np.load('train_data.npy')
    else:
        tra_data = train_data.create_train_data()

    train_amount = len(tra_data)
    neural_network1.network1(tra_data, train_amount)


def makePredictions():
    if os.path.isfile('test_data.npy'):
        inp = input("Test data exist. Want to remove?")
        if inp == "yes" or inp == "y" or inp == "true":
            os.remove("test_data.npy")
            tes_data = test_data.create_test_data()
        else:
            tes_data = np.load('test_data.npy')
    else:
        tes_data = test_data.create_test_data()

    plot_data.plt_dat(tes_data)


def both():
    createModel()
    makePredictions()


if __name__ == "__main__":
    print("1. Create a model based on train dir data.")
    print("2. Make predictions based on test dir data.")
    print("3. Create a model and make preditions.")
    choice = input("Choice: ")

    decision(choice)
