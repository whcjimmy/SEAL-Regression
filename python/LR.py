import pdb
import json
import util
import random
import numpy as np
import matplotlib.pyplot as plt


def train_LR(dataset, learning_rate, iteration_times, approx, weights = None):
    training_x = dataset['training_x']
    training_y = dataset['training_y']
    data_size = len(training_x)
    data_dimension = len(training_x[0])
    W = np.zeros(data_dimension)
    model_list = []
    for _ in range(iteration_times):
        T = []
        for data, label in zip(training_x, training_y):
            t = util.approx_func(approx, W, data, label, weights) * (-1) * label * data
            T.append(t)
        T = np.sum(T, axis = 0) / data_size
        W = W - learning_rate * T
        model_list.append(W)

    # plot_train_acc_curve(model_list, dataset)
    return W


def train_SGD(dataset, data_size, data_dimension, learning_rate, iteration_times):
    training_x = dataset['training_x']
    training_y = dataset['training_y']
    W = np.zeros(data_dimension)
    itr = 0
    for _ in range(iteration_times):
        if itr == data_size: itr = 0
        data = training_x[itr]
        label = training_y[itr]
        t = sigmoid(data, label, W) * (-1) * label * data

        W = W - learning_rate * t
        itr = itr + 1

    return W


def avg_lr(W, data, labels):
    pred = np.dot(data, W.T)
    return np.mean(np.sign(pred) == labels)


def test_lr(W, data):
    training_x = data['training_x']
    training_y = data['training_y']
    testing_x = data['testing_x']
    testing_y = data['testing_y']
    avg_in = avg_lr(W, training_x, training_y)
    avg_out = avg_lr(W, testing_x, testing_y)
    print("avg_in = %.3f avg_out = %.3f " % (avg_in, avg_out))
    return avg_in, avg_out


def plot_train_acc_curve(model_list, dataset):
    # x-axis
    label_x = [i for i in range(len(model_list))]
    
    # y-axis
    training_x = dataset['training_x']
    training_y = dataset['training_y']
    pred_y = [avg_lr(W, training_x, training_y) for W in model_list]

    # plt
    plt.plot(label_x, pred_y, label='accuacy')

    plt.legend()
    plt.savefig('./accuracy.png')
