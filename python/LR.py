import pdb
import json
import util
import random
import numpy as np
import matplotlib.pyplot as plt


def GD(training_x, training_y, learning_rate, iteration_times):
    data_size = len(training_x)
    data_dimension = len(training_x[0])
    W = np.zeros(data_dimension)
    accuracy = []
    for _ in range(iteration_times):
        T = np.zeros(data_dimension)
        for data, label in zip(training_x, training_y):
            t = util.sigmoid(data, label, W) * (-1) * label * data
            T = np.add(T, t)
        T = T / data_size
        W = W - learning_rate * T
        accuracy.append(W)

    plot_train_acc_curve(accuracy, training_x, training_y)
    return W


def ploy_GD(training_x, training_y, weights, learning_rate, iteration_times):
    data_size = len(training_x)
    data_dimension = len(training_x[0])
    weights = [-0.81562 / (8 ** 3), 0, 1.20096 / 8, 0.5]
    poly = np.poly1d(weights)
    W = np.zeros(data_dimension)
    accuracy = []
    for _ in range(iteration_times):
        T = np.zeros(data_dimension)
        for data, label in zip(training_x, training_y):
            t = poly(-1 * label * np.dot(W.T, data)) * (-1) * label * data
            T = np.add(T, t)
        T = T / data_size
        W = W - learning_rate * T
        accuracy.append(W)

    # plot_train_acc_curve(accuracy, training_x, training_y)

    return W


def ploy_GD_2(training_x, training_y, weights, learning_rate, iteration_times):
    data_size = len(training_x)
    data_dimension = len(training_x[0])
    poly = np.poly1d(weights)
    W = np.zeros(data_dimension)
    # W = np.random.uniform(-1, 1, data_dimension)
    weights.reverse()
    accuracy = []
    for _ in range(iteration_times):
        T = np.zeros(data_dimension)
        for data, label in zip(training_x, training_y):
            wx = np.dot(W.T, data)
            t = 0
            for i in range(len(weights)):
                t = t + weights[i] * ((-1 * label) ** (i + 1)) * (wx ** i) * data
            T = np.add(T, t)
        T = T / data_size
        W = W - learning_rate * T
        accuracy.append(W)

    plot_train_acc_curve(accuracy, training_x, training_y)
    
    return W


def SGD(training_x, training_y, data_size, data_dimension, learning_rate, iteration_times):
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


def test_lr(W, data):
    training_x = data['training_x']
    training_y = data['training_y']
    testing_x = data['testing_x']
    testing_y = data['testing_y']
    avg_in = util.avg_lr(W, training_x, training_y)
    avg_out = util.avg_lr(W, testing_x, testing_y)
    print("avg_in = %.3f avg_out = %.3f " % (avg_in, avg_out))


def plot_train_acc_curve(accuracy, training_x, training_y):
    # x-axis
    x = [i for i in range(len(accuracy))]
    
    # y-axis
    pred = [util.avg_lr(W, training_x, training_y) for W in accuracy]

    # plt
    plt.plot(x, pred, label='accuacy')

    plt.legend()
    plt.savefig('./accuracy.png')
