import util
import pdb
import random
import numpy as np


def get_LinearKernel(dataset_1, dataset_2):
    dataset_1_size = dataset_1.shape[0]
    dataset_2_size = dataset_2.shape[0]
    data_dimension = dataset_1[0].shape[0]
    K = np.zeros((dataset_1_size, dataset_2_size))
    for i, data_1 in enumerate(dataset_1):
        for j, data_2 in enumerate(dataset_2):
            K[i][j] = np.dot(data_1, data_2)

    return K


def get_PolyKernel(dataset_1, dataset_2, gamma, power):
    dataset_1_size = dataset_1.shape[0]
    dataset_2_size = dataset_2.shape[0]
    data_dimension = dataset_1[0].shape[0]
    K = np.zeros((dataset_1_size, dataset_2_size))
    for i, data_1 in enumerate(dataset_1):
        for j, data_2 in enumerate(dataset_2):
            K[i][j] = np.power((1 + gamma * np.dot(data_1, data_2)), power)

    return K


def get_RBFKernel(dataset_1, dataset_2, gamma):
    dataset_1_size = dataset_1.shape[0]
    dataset_2_size = dataset_2.shape[0]
    data_dimension = dataset_1[0].shape[0]
    K = np.zeros((dataset_1_size, dataset_2_size))
    for i, data_1 in enumerate(dataset_1):
        for j, data_2 in enumerate(dataset_2):
            K[i][j] = np.exp(-1.0 * gamma * \
                        sum([(data_1[x] - data_2[x])**2 for x in range(data_dimension)]))

    return K


def train_KLR(K, training_y, learning_rate, iteration_times, lamba):
    data_size = K.shape[0]
    beta = np.zeros(data_size)
    for i in range(iteration_times):
        l2_reg = (2 * lamba / data_size) * np.dot(K, beta)
        T = np.array([util.sigmoid(data, label, beta) * (-1) * label * data
                      for data, label in zip(K, training_y)])
        T = l2_reg + np.sum(T, axis = 0) / data_size
        beta = beta - learning_rate * T

    return beta


def train_poly_KLR(K, training_y, weights, learning_rate, iteration_times, lamba):
    data_size = K.shape[0]
    beta = np.zeros(data_size)
    poly = np.poly1d(weights)
    for _ in range(iteration_times):
        l2_reg = (2 * lamba / data_size) * np.dot(K, beta)
        T = np.array([poly(-1 * label * np.dot(beta.T, data)) * (-1) * label * data
                      for data, label in zip(K, training_y)])
        T = l2_reg + np.sum(T, axis = 0) / data_size
        beta = beta - learning_rate * T

    return beta


def train_KRR(K, training_y, learning_rate):
    data_size = K.shape[0]
    return np.dot(np.linalg.inv(learning_rate * np.identity(data_size) + K), training_y)


def test_klr(beta, K_in, K_out, data):
    training_x = data['training_x']
    training_y = data['training_y']
    testing_x = data['testing_x']
    testing_y = data['testing_y']
    avg_in = util.avg_kernel(beta, K_in, training_y)
    avg_out = util.avg_kernel(beta, K_out, testing_y)
    print("avg_in = %s avg_out = %s " % (avg_in, avg_out))
