import util
import pdb
import random
import numpy as np
from sklearn.metrics import f1_score


def get_Kernel(dataset_1, dataset_2, kernel_type = None, gamma = None, power = None):
    dataset_1_size = dataset_1.shape[0]
    dataset_2_size = dataset_2.shape[0]
    data_dimension = dataset_1[0].shape[0]
    K = np.zeros((dataset_1_size, dataset_2_size))
    for i, data_1 in enumerate(dataset_1):
        for j, data_2 in enumerate(dataset_2):
            if kernel_type == 'linear':
                K[i][j] = np.dot(data_1, data_2)
            elif kernel_type == 'polynomial':
                K[i][j] = np.power((1 + gamma * np.dot(data_1, data_2)), power)
            elif kernel_type == 'rbf':
                tmp = -1.0 * gamma * sum([(data_1[x] - data_2[x])**2 for x in range(data_dimension)])
                K[i][j] = 1 + tmp + (tmp ** 2) / 2 
                # K[i][j] = 1 + tmp + (tmp ** 2) / 2 + (tmp ** 3) / 6 + (tmp ** 4) / 24
                # K[i][j] = np.exp(-1.0 * gamma * sum([(data_1[x] - data_2[x])**2 for x in range(data_dimension)]))
    return K


def train_KLR(K, training_y, learning_rate, iteration_times, approx, lamba, weights = None):
    data_size = K.shape[0]
    beta = np.zeros(data_size)
    for i in range(iteration_times):
        l2_reg = (2 * lamba / data_size) * np.dot(K, beta)
        T = []
        for data, label in zip(K, training_y):
            t = util.approx_func(approx, beta, data, label, weights) * (-1) * label * data
            T.append(t)
        T = l2_reg + np.sum(T, axis = 0) / data_size
        beta = beta - learning_rate * T
        # print(beta)

    return beta


def train_KRR(K, training_y, learning_rate):
    data_size = K.shape[0]
    return np.dot(np.linalg.inv(learning_rate * np.identity(data_size) + K), training_y)


def avg_kernel(beta, K, labels):
    pred = np.dot(K.T, beta)
    return np.mean(np.sign(pred) == labels)


def test_klr(beta, K_in, K_out, data):
    training_y = data['training_y']
    testing_y = data['testing_y']
    avg_in = avg_kernel(beta, K_in, training_y)
    avg_out = avg_kernel(beta, K_out, testing_y)
    # print("training f1 ", f1_score(np.sign(np.dot(K_in.T, beta)), training_y))
    # print("testing f1 ", f1_score(np.sign(np.dot(K_out.T, beta)), testing_y))
    print("avg_in = %.3f avg_out = %.3f " % (avg_in, avg_out))
    return avg_in, avg_out
