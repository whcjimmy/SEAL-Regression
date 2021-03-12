import pdb
import util
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score


def get_Kernel(dataset_1, dataset_2, kernel_type = None, gamma = None, poly_deg = None, taylor_deg = None):
    dataset_1_size = dataset_1.shape[0]
    dataset_2_size = dataset_2.shape[0]
    data_dimension = dataset_1[0].shape[0]
    K = np.zeros((dataset_1_size, dataset_2_size))
    for i, data_1 in enumerate(dataset_1):
        for j, data_2 in enumerate(dataset_2):
            if kernel_type == 'linear':
                K[i][j] = np.dot(data_1, data_2)
            elif kernel_type == 'polynomial':
                K[i][j] = np.power((1 + gamma * np.dot(data_1, data_2)), poly_deg)
            elif kernel_type == 'rbf':
                tmp = -1.0 * gamma * sum([(data_1[x] - data_2[x])**2 for x in range(data_dimension)])
                if taylor_deg == 2:
                    K[i][j] = 1 + tmp + (tmp ** 2) / 2 
                elif taylor_deg == 4:
                    K[i][j] = 1 + tmp + (tmp ** 2) / 2 + (tmp ** 3) / 6 + (tmp ** 4) / 24
                else:
                    K[i][j] = np.exp(-1.0 * gamma * sum([(data_1[x] - data_2[x])**2 for x in range(data_dimension)]))
    return K


# Train Model
def train_model(is_kernel, dataset, learning_rate, iteration_times, approx, lamba = None, weights = None):
    training_x = dataset['training_x']
    training_y = dataset['training_y']

    data_dimension = training_x[0].shape[0]
    model = np.zeros(data_dimension)
    model_list = []
    for i in range(iteration_times):
        T = []
        for data, label in zip(training_x, training_y):
            t = util.approx_func(approx, model, data, label, weights) * (-1) * label * data
            T.append(t)

        if is_kernel == True:
            l2_reg = (2 * lamba / data_dimension) * np.dot(training_x, model)
            T = l2_reg + np.sum(T, axis = 0) / data_dimension
        else:
            T = np.sum(T, axis = 0) / data_dimension

        model = model - learning_rate * T
        model_list.append(model)

    if is_kernel is True:
        params = 'KLR_%s_%s_%s' % (learning_rate, iteration_times, lamba)
    else:
        params = 'LR_%s_%s' %(learning_rate, iteration_times)

    util.plot_train_acc_curve(model_list, dataset, params)

    return model


'''
# Logistic Regression 
def train_LR(dataset, learning_rate, iteration_times, approx, weights = None):
    params = 'LR_%s_%s' %(learning_rate, iteration_times)
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

    util.plot_train_acc_curve(model_list, dataset, params)
    return W


# Kernel Logistic Regression

def train_KLR(dataset, learning_rate, iteration_times, approx, lamba, weights = None):
    params = 'KLR_%s_%s_%s' % (learning_rate, iteration_times, lamba)
    training_x = dataset['training_x']
    training_y = dataset['training_y']

    data_size = training_x.shape[0]
    beta = np.zeros(data_size)
    model_list = []
    for i in range(iteration_times):
        l2_reg = (2 * lamba / data_size) * np.dot(training_x, beta)
        T = []
        for data, label in zip(training_x, training_y):
            t = util.approx_func(approx, beta, data, label, weights) * (-1) * label * data
            T.append(t)
        T = l2_reg + np.sum(T, axis = 0) / data_size
        beta = beta - learning_rate * T
        model_list.append(beta)

    util.plot_train_acc_curve(model_list, dataset, params)

    return beta
'''
