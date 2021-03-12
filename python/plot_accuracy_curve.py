import util
import train
import read_data

import pdb
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression


#Original Load Weights
weights_dict = util.read_weight('./poly_sigmoid_weights/weights_100.out')
for k in weights_dict.keys():
    weights_dict[k].reverse()


# Data Settings 
# iteration_times = 40
dataset = read_data.read_make_circles()

training_x = dataset['training_x']
training_y = dataset['training_y']
testing_x = dataset['testing_x']
testing_y = dataset['testing_y']


# make_circles
classifiers_list = [{'model': 'LR', 'sigmoid_deg': 3, 'iteration_times': 600, 'learning_rate': 0.1},
                    {'model': 'KLR_linear', 'sigmoid_deg': 3, 'iteration_times': 600, 'learning_rate': 0.1, 'lamba': 0.01, 'kernel': 'linear'},
                    {'model': 'KLR_polynomial', 'sigmoid_deg': 3, 'iteration_times': 600, 'learning_rate': 0.1, 'lamba': 0.001, 'kernel': 'polynomial', 'poly_deg': 3, 'gamma': 1.0},
                    {'model': 'KLR_rbf', 'sigmoid_deg': 3, 'iteration_times': 600, 'learning_rate': 0.1, 'lamba': 0.001, 'kernel': 'rbf', 'gamma': 1.0, 'taylor_deg': 1}]


for clf in classifiers_list:
    model = clf['model']
    if model == 'LR':
        sigmoid_deg = clf['sigmoid_deg']
        learning_rate = clf['learning_rate']
        iteration_times = clf['iteration_times']

        W = train.train_model(False, dataset, learning_rate, iteration_times, 2, weights = weights_dict[str(sigmoid_deg)])
        avg_in, avg_out = util.test_model(W, dataset)
    else:
        iteration_times = clf['iteration_times']
        sigmoid_deg = clf['sigmoid_deg']
        learning_rate = clf['learning_rate']
        kernel = clf['kernel']
        lamba = clf['lamba']
        if kernel == 'linear':
            K_in = train.get_Kernel(training_x, training_x, kernel)
            K_out = train.get_Kernel(testing_x, training_x, kernel)
        elif kernel == 'polynomial':
            poly_deg = clf['poly_deg']
            gamma = clf['gamma']
            K_in = train.get_Kernel(training_x, training_x, kernel, gamma, poly_deg)
            K_out = train.get_Kernel(testing_x, training_x, kernel, gamma, poly_deg)
        elif kernel == 'rbf':
            gamma = clf['gamma']
            K_in = train.get_Kernel(training_x, training_x, kernel, gamma)
            K_out = train.get_Kernel(testing_x, training_x, kernel, gamma)
        
        dataset['training_x'] = K_in
        dataset['testing_x'] = K_out

        beta = train.train_model(True, dataset, learning_rate, iteration_times, 2, lamba, weights_dict[str(sigmoid_deg)])
        avg_in, avg_out = util.test_model(beta, dataset)

