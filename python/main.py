import util
import train
import read_data

import pdb
import itertools
import numpy as np


# Load Datasets
# dataset = read_data.read_lbw() # use Z-score Standardization

dataset = read_data.read_uci_dataset('load_breast_cancer') # classification
# dataset = read_data.read_uci_dataset('load_iris') # classification
# dataset = read_data.read_uci_dataset('load_boston') # regression
# dataset = read_data.read_uci_dataset('load_diabetes') # regression

# dataset = read_data.read_make_circles()
# dataset = read_data.read_make_moons()

training_x = dataset['training_x']
training_y = dataset['training_y']
testing_x = dataset['testing_x']
testing_y = dataset['testing_y']

#Original Load Weights
weights_dict = util.read_weight('./poly_sigmoid_weights/weights_20.out')
for k in weights_dict.keys():
    weights_dict[k].reverse()

# Logistic Regression
print('---------- Logistic Regression ----------')
iteration_times = 20
learning_rate_list = [0.1, 0.01, 0.001]

for learning_rate in learning_rate_list:
    print('\n\n>>>>>   ', learning_rate, '   <<<<<')
    W = train.train_model(False, dataset, learning_rate, iteration_times, 0)
    util.test_model(W, dataset)

    # Polynomial Logistic Regression
    for deg in range(3, 10, 4):
        print('Poly degree = ', deg,)
        W = train.train_model(False, dataset, learning_rate, iteration_times, 2,  weights = weights_dict[str(deg)])
        util.test_model(W, dataset)


# KERNELS
# Parameters Settings
iteration_times = 100

learning_rate_list = [0.1, 0.5, 0.01]
lamba_list = [0.1, 0.01]
gamma_list = [1, 0.1]

# kernel_type_list = ['linear', 'polynomial', 'rbf']
kernel_type_list = ['rbf']
poly_deg = 3
taylor_deg = 2

for learning_rate, gamma, lamba, kernel in list(itertools.product(learning_rate_list, gamma_list, lamba_list, kernel_type_list)):
    print('\n\n>>>>>   ', learning_rate, lamba, gamma, '   <<<<<')
    print('---------- ', kernel, ' KERNEL ----------')
    if kernel == 'linear':
        K_in = train.get_Kernel(training_x, training_x, 'linear')
        K_out = train.get_Kernel(testing_x, training_x, 'linear')
    elif kernel == 'polynomial':
        K_in = train.get_Kernel(training_x, training_x, 'polynomial', gamma, poly_deg = poly_deg)
        K_out = train.get_Kernel(testing_x, training_x, 'polynomial', gamma, poly_deg = poly_deg)
    elif kernel == 'rbf':
        K_in = train.get_Kernel(training_x, training_x, 'rbf', gamma, taylor_deg = taylor_deg)
        K_out = train.get_Kernel(testing_x, training_x, 'rbf', gamma, taylor_deg = taylor_deg)

    dataset['training_x'] = K_in
    dataset['testing_x'] = K_out

    model = train.train_model(True, dataset, learning_rate, iteration_times, 0, lamba)
    util.test_model(model, dataset)

    for deg in range(3, 10, 4):
        print(deg, )
        model = train.train_model(True, dataset, learning_rate, iteration_times, 2, lamba, weights_dict[str(deg)])
        util.test_model(model, dataset)
