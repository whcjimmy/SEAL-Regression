import util
import LR
import KLR
import pdb
import itertools
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

# Load Datasets
# dataset = util.read_banknote()
# dataset = util.read_load_breast_cancer()
dataset = util.read_heart_disease()
# dataset = util.read_make_circles()
# dataset = util.read_make_moons()
# dataset = util.read_data_hw2('hw2_lssvm_all.dat')
# dataset = util.read_data_hw3('./hw3_train.dat', './hw3_test.dat')
# dataset = util.read_pulsar_stars() # use Z-score Standardization
# dataset = util.read_lbw() # use Z-score Standardization

training_x = dataset['training_x']
training_y = dataset['training_y']
testing_x = dataset['testing_x']
testing_y = dataset['testing_y']

# Feature Scaling (min-max normalization)
scaler = MinMaxScaler()
scaler.fit(training_x)
training_x = scaler.transform(training_x)
scaler.fit(testing_x)
testing_x = scaler.transform(testing_x)

dataset['training_x'] = training_x
dataset['testing_x'] = testing_x


#Original Load Weights
weights_dict = util.read_weight('./weights.out')
for k in weights_dict.keys():
    weights_dict[k].reverse()


# Parameters Settings
learning_rate = 0.01
iteration_times = 10

# Logistic Regression
print('---------- Logistic Regression ----------')
'''
print('Sklearn Logistic Regression')
clf = LogisticRegression(random_state=0).fit(training_x, training_y)
print(clf.score(training_x, training_y), clf.score(testing_x, testing_y))
print('Sklearn SGD Classifier')
clf = SGDClassifier(alpha = 10 * learning_rate).fit(training_x, training_y)
print(clf.score(training_x, training_y), clf.score(testing_x, testing_y))
'''
print('Original')
W = LR.train_LR(dataset, learning_rate, iteration_times, 0)
LR.test_lr(W, dataset)

# Polynomial Logistic Regression
for deg in range(3, 11, 4):
    print('Poly degree = ', deg,)
    W = LR.train_LR(dataset, learning_rate, iteration_times, 2, weights_dict[str(deg)])
    LR.test_lr(W, dataset)


# KERNELS
learning_rate = 0.004
iteration_times = 10

# Kernel Logistic Regression
print('---------- Kernel Logistic Regression ----------')
lamba = 0.15

'''
# Linear Kernel
print('---------- LINEAR KERNEL ----------')
K_in = KLR.get_Kernel(training_x, training_x, 'linear')
K_out = KLR.get_Kernel(training_x, testing_x, 'linear')

beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 0, lamba)
KLR.test_klr(beta, K_in, K_out, dataset)

for deg in range(3, 11, 4):
    print(deg)
    beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 2, lamba, weights_dict[str(deg)])
    KLR.test_klr(beta, K_in, K_out, dataset)


# Polynomial Kernel
print('---------- POLYNOMIAL KERNEL ----------')
learning_rate = 0.001
gamma = 0.8
lamba = 0.001
power = 3

K_in = KLR.get_Kernel(training_x, training_x, 'polynomial', gamma, power)
K_out = KLR.get_Kernel(training_x, testing_x, 'polynomial', gamma, power)

beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 0, lamba)
KLR.test_klr(beta, K_in, K_out, dataset)

for deg in range(3, 11, 4):
    print(deg)
    beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 2, lamba, weights_dict[str(deg)])
    KLR.test_klr(beta, K_in, K_out, dataset)


# RBF Kernel
print('---------- RBF KERNEL ----------')
K_in = KLR.get_Kernel(training_x, training_x, 'rbf', gamma)
K_out = KLR.get_Kernel(training_x, testing_x, 'rbf', gamma)

beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 0, lamba)
KLR.test_klr(beta, K_in, K_out, dataset)

for deg in range(3, 11, 4):
    print(deg)
    beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 2, lamba, weights_dict[str(deg)])
    KLR.test_klr(beta, K_in, K_out, dataset)


# Kernel Ridge Regression
print('---------- Kernel Ridge Regression ----------')
beta = KLR.train_KRR(K_in, training_y, lamba)
KLR.test_klr(beta, K_in, K_out, dataset)
'''


gamma_list = [0.1, 0.5, 0.01, 0.05, 0.001, 0.0001]
lamba_list = [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.0001]
power = 3
for gamma, lamba in list(itertools.product(gamma_list, lamba_list)):
    print(gamma, lamba)
    print('---------- POLYNOMIAL KERNEL ----------')
    K_in = KLR.get_Kernel(training_x, training_x, 'polynomial', gamma, power)
    K_out = KLR.get_Kernel(training_x, testing_x, 'polynomial', gamma, power)

    beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 0, lamba)
    KLR.test_klr(beta, K_in, K_out, dataset)

    for deg in range(3, 11, 4):
        print(deg)
        beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 2, lamba, weights_dict[str(deg)])
        KLR.test_klr(beta, K_in, K_out, dataset)


    print('---------- RBF KERNEL ----------')
    K_in = KLR.get_Kernel(training_x, training_x, 'rbf', gamma)
    K_out = KLR.get_Kernel(training_x, testing_x, 'rbf', gamma)

    beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 0, lamba)
    KLR.test_klr(beta, K_in, K_out, dataset)
   
    for deg in range(3, 11, 4):
        print(deg)
        beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 2, lamba, weights_dict[str(deg)])
        KLR.test_klr(beta, K_in, K_out, dataset)
