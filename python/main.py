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
dataset = util.read_make_circles()
# dataset = util.read_make_moons()
# dataset = util.read_data_hw2('hw2_lssvm_all.dat')
# dataset = util.read_data_hw3('./hw3_train.dat', './hw3_test.dat')
# dataset = util.read_pulsar_stars() # use Z-score Standardization
# dataset = util.read_lbw() # use Z-score Standardization

training_x = dataset['training_x']
training_y = dataset['training_y']
testing_x = dataset['testing_x']
testing_y = dataset['testing_y']


# Normalize
# training_x_col_sums = np.sum(training_x, axis = 0)
# training_x_norm = training_x / training_x_col_sums[np.newaxis, :]

'''
# Feature Scaling (Standardization)
training_x = stats.zscore(training_x)
testing_x  = stats.zscore(testing_x)
'''

'''
# Feature Scaling (min-max normalization)
scaler = MinMaxScaler()
scaler.fit(training_x)
training_x = scaler.transform(training_x)
scaler.fit(testing_x)
testing_x = scaler.transform(testing_x)
'''

dataset['training_x'] = training_x
dataset['testing_x'] = testing_x

#Original Load Weights
weights_dict = util.read_weight('./weights.out')
for k in weights_dict.keys():
    weights_dict[k].reverse()


# Parameters Settings
learning_rate = 0.005
iteration_times = 15

# Logistic Regression
print('---------- Logistic Regression ----------')
'''
print('Sklearn Logistic Regression')
clf = LogisticRegression(random_state=0).fit(training_x, training_y)
print(clf.score(training_x, training_y), clf.score(testing_x, testing_y))
print('Sklearn SGD Classifier')
clf = SGDClassifier(alpha = 10 * learning_rate).fit(training_x, training_y)
print(clf.score(training_x, training_y), clf.score(testing_x, testing_y))
print('Original')
W = LR.train_LR(dataset, learning_rate, iteration_times, 0)
LR.test_lr(W, dataset)
'''

# Polynomial Logistic Regression
for deg in range(3, 11, 4):
    print('Poly degree = ', deg,)
    W = LR.train_LR(dataset, learning_rate, iteration_times, 2, weights_dict[str(deg)])
    LR.test_lr(W, dataset)


# KERNELS
gamma_list = [32, 2, 0.125]
lambda_list = [0.001, 0.01, 0.1, 1, 10]
gamma = 0.125
lamba = 10
power = 2

# learning_rate = 0.01

# Kernel Logistic Regression
print('---------- Kernel Logistic Regression ----------')
# Linear Kernel
print('---------- LINEAR KERNEL ----------')
K_in = KLR.get_Kernel(training_x, training_x, 'linear')
K_out = KLR.get_Kernel(training_x, testing_x, 'linear')

'''
beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 0, lamba)
KLR.test_klr(beta, K_in, K_out, dataset)
'''

for deg in range(3, 11, 4):
    print(deg)
    beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 2, lamba, weights_dict[str(deg)])
    KLR.test_klr(beta, K_in, K_out, dataset)


# Polynomial Kernel
print('---------- POLYNOMIAL KERNEL ----------')
K_in = KLR.get_Kernel(training_x, training_x, 'polynomial', gamma, power)
K_out = KLR.get_Kernel(training_x, testing_x, 'polynomial', gamma, power)

'''
beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 0, lamba)
KLR.test_klr(beta, K_in, K_out, dataset)
'''

for deg in range(3, 11, 4):
    print(deg)
    beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 2, lamba, weights_dict[str(deg)])
    KLR.test_klr(beta, K_in, K_out, dataset)

# RBF Kernel
print('---------- RBF KERNEL ----------')
K_in = KLR.get_Kernel(training_x, training_x, 'rbf', gamma)
K_out = KLR.get_Kernel(training_x, testing_x, 'rbf', gamma)

'''
beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 0, lamba)
KLR.test_klr(beta, K_in, K_out, dataset)
'''

for deg in range(3, 11, 4):
    print(deg)
    beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 2, lamba, weights_dict[str(deg)])
    KLR.test_klr(beta, K_in, K_out, dataset)

'''
# Kernel Ridge Regression
for gamma, my_lambda in list(itertools.product(gamma_list, lambda_list)):
    K_in = KLR.get_RBFKernel(training_x, training_x, gamma)
    K_out = KLR.get_RBFKernel(training_x, testing_x, gamma)
    beta = KLR.train_KRR(K_in, training_y, my_lambda)
    KLR.test_klr(beta, K_in, K_out, data)
'''
