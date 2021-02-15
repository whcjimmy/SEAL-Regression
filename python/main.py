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
data = util.read_banknote()
# data = util.read_load_breast_cancer()
# data = util.read_data_hw2('hw2_lssvm_all.dat')
# data = util.read_data_hw3('./hw3_train.dat', './hw3_test.dat')
# data = util.read_pulsar_stars() # use Z-score Standardization
# data = util.read_lbw() # use Z-score Standardization

training_x = data['training_x']
training_y = data['training_y']
testing_x = data['testing_x']
testing_y = data['testing_y']


# Normalize
# training_x_col_sums = np.sum(training_x, axis = 0)
# training_x_norm = training_x / training_x_col_sums[np.newaxis, :]

# Feature Scaling (Standardization)
training_x = stats.zscore(training_x)
testing_x  = stats.zscore(testing_x)

# Feature Scaling (min-max normalization)
'''
scaler = MinMaxScaler()
scaler.fit(training_x)
training_x = scaler.transform(training_x)
scaler.fit(testing_x)
testing_x = scaler.transform(testing_x)
'''

data['training_x'] = training_x
data['testing_x'] = testing_x


#Original Load Weights
weights_dict = util.read_weight('./weights.out')

# Parameters Settings
learning_rate = 0.01
iteration_times = 20

# Logistic Regression
print('---------- Logistic Regression ----------')
print('Sklearn Logistic Regression')
clf = LogisticRegression(random_state=0).fit(training_x, training_y)
print(clf.score(training_x, training_y), clf.score(testing_x, testing_y))
print('Sklearn SGD Classifier')
clf = SGDClassifier(alpha = 10 * learning_rate).fit(training_x, training_y)
print(clf.score(training_x, training_y), clf.score(testing_x, testing_y))
print('Original')
W = LR.GD(training_x, training_y, learning_rate, iteration_times)
LR.test_lr(W, data)

# Polynomial Logistic Regression
for deg in range(3, 11, 4):
    W = LR.ploy_GD(training_x, training_y, weights_dict[str(deg)], learning_rate, iteration_times)
    print('Poly degree = ', deg,)
    LR.test_lr(W, data)
    '''
    W = LR.ploy_GD_2(training_x, training_y, weights_dict[str(deg)], learning_rate, iteration_times)
    print('degree = ', deg,)
    LR.test_lr(W, data)
    '''

# KERNELS
gamma_list = [32, 2, 0.125]
lambda_list = [0.001, 0.01, 0.1, 1, 10]
gamma = 0.125
lamba = 0.01
power = 2

learning_rate = 0.01

'''
# Kernel Logistic Regression
print('---------- Kernel Logistic Regression ----------')
# Linear Kernel
K_in = KLR.get_LinearKernel(training_x, training_x)
K_out = KLR.get_LinearKernel(training_x, testing_x)

beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, lamba)
KLR.test_klr(beta, K_in, K_out, data)
for deg in range(2, 10, 6):
    print(deg)
    beta = KLR.train_poly_KLR(K_in, training_y, weights_dict[str(deg)], learning_rate, iteration_times, lamba)
    KLR.test_klr(beta, K_in, K_out, data)
'''


'''
# Polynomial Kernel
K_in = KLR.get_PolyKernel(training_x, training_x, gamma, power)
K_out = KLR.get_PolyKernel(training_x, testing_x, gamma, power)

beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, lamba)
KLR.test_klr(beta, K_in, K_out, data)
'''

'''
# RBF Kernel
K_in = KLR.get_RBFKernel(training_x, training_x, gamma)
K_out = KLR.get_RBFKernel(training_x, testing_x, gamma)

beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, lamba)
KLR.test_klr(beta, K_in, K_out, data)

for deg in range(2, 10, 6):
    print(deg)
    beta = KLR.train_poly_KLR(K_in, training_y, weights_dict[str(deg)], learning_rate, iteration_times, lamba)
    KLR.test_klr(beta, K_in, K_out, data)
'''

'''
# Kernel Ridge Regression
for gamma, my_lambda in list(itertools.product(gamma_list, lambda_list)):
    K_in = KLR.get_RBFKernel(training_x, training_x, gamma)
    K_out = KLR.get_RBFKernel(training_x, testing_x, gamma)
    beta = KLR.train_KRR(K_in, training_y, my_lambda)
    KLR.test_klr(beta, K_in, K_out, data)
'''
