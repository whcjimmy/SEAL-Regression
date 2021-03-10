import util
import LR
import KLR
import pdb
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

# Load Datasets
# dataset = util.read_lbw() # use Z-score Standardization
# dataset = util.read_pulsar_stars() # use Z-score Standardization
# dataset = util.read_banknote()
# dataset = util.read_heart_disease()
# dataset = util.read_load_breast_cancer()
dataset = util.read_make_circles()
# dataset = util.read_make_moons()

training_x = dataset['training_x']
training_y = dataset['training_y']
testing_x = dataset['testing_x']
testing_y = dataset['testing_y']


#Original Load Weights
weights_dict = util.read_weight('./weights.out')
for k in weights_dict.keys():
    weights_dict[k].reverse()


'''
# Parameters Settings
learning_rate = 0.1
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
W = LR.train_LR(dataset, learning_rate, iteration_times, 0)
LR.test_lr(W, dataset)

# Polynomial Logistic Regression
for deg in range(3, 10, 2):
    print('Poly degree = ', deg,)
    W = LR.train_LR(dataset, learning_rate, iteration_times, 2, weights_dict[str(deg)])
    LR.test_lr(W, dataset)
'''

# KERNELS
learning_rate = 0.001
iteration_times = 20

'''
learning_rate_list = [0.1, 0.01, 0.001]
gamma_list = [1, 0.5, 0.1, 0.01]
lamba_list = [1, 0.5, 0.1, 0.01]
'''
learning_rate_list = [1, 0.1, 0.01]
gamma_list = [1, 0.1]
lamba_list = [1, 0.1]

# kernel_type_list = ['linear', 'polynomial', 'rbf']
kernel_type_list = ['polynomial']
poly_deg = 2

for learning_rate, gamma, lamba, kernel in list(itertools.product(learning_rate_list, gamma_list, lamba_list, kernel_type_list)):
    print('\n\n>>>>>   ', learning_rate, lamba, gamma, '   <<<<<')
    print('---------- ', kernel, ' KERNEL ----------')
    if kernel == 'linear':
        K_in = KLR.get_Kernel(training_x, training_x, 'linear')
        K_out = KLR.get_Kernel(training_x, testing_x, 'linear')
    elif kernel == 'polynomial':
        K_in = KLR.get_Kernel(training_x, training_x, 'polynomial', gamma, poly_deg)
        K_out = KLR.get_Kernel(training_x, testing_x, 'polynomial', gamma, poly_deg)
    elif kernel == 'rbf':
        K_in = KLR.get_Kernel(training_x, training_x, 'rbf', gamma)
        K_out = KLR.get_Kernel(training_x, testing_x, 'rbf', gamma)

    beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 0, lamba)
    KLR.test_klr(beta, K_in, K_out, dataset)

    for deg in range(3, 10, 1):
        print(deg, )
        beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 2, lamba, weights_dict[str(deg)])
        KLR.test_klr(beta, K_in, K_out, dataset)
