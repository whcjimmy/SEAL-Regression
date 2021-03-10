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


# Parameters Settings
learning_rate = 0.1
iteration_times = 20


'''
# Logistic Regression
print('---------- Logistic Regression ----------')
print('Sklearn Logistic Regression')
clf = LogisticRegression(random_state=0).fit(training_x, training_y)
print(clf.score(training_x, training_y), clf.score(testing_x, testing_y))
pdb.set_trace()
print('Sklearn SGD Classifier')
clf = SGDClassifier(alpha = 10 * learning_rate).fit(training_x, training_y)
print(clf.score(training_x, training_y), clf.score(testing_x, testing_y))
print('Original')
W = LR.train_LR(dataset, learning_rate, iteration_times, 0)
print(W)
LR.test_lr(W, dataset)

# Polynomial Logistic Regression
for deg in range(3, 11, 4):
    print('Poly degree = ', deg,)
    W = LR.train_LR(dataset, learning_rate, iteration_times, 2, weights_dict[str(deg)])
    # print(W)
    LR.test_lr(W, dataset)
'''

'''
# KERNELS
learning_rate = 0.001
iteration_times = 20

learning_rate_list = [0.1, 0.05, 0.01, 0.005, 0.001]
gamma_list = [0.1, 0.01]
lamba_list = [0.1, 0.01]

# kernel_type_list = ['linear', 'polynomial', 'rbf']
kernel_type_list = ['rbf']
power = 3

for learning_rate, gamma, lamba, kernel in list(itertools.product(learning_rate_list, gamma_list, lamba_list, kernel_type_list)):
    print('\n\n>>>>>   ', learning_rate, gamma, lamba, '   <<<<<')

    print('---------- ', kernel, ' KERNEL ----------')
    if kernel == 'linear':
        K_in = KLR.get_Kernel(training_x, training_x, 'linear')
        K_out = KLR.get_Kernel(training_x, testing_x, 'linear')
    elif kernel == 'polynomial':
        K_in = KLR.get_Kernel(training_x, training_x, 'polynomial', gamma, power)
        K_out = KLR.get_Kernel(training_x, testing_x, 'polynomial', gamma, power)
    elif kernel == 'rbf':
        K_in = KLR.get_Kernel(training_x, training_x, 'rbf', gamma)
        K_out = KLR.get_Kernel(training_x, testing_x, 'rbf', gamma)


    beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 0, lamba)
    KLR.test_klr(beta, K_in, K_out, dataset)

    for deg in range(3, 11, 4):
        print(deg, )
        beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 
                             2, lamba, weights_dict[str(deg)])
        KLR.test_klr(beta, K_in, K_out, dataset)
'''

dataset_list = [util.read_make_circles(), util.read_make_moons()]
classifiers_list = ['LR', 'KLR']


h = 0.2
plot_idx = len(dataset_list) * (len(classifiers_list) + 1)
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

figure = plt.figure(figsize=(15, 9))

for ds_cnt, dataset in enumerate(dataset_list):
    print(plot_idx)
    training_x = dataset['training_x']
    training_y = dataset['training_y']
    testing_x = dataset['testing_x']
    testing_y = dataset['testing_y']

    X = np.concatenate((training_x, testing_x))

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))


    ax = plt.subplot(len(dataset_list), len(classifiers_list) + 1, plot_idx)
    if ds_cnt == 0:
        ax.set_title('input data')

    ax.scatter(training_x[:,0], training_x[:,1], c = training_y, cmap = cm_bright, edgecolors = 'k')
    ax.scatter(testing_x[:,0], testing_x[:,1], c = testing_y, alpha = 0.6, cmap = cm_bright, edgecolors = 'k')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    plot_idx -= 1

    for clf in classifiers_list:
        if clf == 'LR':
            W = LR.train_LR(dataset, learning_rate, iteration_times, 0)
            print(W)
            avg_in, avg_out = LR.test_lr(W, dataset)
            Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W)
            Z = Z.reshape(xx.shape)

            ax.contourf(xx, yy, Z, cmap = cm, alpha = 0.8)
        elif clf == 'KLR':
            lamba = 0.001
            K_in = KLR.get_Kernel(training_x, training_x, 'linear')
            K_out = KLR.get_Kernel(training_x, testing_x, 'linear')
            K_mesh = KLR.get_Kernel(training_x, np.c_[xx.ravel(), yy.ravel()], 'linear')

            beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 0, lamba)
            avg_in, avg_out = KLR.test_klr(beta, K_in, K_out, dataset)

            Z = np.dot(K_mesh.T, beta)
            Z = Z.reshape(xx.shape)

            ax.contourf(xx, yy, Z, cmap = cm, alpha = 0.8)
        

        ax = plt.subplot(len(dataset_list), len(classifiers_list) + 1, plot_idx)

        ax.scatter(training_x[:,0], training_x[:,1], c = training_y, cmap = cm_bright, edgecolors = 'k')
        ax.scatter(testing_x[:,0], testing_x[:,1], c = testing_y, alpha = 0.6, cmap = cm_bright, edgecolors = 'k')

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        if ds_cnt == 0:
            ax.set_title(clf)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % avg_out).lstrip('0'),
                size=15, horizontalalignment='right')
        plot_idx -= 1

plt.tight_layout()
plt.savefig('./plot_dataset.png')








'''
# Kernel Logistic Regression
print('---------- Kernel Logistic Regression ----------')
lamba = 0.0001

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
'''

'''
# Kernel Ridge Regression
print('---------- Kernel Ridge Regression ----------')
beta = KLR.train_KRR(K_in, training_y, lamba)
KLR.test_klr(beta, K_in, K_out, dataset)
'''

