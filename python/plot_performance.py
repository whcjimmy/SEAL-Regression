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


#Original Load Weights
weights_dict = util.read_weight('./weights.out')
for k in weights_dict.keys():
    weights_dict[k].reverse()


# Data Settings 
iteration_times = 40
dataset_list = [util.read_make_circles(), util.read_make_moons()]

'''
# make_circles
classifiers_list = [{'model': 'LR', 'sigmoid_deg': 3, 'learning_rate': 0.1},
                    {'model': 'LR', 'sigmoid_deg': 7, 'learning_rate': 0.1},
                    {'model': 'KLR_linear', 'sigmoid_deg': 3, 'learning_rate': 0.1, 'lamba': 0.1, 'kernel': 'linear'},
                    {'model': 'KLR_linear', 'sigmoid_deg': 7, 'learning_rate': 0.1, 'lamba': 0.1, 'kernel': 'linear'},
                    {'model': 'KLR_polynomial', 'sigmoid_deg': 3, 'learning_rate': 0.01, 'lamba': 0.01, 'kernel': 'polynomial', 'poly_deg': 3, 'gamma': 1.0},
                    {'model': 'KLR_polynomial', 'sigmoid_deg': 7, 'learning_rate': 0.01, 'lamba': 0.001, 'kernel': 'polynomial', 'poly_deg': 3, 'gamma': 1.0},
                    {'model': 'KLR_rbf', 'sigmoid_deg': 3, 'learning_rate': 0.01, 'lamba': 0.1, 'kernel': 'rbf', 'gamma': 0.5}]
'''
# make_moons
classifiers_list = [{'model': 'LR', 'sigmoid_deg': 3, 'learning_rate': 0.1},
                    {'model': 'LR', 'sigmoid_deg': 7, 'learning_rate': 0.1},
                    {'model': 'KLR_linear', 'sigmoid_deg': 3, 'learning_rate': 0.1, 'lamba': 0.1, 'kernel': 'linear'},
                    {'model': 'KLR_linear', 'sigmoid_deg': 7, 'learning_rate': 0.1, 'lamba': 0.1, 'kernel': 'linear'},
                    {'model': 'KLR_polynomial', 'sigmoid_deg': 3, 'learning_rate': 0.001, 'lamba': 0.1, 'kernel': 'polynomial', 'poly_deg': 3, 'gamma': 0.1},
                    {'model': 'KLR_polynomial', 'sigmoid_deg': 7, 'learning_rate': 0.001, 'lamba': 0.01, 'kernel': 'polynomial', 'poly_deg': 3, 'gamma': 0.1},
                    {'model': 'KLR_rbf', 'sigmoid_deg': 3, 'learning_rate': 0.01, 'lamba': 0.1, 'kernel': 'rbf', 'gamma': 1}]

# Plot Settings 
h = 0.2
plot_idx = 1
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

figure = plt.figure(figsize=(30, 9))


for ds_cnt, dataset in enumerate(dataset_list):
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
    plot_idx += 1

    for clf_dict in classifiers_list:
        ax = plt.subplot(len(dataset_list), len(classifiers_list) + 1, plot_idx)
        avg_in = 0
        avg_out = 0

        model = clf_dict['model']
        if model == 'LR':
            sigmoid_deg = clf_dict['sigmoid_deg']
            learning_rate = clf_dict['learning_rate']

            W = LR.train_LR(dataset, learning_rate, iteration_times, 2, weights_dict[str(sigmoid_deg)])
            avg_in, avg_out = LR.test_lr(W, dataset)

            Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W)
            Z = Z.reshape(xx.shape)

            ax.contourf(xx, yy, Z, cmap = cm, alpha = 0.8)
        elif 'KLR' in model:
            sigmoid_deg = clf_dict['sigmoid_deg']
            learning_rate = clf_dict['learning_rate']
            kernel = clf_dict['kernel']
            lamba = clf_dict['lamba']
            if kernel == 'linear':
                K_in = KLR.get_Kernel(training_x, training_x, kernel)
                K_out = KLR.get_Kernel(training_x, testing_x, kernel)
                K_mesh = KLR.get_Kernel(training_x, np.c_[xx.ravel(), yy.ravel()], kernel)
            elif kernel == 'polynomial':
                poly_deg = clf_dict['poly_deg']
                gamma = clf_dict['gamma']
                K_in = KLR.get_Kernel(training_x, training_x, kernel, gamma, poly_deg)
                K_out = KLR.get_Kernel(training_x, testing_x, kernel, gamma, poly_deg)
                K_mesh = KLR.get_Kernel(training_x, np.c_[xx.ravel(), yy.ravel()], kernel, gamma, poly_deg)
            elif kernel == 'rbf':
                gamma = clf_dict['gamma']
                K_in = KLR.get_Kernel(training_x, training_x, kernel, gamma)
                K_out = KLR.get_Kernel(training_x, testing_x, kernel, gamma)
                K_mesh = KLR.get_Kernel(training_x, np.c_[xx.ravel(), yy.ravel()], kernel, gamma)

            beta = KLR.train_KLR(K_in, training_y, learning_rate, iteration_times, 2, lamba, weights_dict[str(sigmoid_deg)])
            avg_in, avg_out = KLR.test_klr(beta, K_in, K_out, dataset)

            Z = np.dot(K_mesh.T, beta)
            Z = Z.reshape(xx.shape)

            ax.contourf(xx, yy, Z, cmap = cm, alpha = 0.8)


        ax.scatter(training_x[:,0], training_x[:,1], c = training_y, cmap = cm_bright, edgecolors = 'k')
        ax.scatter(testing_x[:,0], testing_x[:,1], c = testing_y, alpha = 0.6, cmap = cm_bright, edgecolors = 'k')

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        if ds_cnt == 0:
            ax.set_title(clf_dict['model'] + '_' + str(clf_dict['sigmoid_deg']))
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f %.2f' % (avg_in, avg_out)),
                size=15, horizontalalignment='right')
        plot_idx += 1

plt.tight_layout()
plt.savefig('./plot_performance.png')
