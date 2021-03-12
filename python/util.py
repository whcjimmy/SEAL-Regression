import pdb
import json
import numpy as np
import matplotlib.pyplot as plt


def read_weight(weight_filename):
    with open(weight_filename, 'r') as infile:
        weights_dict = json.load(infile)
    return weights_dict


def sigmoid(value): 
    return 1.0 / (1 + np.exp(-1 * value))


def approx_func(approx, model, data, label, weights = None):
    value = -1 * label * np.dot(model.T, data)

    if approx == 0:
        return sigmoid(value)
    elif approx == 1:
        poly = np.poly1d(weights)
        return poly(value)
    elif approx == 2:
        tmp = 0
        for i in range(len(weights)):
            tmp = tmp + weights[i] * (value ** i)

    return tmp


def accuracy(model, data, labels):
    score = np.dot(data, model)
    return np.mean(np.sign(score) == labels)


def test_model(model, dataset):
    training_x = dataset['training_x']
    training_y = dataset['training_y']
    testing_x = dataset['testing_x']
    testing_y = dataset['testing_y']
    avg_in = accuracy(model, training_x, training_y)
    avg_out = accuracy(model, testing_x, testing_y)
    print("avg_in = %.3f avg_out = %.3f " % (avg_in, avg_out))
    return avg_in, avg_out


def plot_train_acc_curve(model_list, dataset, params):
    # x-axis
    label_x = [i for i in range(len(model_list))]

    # y-axis
    training_x = dataset['training_x']
    training_y = dataset['training_y']
    pred_y = [accuracy(W, training_x, training_y) for W in model_list]
    # print(pred_y)

    # plt
    plt.plot(label_x, pred_y, label=params)

    plt.legend()
    plt.savefig('./accuracy.png')
