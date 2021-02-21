import pdb
import json
import numpy as np
from sklearn.datasets import load_breast_cancer

def read_lbw():
    data = {}
    with open('./datasets/logistic/lowbwt.dat') as f:
        ff = [x.rstrip() for x in f.readlines()]
        ff = [np.fromstring(x, dtype='float', sep=',') for x in ff]
        ff = np.array(ff)
        np.random.shuffle(ff)
    samples = np.delete(ff, [0, 1], axis = 1)
    labels = np.array([1 if x == 1 else -1 for x in ff[:, 1]])
    data['training_x'] = samples
    data['training_y'] = labels
    data['testing_x'] = samples
    data['testing_y'] = labels

    return data

def read_pulsar_stars():
    data = {}
    with open('./datasets/pulsar_stars.csv') as f:
        ff = [x.rstrip() for x in f.readlines()]
        ff = ff[1:]
        ff = [np.fromstring(x, dtype='float', sep=',') for x in ff]
        ff = np.array(ff)
        # np.random.shuffle(ff)
    samples = ff[:, :-1]
    labels = ff[:, -1]
    data['training_x'] = samples
    data['training_y'] = labels
    data['testing_x'] = samples
    data['testing_y'] = labels
    '''
    data['training_x'] = samples[:1600, :]
    data['training_y'] = labels[:1600]
    data['testing_x'] = samples[1600:, :]
    data['testing_y'] = labels[1600:]
    '''

    return data


def read_banknote():
    data = {}
    with open('./datasets/banknote/data_banknote_authentication.txt') as f:
        ff = [x.rstrip() for x in f.readlines()]
        ff = [np.fromstring(x, dtype='float', sep=',') for x in ff]
        ff = np.array(ff)
        np.random.shuffle(ff)
    samples = ff[:, :-1]
    labels = np.array([1 if x == 1 else -1 for x in ff[:,-1]])
    data['training_x'] = samples[:1100, :]
    data['training_y'] = labels[:1100]
    data['testing_x'] = samples[1100:, :]
    data['testing_y'] = labels[1100:]

    return data

def read_load_breast_cancer():
    data = {}
    dataset = load_breast_cancer()
    samples = dataset.data
    labels = dataset.target
    
    data['training_x'] = samples[:455, :]
    data['training_y'] = labels[:455]
    data['testing_x'] = samples[455:, :]
    data['testing_y'] = labels[455:]

    return data


def read_data_hw3(train_filename, test_filename):
    data = {}
    f = open(train_filename, 'r')
    ff = [x.lstrip().rstrip() for x in f.readlines()]
    ff = [np.fromstring(x, dtype='float', sep=' ') for x in ff]
    ff = np.array(ff)

    data['training_x'] = ff[:, :-1]
    data['training_y'] = ff[:, -1]

    f = open(test_filename, 'r')
    ff = [x.lstrip().rstrip() for x in f.readlines()]
    ff = [np.fromstring(x, dtype='float', sep=' ') for x in ff]
    ff = np.array(ff)

    data['testing_x'] = ff[:, :-1]
    data['testing_y'] = ff[:, -1]

    return data


def read_data_hw2(filename):
    data = {}
    f = open(filename, 'r')
    ff = [x.lstrip().rstrip() for x in f.readlines()]
    ff = [np.fromstring(x, dtype='float', sep=' ') for x in ff]
    ff = np.array(ff)

    samples = ff[:, :-1]
    labels = ff[:,-1]
    
    data['training_x'] = samples[:400,]
    data['training_y'] = labels[:400,]
    data['testing_x']  = samples[400:,]
    data['testing_y']  = labels[400:,]
    
    return data


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
        t = 0
        for i in range(len(weights)):
            t = t + weights[i] * (value ** i)
        return t
