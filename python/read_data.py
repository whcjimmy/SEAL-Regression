import pdb
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_circles, make_moons


def switch_labels(labels):
    return [1 if y == 1 else -1 for y in labels]


def package_dataset(training_x, training_y, testing_x, testing_y):
    data = {}
    data['training_x'] = training_x
    data['training_y'] = training_y
    data['testing_x'] = testing_x
    data['testing_y'] = testing_y

    return data


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


def read_heart_disease():
    data = {}
    with open('./datasets/Heart-Disease-Machine-Learning/data.csv') as f:
        ff = [x.rstrip() for x in f.readlines()]
        ff = [np.fromstring(x, dtype='float', sep=',') for x in ff]
        ff = np.array(ff)
        # np.random.shuffle(ff)
    samples = data_preprocess(ff[:, :-1], 2)
    # samples = ff[:, :-1]
    labels = np.array([1 if x == 1 else -1 for x in ff[:,-1]])

    data['training_x'] = samples[: 736]
    data['testing_x'] = samples[736:]
    data['training_y'] = labels[: 736]
    data['testing_y'] = labels[736:]

    return data


def read_load_breast_cancer():
    data = {}
    dataset = load_breast_cancer()
    samples = dataset.data
    labels = dataset.target
    samples = data_preprocess(samples, 2)
    labels = np.array([1 if x == 1 else -1 for x in labels])

    data['training_x'] = samples[:364, :]
    data['training_y'] = labels[:364]
    data['testing_x'] = samples[364:, :]
    data['testing_y'] = labels[364:]

    return data


def read_make_circles():
    samples, labels = make_circles(n_samples = 500, noise = 0.1, factor = 0.3, random_state = 1)
    # samples = data_preprocess(samples, 'StandardScaler')
    labels = switch_labels(labels)
    training_x, testing_x, training_y, testing_y = train_test_split(samples, labels, test_size = 0.2)
    dataset = package_dataset(training_x, training_y, testing_x, testing_y)

    samples = np.concatenate((samples, np.array(labels).reshape(-1, 1)), axis = 1)
    pd.DataFrame(samples).to_csv('./datasets/make_circles.csv', header = False, index = False)


    return dataset


def read_make_moons():
    samples, labels = make_moons(n_samples = 500, noise = 0.1)
    samples = data_preprocess(samples, 'MinMaxScaler')
    labels = switch_labels(labels)
    training_x, testing_x, training_y, testing_y = train_test_split(samples, labels, test_size = 0.2)
    dataset = package_dataset(training_x, training_y, testing_x, testing_y)

    return dataset


def data_preprocess(samples, preprocesss_type):
    if preprocesss_type == 'StandardScaler':
        # Feature Scaling (Standardization)
        samples = StandardScaler().fit_transform(samples)
    elif preprocesss_type == 'MinMaxScaler':
        # Feature Scaling (min-max normalization)
        samples = MinMaxScaler().fit_transform(samples)
    elif preprocesss_type == 'Normalize':
        # Normalize
        samples_col_sums = np.sum(samples, axis = 0)
        samples_norm = samples / samples_col_sums[np.newaxis, :]
        samples = samples_norm

    return samples
