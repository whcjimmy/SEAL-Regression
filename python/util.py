import pdb
import json
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
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


def read_pulsar_stars():
    data = {}
    with open('./datasets/pulsar_stars.csv') as f:
        ff = [x.rstrip() for x in f.readlines()]
        ff = ff[1:]
        ff = [np.fromstring(x, dtype='float', sep=',') for x in ff]
        ff = np.array(ff)
        # np.random.shuffle(ff)
    samples = ff[:, :-1]
    labels = np.array([1 if x == 1 else -1 for x in ff[:,-1]])
    data['training_x'] = samples[:14318, :]
    data['training_y'] = labels[:14318]
    data['testing_x'] = samples[14318:, :]
    data['testing_y'] = labels[14318:]

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
    # samples, labels = make_circles(n_samples = 1000, noise = 0.1, factor = 0.5)
    samples, labels = make_circles(n_samples = 400, noise = 0.1, factor = 0.5, random_state = 1)
    labels = switch_labels(labels)
    training_x, testing_x, training_y, testing_y = train_test_split(samples, labels, test_size = 0.2, random_state = 42)
    dataset = package_dataset(training_x, training_y, testing_x, testing_y)

    return dataset


def read_make_moons():
    samples, labels = make_moons(n_samples = 400, noise = 0.3, random_state = 1)
    labels = switch_labels(labels)
    training_x, testing_x, training_y, testing_y = train_test_split(samples, labels, test_size = 0.2, random_state = 42)
    dataset = package_dataset(training_x, training_y, testing_x, testing_y)

    return dataset


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

def data_preprocess(samples, preprocesss_type):
    if preprocesss_type == 1:
        # Feature Scaling (Standardization)
        samples = stats.zscore(samples)
    elif preprocesss_type == 2:
        # Feature Scaling (min-max normalization)
        scaler = MinMaxScaler()
        scaler.fit(samples)
        samples = scaler.transform(samples)
    elif preprocesss_type == 3:
        # Normalize
        samples_col_sums = np.sum(samples, axis = 0)
        samples_norm = samples / samples_col_sums[np.newaxis, :]
        samples = samples_norm

    return samples
    '''
    # Normalize
    training_x_col_sums = np.sum(training_x, axis = 0)
    training_x_norm = training_x / training_x_col_sums[np.newaxis, :]
    '''

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
