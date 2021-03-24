import pdb
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_boston, load_diabetes, load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_dataset(load_func, dataset_name):
    dataset = load_func

    samples = dataset.data
    samples = StandardScaler().fit_transform(samples)

    labels = dataset.target
    labels = [1 if y == 1 else -1 for y in labels]

    labels = np.array([1 if x == 1 else -1 for x in labels]).reshape((-1, 1))
    samples = np.concatenate((samples, labels), axis = 1)

    pd.DataFrame(samples).to_csv('%s.csv' % dataset_name, header = False, index = False)


datasets_func = [load_breast_cancer(), load_boston(), load_diabetes(), load_iris()]
datasets_name = ['breast_cancer', 'boston_housing', 'diabetes', 'iris']


for load_func, dataset_name in zip(datasets_func, datasets_name):
    load_dataset(load_func, dataset_name)
