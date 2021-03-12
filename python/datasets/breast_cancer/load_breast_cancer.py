import pdb
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

dataset = load_breast_cancer()
samples = dataset.data
labels = dataset.target

# Feature Scaling (min-max normalization)
scaler = MinMaxScaler()
scaler.fit(samples)
samples = scaler.transform(samples)

labels = np.array([1 if x == 1 else -1 for x in labels]).reshape((-1, 1))
samples = np.concatenate((samples, labels), axis = 1)
# zeros = np.zeros(labels.shape[0]).reshape((-1, 1))
# samples = np.concatenate((samples, zeros, zeros, labels), axis = 1)

pd.DataFrame(samples).to_csv('data.csv', header = False, index = False)

