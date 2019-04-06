# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 17:25:35 2018

@author: DaviesI
"""

from sklearn.neural_network import MLPClassifier
import numpy as np

values = np.random.randint(0,7+1,10000)

def one_hot_encoding(x):
    return_list = np.zeros(8).astype(int)
    return_list[x] = 1
    return list(return_list)

values = [one_hot_encoding(x) for x in values]

classifier = MLPClassifier(
            hidden_layer_sizes=[3],
            activation='logistic',
            solver='lbfgs',
            alpha = 1,
            verbose=True)

classifier.fit(values, values)

weights = classifier.coefs_[0]
intercepts = classifier.intercepts_[0]

zero_to_seven = [one_hot_encoding(x) for x in range(8)]

logit = lambda x: 1/(1+np.exp(-x))

def calculate_encoding(x, intercepts, weights):
    return logit(intercepts + np.matmul(x, weights))

encodings = np.array([calculate_encoding(x, intercepts, weights) for x in zero_to_seven])
rounded_encodings = np.round(encodings)
