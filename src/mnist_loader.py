import _pickle as cPickle
import gzip

import numpy as np


def load_data(path):
    with gzip.open(path, "rb") as f:
        training, validation, test = list(map(list, cPickle.load(f, encoding="latin1")))
        return (training, validation, test)


def load_data_wrapper(path):
    training, validation, test = load_data(path)

    training[1] = np.asarray(list(map(vectorize, training[1])))
    validation[1] = np.asarray(list(map(vectorize, validation[1])))
    test[1] = np.asarray(list(map(vectorize, test[1])))

    # training = [
    #     np.asarray(list(training[0]) + list(validation[0])),
    #     np.asarray(list(training[1]) + list(validation[1])),
    # ]
    return (training, validation, test)


def vectorize(j):
    v = np.zeros(10)
    v[j] = 1
    return v
