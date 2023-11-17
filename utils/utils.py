import numpy as np


def shuffle_data(x, y):
    n, d = x.shape  # Get the rows and columns of data
    n_train = int(0.8 * n)  # Cut 80% of data
    shuffler = np.random.permutation(n)

    x_train = x[shuffler[:n_train]]
    y_train = y[shuffler[:n_train]]
    x_test = x[shuffler[n_train:]]
    y_test = y[shuffler[n_train:]]
    return x_train, y_train, x_test, y_test
