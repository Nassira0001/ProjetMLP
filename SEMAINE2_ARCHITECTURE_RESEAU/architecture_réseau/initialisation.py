import numpy as np

def he_initializer(n_in, n_out):
    std = np.sqrt(2.0 / n_in)
    return np.random.randn(n_in, n_out) * std

def xavier_initializer(n_in, n_out):
    std = np.sqrt(2.0 / (n_in + n_out))
    return np.random.randn(n_in, n_out) * std

def random_small(n_in, n_out):
    return np.random.randn(n_in, n_out) * 0.01