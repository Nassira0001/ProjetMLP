import numpy as np

def relu(Z):
    return np.maximum(0, Z)

def linear(Z):
    return Z