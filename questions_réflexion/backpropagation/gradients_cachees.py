import numpy as np

def relu_derivative(Z):
    return (Z > 0).astype(float)

def linear_derivative(Z):
    return np.ones_like(Z)

def hidden_layer_backward(dA_prev, A_prev, W, b, Z, activation='relu'):
    m = dA_prev.shape[0]
    if activation == 'relu':
        dZ = dA_prev * relu_derivative(Z)
    elif activation == 'linear':
        dZ = dA_prev * linear_derivative(Z)
    else:
        raise ValueError("Unsupported activation for hidden layer")
    dW = (A_prev.T @ dZ) / m
    db = np.sum(dZ, axis=0, keepdims=True) / m
    dA_prev_prev = dZ @ W.T
    return dW, db, dA_prev_prev