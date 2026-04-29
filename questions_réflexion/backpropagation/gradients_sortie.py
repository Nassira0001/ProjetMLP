import numpy as np

def output_layer_backward(dA, A_prev, W, b, Z, activation='linear'):
    m = dA.shape[0]
    if activation == 'linear':
        dZ = dA * 1.0
    else:
        raise ValueError("Unsupported activation for output layer")
    dW = (A_prev.T @ dZ) / m
    db = np.sum(dZ, axis=0, keepdims=True) / m
    dA_prev = dZ @ W.T
    return dW, db, dA_prev