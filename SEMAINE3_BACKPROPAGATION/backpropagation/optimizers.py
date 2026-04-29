def sgd_update(weights, biases, grad_weights, grad_biases, learning_rate):
    new_weights = [w - learning_rate * dw for w, dw in zip(weights, grad_weights)]
    new_biases = [b - learning_rate * db for b, db in zip(biases, grad_biases)]
    return new_weights, new_biases