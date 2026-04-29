import numpy as np
from gradients_sortie import output_layer_backward
from gradients_cachees import hidden_layer_backward
from optimizers import sgd_update

def train(model, X, y, epochs=500, learning_rate=0.01, batch_size=64, verbose=True):
    n_samples = X.shape[0]
    losses = []
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        epoch_loss = 0
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            m_batch = X_batch.shape[0]
            y_pred = model.forward(X_batch)
            loss = model.compute_loss(y_pred, y_batch)
            epoch_loss += loss * (end - start)
            dA = (2 / m_batch) * (y_pred - y_batch)
            cache = model.cache
            grad_weights = [None] * model.num_layers
            grad_biases = [None] * model.num_layers
            Z_out, A_out = cache[-1]
            if model.num_layers > 1:
                A_prev_out = cache[-2][1]
            else:
                A_prev_out = X_batch
            dW, db, dA_prev = output_layer_backward(
                dA, A_prev_out, model.weights[-1], model.biases[-1], Z_out,
                activation=model.output_activation_name
            )
            grad_weights[-1] = dW
            grad_biases[-1] = db
            for i in range(model.num_layers - 2, -1, -1):
                Z_i, A_i = cache[i]
                if i == 0:
                    A_prev_i = X_batch
                else:
                    A_prev_i = cache[i-1][1]
                dW_i, db_i, dA_prev = hidden_layer_backward(
                    dA_prev, A_prev_i, model.weights[i], model.biases[i], Z_i,
                    activation=model.activation_name
                )
                grad_weights[i] = dW_i
                grad_biases[i] = db_i
            new_weights, new_biases = sgd_update(
                model.weights, model.biases, grad_weights, grad_biases, learning_rate
            )
            model.weights = new_weights
            model.biases = new_biases
        avg_loss = epoch_loss / n_samples
        losses.append(avg_loss)
        if verbose and (epoch % 50 == 0 or epoch == epochs-1):
            print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.6f}")
    return losses