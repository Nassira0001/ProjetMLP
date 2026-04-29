import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SEMAINE2_ARCHITECTURE_RESEAU'))
from architecture_réseau.mlp import MLP

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SEMAINE3_BACKPROPAGATION', 'backpropagation'))
from gradients_sortie import output_layer_backward
from gradients_cachees import hidden_layer_backward

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SEMAINE4_ENTRAINEMENT_ET_VISUALISATION'))
from evaluation import load_normalization_params, generate_mesh, normalize_mesh, predict_mesh, denormalize, ground_truth, compute_mse
from visualisation import plot_loss, plot_side_by_side

def sgd_momentum_update(weights, biases, grad_weights, grad_biases, learning_rate, momentum, velocities_w, velocities_b):
    new_velocities_w = []
    new_velocities_b = []
    new_weights = []
    new_biases = []
    for i in range(len(weights)):
        vw = momentum * velocities_w[i] - learning_rate * grad_weights[i]
        vb = momentum * velocities_b[i] - learning_rate * grad_biases[i]
        new_velocities_w.append(vw)
        new_velocities_b.append(vb)
        new_weights.append(weights[i] + vw)
        new_biases.append(biases[i] + vb)
    return new_weights, new_biases, new_velocities_w, new_velocities_b

def train_momentum(model, X, y, epochs=500, learning_rate=0.01, batch_size=64, momentum=0.9, verbose=True):
    n_samples = X.shape[0]
    losses = []
    velocities_w = [np.zeros_like(w) for w in model.weights]
    velocities_b = [np.zeros_like(b) for b in model.biases]
    
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
            new_weights, new_biases, velocities_w, velocities_b = sgd_momentum_update(
                model.weights, model.biases, grad_weights, grad_biases,
                learning_rate, momentum, velocities_w, velocities_b
            )
            model.weights = new_weights
            model.biases = new_biases
        avg_loss = epoch_loss / n_samples
        losses.append(avg_loss)
        if verbose and (epoch % 50 == 0 or epoch == epochs-1):
            print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.6f}")
    return losses

data = np.load(os.path.join(os.path.dirname(__file__), '..', 'processed_data.npz'))
X_norm = np.column_stack((data['x_norm'], data['y_norm']))
y_norm = data['z_norm'].reshape(-1, 1)

norm_params = load_normalization_params(os.path.join(os.path.dirname(__file__), '..', 'norm_params.json'))
mean_x, std_x = norm_params['x_mean'], norm_params['x_std']
mean_y, std_y = norm_params['y_mean'], norm_params['y_std']
mean_z, std_z = norm_params['z_mean'], norm_params['z_std']

model = MLP([2, 64, 64, 1], activation='relu', output_activation='linear', init_method='he')

print("Entraînement avec momentum (0.9)...")
losses = train_momentum(model, X_norm, y_norm, epochs=500, learning_rate=0.01, batch_size=64, momentum=0.9)

plot_loss(losses, title="Loss avec momentum", save_path="loss_momentum.png")

X_mesh, Y_mesh = generate_mesh(n_points=100)
X_norm_mesh, Y_norm_mesh = normalize_mesh(X_mesh, Y_mesh, mean_x, std_x, mean_y, std_y)
Z_pred_norm = predict_mesh(model, X_norm_mesh, Y_norm_mesh)
Z_pred = denormalize(Z_pred_norm, mean_z, std_z)
Z_true = ground_truth(X_mesh, Y_mesh)

mse = compute_mse(Z_true, Z_pred)
print(f"MSE avec momentum : {mse:.6f}")

x_vals = np.linspace(-5, 5, X_mesh.shape[1])
y_vals = np.linspace(-5, 5, X_mesh.shape[0])
plot_side_by_side(Z_true, Z_pred, x_vals, y_vals,
                  title_true="Ground Truth",
                  title_pred="Prédictions avec momentum",
                  save_path="comparison_momentum.png")