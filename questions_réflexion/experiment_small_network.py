import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SEMAINE2_ARCHITECTURE_RESEAU'))
from architecture_réseau.mlp import MLP

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SEMAINE3_BACKPROPAGATION', 'backpropagation'))
from train import train

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SEMAINE4_ENTRAINEMENT_ET_VISUALISATION'))
from evaluation import load_normalization_params, generate_mesh, normalize_mesh, predict_mesh, denormalize, ground_truth, compute_mse
from visualisation import plot_loss, plot_side_by_side

data = np.load(os.path.join(os.path.dirname(__file__), '..', 'processed_data.npz'))
X_norm = np.column_stack((data['x_norm'], data['y_norm']))
y_norm = data['z_norm'].reshape(-1, 1)

norm_params = load_normalization_params(os.path.join(os.path.dirname(__file__), '..', 'norm_params.json'))
mean_x, std_x = norm_params['x_mean'], norm_params['x_std']
mean_y, std_y = norm_params['y_mean'], norm_params['y_std']
mean_z, std_z = norm_params['z_mean'], norm_params['z_std']

model = MLP([2, 4, 1], activation='relu', output_activation='linear', init_method='he')

print("Entraînement avec petite architecture [2,4,1]...")
losses = train(model, X_norm, y_norm, epochs=500, learning_rate=0.01, batch_size=64)

plot_loss(losses, title="Loss avec architecture [2,4,1]", save_path="loss_small.png")

X_mesh, Y_mesh = generate_mesh(n_points=100)
X_norm_mesh, Y_norm_mesh = normalize_mesh(X_mesh, Y_mesh, mean_x, std_x, mean_y, std_y)
Z_pred_norm = predict_mesh(model, X_norm_mesh, Y_norm_mesh)
Z_pred = denormalize(Z_pred_norm, mean_z, std_z)
Z_true = ground_truth(X_mesh, Y_mesh)

mse = compute_mse(Z_true, Z_pred)
print(f"MSE avec architecture [2,4,1] : {mse:.6f}")

x_vals = np.linspace(-5, 5, X_mesh.shape[1])
y_vals = np.linspace(-5, 5, X_mesh.shape[0])
plot_side_by_side(Z_true, Z_pred, x_vals, y_vals,
                  title_true="Ground Truth",
                  title_pred="Prédictions (petit réseau [2,4,1])",
                  save_path="comparison_small.png")