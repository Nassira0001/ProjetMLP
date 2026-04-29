import numpy as np
import json

def load_normalization_params(filename="norm_params.json"):
    with open(filename, 'r') as f:
        return json.load(f)

def generate_mesh(x_min=-5, x_max=5, y_min=-5, y_max=5, n_points=100):
    x_vals = np.linspace(x_min, x_max, n_points)
    y_vals = np.linspace(y_min, y_max, n_points)
    X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
    return X_mesh, Y_mesh

def normalize_mesh(X_mesh, Y_mesh, mean_x, std_x, mean_y, std_y):
    X_norm = (X_mesh - mean_x) / std_x
    Y_norm = (Y_mesh - mean_y) / std_y
    return X_norm, Y_norm

def predict_mesh(model, X_norm, Y_norm):
    flat_X = X_norm.ravel().reshape(-1, 1)
    flat_Y = Y_norm.ravel().reshape(-1, 1)
    X_flat = np.column_stack((flat_X, flat_Y))
    Z_pred_norm = model.forward(X_flat)
    return Z_pred_norm.reshape(X_norm.shape)

def denormalize(Z_norm, mean_z, std_z):
    return Z_norm * std_z + mean_z

def ground_truth(X_mesh, Y_mesh):
    return np.sin(np.sqrt(X_mesh**2 + Y_mesh**2)) + 0.5 * np.cos(2*X_mesh + 2*Y_mesh)

def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)