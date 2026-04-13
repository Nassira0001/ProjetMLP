import sys
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(script_dir)

from architecture_réseau.mlp import MLP

data_path = os.path.join(os.path.dirname(script_dir), 'donnees_normalisees.npz')
data = np.load(data_path)

X_norm = np.column_stack((data['x_norm'], data['y_norm']))
y_norm = data['z_norm'].reshape(-1, 1)

mlp = MLP([2, 64, 64, 1], activation='relu', output_activation='linear', init_method='he')
y_pred = mlp.forward(X_norm)
loss = mlp.compute_loss(y_pred, y_norm)

print(f"Loss initiale : {loss:.6f}")