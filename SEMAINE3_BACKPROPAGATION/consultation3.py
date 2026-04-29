import sys
import os
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(__file__))  
sys.path.insert(0, project_root)

from SEMAINE2_ARCHITECTURE_RESEAU.architecture_réseau.mlp import MLP
from SEMAINE3_BACKPROPAGATION.backpropagation.train import train

data = np.load(os.path.join(os.path.dirname(__file__), '..', 'processed_data.npz'))
X_norm = np.column_stack((data['x_norm'], data['y_norm']))
y_norm = data['z_norm'].reshape(-1, 1)

model = MLP([2, 64, 64, 1], activation='relu', output_activation='linear', init_method='he')

losses = train(model, X_norm, y_norm, epochs=500, learning_rate=0.01, batch_size=64)

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss')
plt.grid(True)
plt.show()