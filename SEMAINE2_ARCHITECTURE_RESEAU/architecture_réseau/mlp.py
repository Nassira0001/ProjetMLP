import numpy as np
from .activations import relu, linear
from .initialisation import he_initializer, xavier_initializer, random_small
from .loss import mse_loss

class MLP:
    def __init__(self, layer_sizes, activation='relu', output_activation='linear', init_method='he'):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.activation_name = activation
        self.output_activation_name = output_activation
        self.init_method = init_method
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i+1]
            W, b = self._initialize_layer(n_in, n_out)
            self.weights.append(W)
            self.biases.append(b)
    
    def _initialize_layer(self, n_in, n_out):
        if self.init_method == 'he':
            W = he_initializer(n_in, n_out)
        elif self.init_method == 'xavier':
            W = xavier_initializer(n_in, n_out)
        else:
            W = random_small(n_in, n_out)
        b = np.zeros((1, n_out))
        return W, b
    
    def _activation(self, Z, name):
        if name == 'relu':
            return relu(Z)
        elif name == 'linear':
            return linear(Z)
        else:
            raise ValueError("Activation not supported")
    
    def forward(self, X):
        self.cache = []
        A = X
        for i in range(self.num_layers - 1):
            Z = A @ self.weights[i] + self.biases[i]
            A = self._activation(Z, self.activation_name)
            self.cache.append((Z, A))
        Z_out = A @ self.weights[-1] + self.biases[-1]
        A_out = self._activation(Z_out, self.output_activation_name)
        self.cache.append((Z_out, A_out))
        return A_out
    
    def compute_loss(self, y_pred, y_true):
        return mse_loss(y_pred, y_true)