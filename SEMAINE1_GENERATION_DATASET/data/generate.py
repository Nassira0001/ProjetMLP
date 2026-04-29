import numpy as np
def mystery_function(x, y):
    return np.sin(np.sqrt(x**2 + y**2)) + 0.5 * np.cos(2*x + 2*y)
def generate_data(n_points=2000, x_min=-5, x_max=5, y_min=-5, y_max=5, seed=42):
    np.random.seed(seed)
    x = np.random.uniform(x_min, x_max, n_points)
    y = np.random.uniform(y_min, y_max, n_points)
    z = mystery_function(x, y)
    return x, y, z