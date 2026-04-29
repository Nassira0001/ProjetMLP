import numpy as np
import json
def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized = (data - mean) / std
    return normalized, mean, std
def save_normalization_params(params, filename="norm_params.json"):
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)
def load_normalization_params(filename="norm_params.json"):
    with open(filename, 'r') as f:
        return json.load(f)