import numpy as np
from data.generate import generate_data
from data.normalize import normalize, save_normalization_params
from data.visualize import plot_ground_truth
def main():
    x, y, z = generate_data(n_points=2000)
    x_norm, x_mean, x_std = normalize(x)
    y_norm, y_mean, y_std = normalize(y)
    z_norm, z_mean, z_std = normalize(z)
    norm_params = {
        'x_mean': x_mean, 'x_std': x_std,
        'y_mean': y_mean, 'y_std': y_std,
        'z_mean': z_mean, 'z_std': z_std
    }
    save_normalization_params(norm_params, "norm_params.json")
    plot_ground_truth(x, y, z, title="Terrain réel (Scatter 3D)", save_path="ground_truth_3d.png")
    np.savez('processed_data.npz', x_norm=x_norm, y_norm=y_norm, z_norm=z_norm)
if __name__ == "__main__":
    main()