import matplotlib.pyplot as plt
import numpy as np

def plot_ground_truth(x, y, z, title="Surface réelle", save_path=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=1, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.colorbar(sc, label='Altitude (z)')
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_heatmap(x, y, z, title="Carte thermique", save_path=None):
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    zi = np.zeros((len(yi), len(xi)))
    for i, xi_val in enumerate(xi):
        for j, yi_val in enumerate(yi):
            mask = (x >= xi_val-0.1) & (x < xi_val+0.1) & (y >= yi_val-0.1) & (y < yi_val+0.1)
            if np.any(mask):
                zi[j, i] = np.mean(z[mask])
    plt.figure(figsize=(10, 7))
    plt.imshow(zi, extent=[min(x), max(x), min(y), max(y)], origin='lower', cmap='hot', aspect='auto')
    plt.colorbar(label='Altitude (z)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()