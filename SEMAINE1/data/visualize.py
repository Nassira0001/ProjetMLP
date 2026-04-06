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
