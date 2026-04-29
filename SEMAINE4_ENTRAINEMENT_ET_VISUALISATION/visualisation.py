import matplotlib.pyplot as plt

def plot_loss(losses, title="Training Loss", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_side_by_side(z_true, z_pred, x_vals, y_vals, title_true="Ground Truth", title_pred="Predictions", save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    im1 = ax1.imshow(z_true, extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]], origin='lower', cmap='viridis')
    ax1.set_title(title_true)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1, label='Altitude')
    
    im2 = ax2.imshow(z_pred, extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]], origin='lower', cmap='viridis')
    ax2.set_title(title_pred)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2, label='Altitude')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()