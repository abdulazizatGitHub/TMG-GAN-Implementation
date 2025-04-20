import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_data_fitting(real_data, fake_data, epoch, ax):
    real_np = real_data.detach().cpu().numpy()
    fake_np = fake_data.detach().cpu().numpy()
    combined = np.vstack([real_np, fake_np])
    
    # Optional normalization to avoid scale dominance
    combined = (combined - np.mean(combined, axis=0)) / (np.std(combined, axis=0) + 1e-8)

    # PCA to 2D
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)

    # Split PCA result
    n_real = real_np.shape[0]
    real_2d = combined_2d[:n_real]
    fake_2d = combined_2d[n_real:]

    # Plot cleanly with adjusted aesthetics
    ax.clear()
    ax.scatter(real_2d[:, 0], real_2d[:, 1], c='blue', alpha=0.7, s=25, label='Real')
    ax.scatter(fake_2d[:, 0], fake_2d[:, 1], c='orange', alpha=0.7, s=25, label='Fake')
    ax.set_title(f"Epoch {epoch}", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(False)

    # Optional: limit zoom to tighten layout around clusters
    ax.set_xlim(np.min(combined_2d[:, 0]) - 1, np.max(combined_2d[:, 0]) + 1)
    ax.set_ylim(np.min(combined_2d[:, 1]) - 1, np.max(combined_2d[:, 1]) + 1)
