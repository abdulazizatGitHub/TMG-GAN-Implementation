import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_data_fitting(real_data, fake_data, epoch, ax):
    # Independent normalization
    real_np = (real_data.cpu().numpy() - real_data.mean().item()) / (real_data.std().item() + 1e-8)
    fake_np = (fake_data.cpu().numpy() - fake_data.mean().item()) / (fake_data.std().item() + 1e-8)
    
    # Combined PCA
    pca = PCA(n_components=2)
    combined = np.vstack([real_np, fake_np])
    pca.fit(combined)
    
    # Transform separately
    real_2d = pca.transform(real_np)
    fake_2d = pca.transform(fake_np)
    
    # Plotting
    ax.clear()
    ax.scatter(real_2d[:,0], real_2d[:,1], c='blue', alpha=0.5, s=10, label='Real')
    ax.scatter(fake_2d[:,0], fake_2d[:,1], c='orange', alpha=0.5, s=10, label='Fake')
    ax.set_title(f"Epoch {epoch}", fontsize=8)
    ax.legend(fontsize=6, markerscale=0.7)
    ax.axis('equal')
    ax.grid(False)