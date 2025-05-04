import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 512)),
            nn.BatchNorm1d(512, track_running_stats=False),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(512, 128)),
            nn.BatchNorm1d(128, track_running_stats=False),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 32)),
            nn.BatchNorm1d(32, track_running_stats=False),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(32, output_dim)),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        # Handle edge case of single sample
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.model(z)