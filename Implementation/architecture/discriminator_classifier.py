import torch.nn as nn
from torch.nn.utils import spectral_norm

class SharedBackbone(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared_layers = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, 1024)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(1024, 512)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(512, 128)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 32)),
            nn.LeakyReLU(0.2),
        )
        self.feature_layer = spectral_norm(nn.Linear(32, 16))
        self.feature_activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.shared_layers(x)
        features = self.feature_activation(self.feature_layer(x))
        return features

class DiscriminatorHead(nn.Module):
    def __init__(self, shared_backbone):
        super().__init__()
        self.shared = shared_backbone
        self.head = nn.Sequential(
            spectral_norm(nn.Linear(16, 1)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.shared(x)
        return self.head(features)

class ClassifierHead(nn.Module):
    def __init__(self, shared_backbone, num_classes):
        super().__init__()
        self.shared = shared_backbone
        self.head = nn.Sequential(
            spectral_norm(nn.Linear(16, num_classes)),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        features = self.shared(x)
        return self.head(features)