import torch.nn as nn
from torch.nn.utils import spectral_norm

class DiscriminatorClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DiscriminatorClassifier, self).__init__()

        self.shared = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, 1024)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(1024, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(512, 128)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(128, 32)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(32, 16)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.discriminator_head = nn.Sequential(
            spectral_norm(nn.Linear(16, 1)),
            nn.Sigmoid()
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x, mode='both'):
        features = self.shared(x)
        if mode == 'discriminator':
            return self.discriminator_head(features)
        elif mode == 'classifier':
            return self.classifier_head(features)
        elif mode == 'features':
            return features
        else:
            return self.discriminator_head(features), self.classifier_head(features), features
