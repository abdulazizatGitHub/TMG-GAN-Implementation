import torch.nn as nn
from torch.nn.utils import spectral_norm

class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(Classifier, self).__init__()

        # Shared backbone for feature extraction
        self.shared = nn.Sequential(
            spectral_norm(nn.Linear(feature_dim, 1024)),
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

        # Classifier head (final layer for classification)
        # Discriminator head
        self.classifier_head = nn.Sequential(
            spectral_norm(nn.Linear(16, num_classes)),
            nn.Softmax()
        )

    def forward(self, x):
        # Extract features using the shared backbone
        features = self.shared(x)
        # Directly use the classifier head for classification
        pred_class = self.classifier_head(features)

        return pred_class
