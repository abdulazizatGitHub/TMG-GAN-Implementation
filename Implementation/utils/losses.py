import torch
import torch.nn as nn
import torch.nn.functional as F

class TMGCosineLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, real_features, fake_features, all_classes_features):
        """
        Args:
            real_features: Features of REAL samples from current class [batch_size, feat_dim]
            fake_features: Features of FAKE samples from current class [batch_size, feat_dim]
            all_classes_features: List of features for ALL classes [num_classes, batch_size, feat_dim]
        Returns:
            O_k: Cosine loss value (Eq.4)
        """
        # Intra-class similarity (Eq.2) - Maximize
        intra_sim = F.cosine_similarity(fake_features, real_features, dim=1).mean()

        # Inter-class dissimilarity (Eq.3) - Minimize
        inter_sim = 0
        for k in range(self.num_classes):
            if k != self.current_class:
                inter_sim += F.cosine_similarity(fake_features, all_classes_features[k], dim=1).mean()
        inter_sim /= (self.num_classes - 1)

        return inter_sim - intra_sim  # Eq.4