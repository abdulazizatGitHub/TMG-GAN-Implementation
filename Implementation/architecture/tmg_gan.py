import torch.nn as nn
from .generator import Generator
from .discriminator_classifier import DiscriminatorClassifier

class TMGGAN(nn.Module):
    def __init__(self, latent_dim, input_dim, num_classes):
        super(TMGGAN, self).__init__()
        self.generator = Generator(latent_dim, input_dim)  # Only one generator
        self.dc = DiscriminatorClassifier(input_dim, num_classes)

    def generate_samples(self, z):
        return self.generator(z)  # No need for class_idx
