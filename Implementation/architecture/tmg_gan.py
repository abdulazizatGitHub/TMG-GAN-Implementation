import torch.nn as nn
from .generator import Generator
from .discriminator_classifier import DiscriminatorClassifier

class TMGGAN(nn.Module):
    def __init__(self, latent_dim, input_dim, num_classes):
        super(TMGGAN, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # Define one generator per class
        self.generators = nn.ModuleList([
            Generator(latent_dim, input_dim) for _ in range(num_classes)
        ])

        # Unified discriminator + classifier
        self.dc = DiscriminatorClassifier(input_dim, num_classes)

    def generate_samples(self, z, class_idx):
        """
        Generates samples using the generator corresponding to class_idx.
        :param z: Tensor of shape (batch_size, latent_dim)
        :param class_idx: Integer class index (0 to num_classes-1)
        :return: Generated data samples for class class_idx
        """
        return self.generators[class_idx](z)

    def generate_all_classes(self, z_batch):
        """
        Generate a list of fake samples, one per class, using the same noise input.
        :param z_batch: List of z tensors (or a tensor of shape [num_classes, batch_size, latent_dim])
        :return: List of tensors of generated samples for each class
        """
        return [self.generators[k](z_batch[k]) for k in range(self.num_classes)]
