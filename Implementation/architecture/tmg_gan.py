import torch.nn as nn
from .generator import Generator
from .discriminator_classifier import SharedBackbone, DiscriminatorHead, ClassifierHead

class TMGGAN(nn.Module):
    def __init__(self, latent_dim, input_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        # Shared backbone
        self.shared_backbone = SharedBackbone(input_dim)
        
        # Components
        self.generators = nn.ModuleList([
            Generator(latent_dim, input_dim) for _ in range(num_classes)
        ])
        self.discriminator = DiscriminatorHead(self.shared_backbone)
        self.classifier = ClassifierHead(self.shared_backbone, num_classes)

    def get_features(self, x):
        """Returns F(x) - the last hidden layer features from classifier"""
        return self.shared_backbone(x)  # This now returns the 128-dim features
    
    def discriminate(self, x):
        return self.discriminator(x)
    
    def classify(self, x):
        return self.classifier(x)

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
