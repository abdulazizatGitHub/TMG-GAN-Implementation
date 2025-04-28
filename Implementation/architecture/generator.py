import torch.nn as nn

# In architecture/generator.py
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # First layer without batchnorm
        self.fc1 = nn.Linear(latent_dim, 512)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        
        # Rest of the layers with conditional batchnorm
        self.model = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, output_dim),
            nn.Tanh()  # Or Sigmoid if you prefer for normalization
        )
        
        # Create separate batchnorm for when we have sufficient samples
        self.bn = nn.BatchNorm1d(512)
    
    def forward(self, z):
        # First layer
        x = self.fc1(z)
        batch_size = z.size(0)
        
        # Apply batchnorm only if batch size > 1
        if batch_size > 1:
            x = self.bn(x)
        
        x = self.lrelu1(x)
        
        # Rest of the model
        return self.model(x)