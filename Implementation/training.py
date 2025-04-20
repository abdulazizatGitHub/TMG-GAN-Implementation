import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving PNG files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torch.nn.utils import spectral_norm
from sklearn.decomposition import PCA

#############################################
# Utility: Min-Max Scale
#############################################
def minmax_scale(values):
    vmin, vmax = min(values), max(values)
    if abs(vmax - vmin) < 1e-12:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]

#############################################
# 1) CSV Dataset for Intrusion Detection
#############################################
class CSVIntrusionDataset(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        df = pd.read_csv(csv_file).values.astype(np.float32)
        self.features = torch.tensor(df[:, :-1], dtype=torch.float32)
        self.labels = torch.tensor(df[:, -1], dtype=torch.long)

        self.num_samples, self.num_features = self.features.shape
        unique_labels = np.unique(self.labels.numpy())
        self.num_classes = len(unique_labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

#############################################
# 2) Generator Architecture
#############################################
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)

#############################################
# 3) Shared Discriminator + Classifier
#############################################
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
            return (self.discriminator_head(features),
                    self.classifier_head(features),
                    features)

#############################################
# 4) Cosine Similarity Loss
#############################################
def cosine_similarity_loss(real_features, fake_features):
    cos_sim = F.cosine_similarity(real_features, fake_features, dim=1)
    return 1.0 - cos_sim

#############################################
# 5) TMG-GAN (Multi-Generators)
#############################################
class TMGGAN(nn.Module):
    def __init__(self, latent_dim, num_classes, input_dim):
        super(TMGGAN, self).__init__()
        self.num_classes = num_classes

        # One Generator per class
        self.generators = nn.ModuleList([
            Generator(latent_dim, input_dim) for _ in range(num_classes)
        ])
        self.dc = DiscriminatorClassifier(input_dim, num_classes)

    def generate_samples(self, z, class_idx):
        return self.generators[class_idx](z)

#############################################
# 6) Plot Real vs Fake in 2D (PCA)
#############################################
def plot_data_fitting(real_data, fake_data, epoch, ax):
    # Detach if necessary
    real_np = real_data.detach().cpu().numpy()
    fake_np = fake_data.detach().cpu().numpy()

    combined = np.vstack([real_np, fake_np])
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)

    n_real = real_np.shape[0]
    real_2d = combined_2d[:n_real]
    fake_2d = combined_2d[n_real:]

    ax.scatter(real_2d[:, 0], real_2d[:, 1], c='blue', alpha=0.6, label='Real')
    ax.scatter(fake_2d[:, 0], fake_2d[:, 1], c='orange', alpha=0.6, label='Fake')
    ax.set_title(f"Epoch {epoch}")
    ax.legend()

#############################################
# 7) Training Function for 500 epochs
#############################################
def train_tmg_gan(
    model,
    dataloader,
    latent_dim=100,
    epochs=500,   # Reverted to 500 epochs
    lr=0.0002,
    beta1=0.5,
    beta2=0.999,
    device="cuda"
):
    dc = model.dc
    generators = model.generators
    dc.to(device)
    for g in generators:
        g.to(device)

    disc_params = list(dc.shared.parameters()) + list(dc.discriminator_head.parameters())
    gen_params = list(dc.classifier_head.parameters())
    for g in generators:
        gen_params += list(g.parameters())

    disc_optim = optim.Adam(disc_params, lr=lr, betas=(beta1, beta2))
    gen_optim = optim.Adam(gen_params, lr=lr, betas=(beta1, beta2))

    bce = nn.BCELoss()
    eps = 1e-8

    d_losses, g_losses = [], []
    classifier_losses, classifier_accuracies = [], []
    cos_sims = []

    # We'll do data-fitting subplots at these epochs for the long run
    checkpoints = [100, 200, 300, 400, 500]
    fig, axes = plt.subplots(1, len(checkpoints), figsize=(5 * len(checkpoints), 4))
    axes = np.array(axes).ravel()

    for epoch in range(1, epochs + 1):
        epoch_d_loss, epoch_g_loss, epoch_class_loss = 0.0, 0.0, 0.0
        epoch_acc, epoch_cos_sim = 0.0, 0.0
        num_batches = 0

        for real_data, labels in dataloader:
            num_batches += 1
            real_data = real_data.to(device)
            labels = labels.to(device)
            batch_size = real_data.size(0)

            # Train Discriminator
            disc_optim.zero_grad()
            real_lbls = torch.ones(batch_size, 1, device=device)
            fake_lbls = torch.zeros(batch_size, 1, device=device)

            pred_real = dc(real_data, mode='discriminator')
            d_real_loss = bce(pred_real, real_lbls)

            class_idx = labels[0].item()  # simplistic approach: assume single-class batch
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_data = model.generate_samples(z, class_idx)

            pred_fake = dc(fake_data.detach(), mode='discriminator')
            d_fake_loss = bce(pred_fake, fake_lbls)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            disc_optim.step()
            epoch_d_loss += d_loss.item()

            # Train Generator + Classifier
            gen_optim.zero_grad()
            pred_fake_g = dc(fake_data, mode='discriminator')
            adv_loss = bce(pred_fake_g, real_lbls)

            # Classifier
            class_fake = dc(fake_data, mode='classifier')
            class_real = dc(real_data, mode='classifier')

            # Cosine similarity
            fake_feats = dc(fake_data, mode='features')
            real_feats = dc(real_data, mode='features')
            cos_vals = cosine_similarity_loss(real_feats, fake_feats)
            epoch_cos_sim += cos_vals.mean().item()

            # partial updates
            predicted_labels = torch.argmax(class_fake, dim=1)
            correct_mask = (predicted_labels == labels)
            if correct_mask.sum().item() > 0:
                cos_correct = cos_vals[correct_mask].mean()
                g_loss = adv_loss + cos_correct
                g_loss.backward()
                gen_optim.step()
                epoch_g_loss += g_loss.item()

            # Classifier loss & accuracy on real
            eps_probs = class_real[range(batch_size), labels] + eps
            c_loss = -torch.log(eps_probs).mean()
            epoch_class_loss += c_loss.item()

            acc = (torch.argmax(class_real, dim=1) == labels).float().mean()
            epoch_acc += acc.item()

        d_losses.append(epoch_d_loss / num_batches)
        g_losses.append(epoch_g_loss / num_batches)
        classifier_losses.append(epoch_class_loss / num_batches)
        classifier_accuracies.append(epoch_acc / num_batches)
        cos_sims.append(epoch_cos_sim / num_batches)

        print(f"[Epoch {epoch}/{epochs}] "
              f"D Loss: {d_losses[-1]:.4f}, "
              f"G Loss: {g_losses[-1]:.4f}, "
              f"Class Loss: {classifier_losses[-1]:.4f}, "
              f"Clf Acc: {classifier_accuracies[-1]:.4f}, "
              f"Cos Sim: {cos_sims[-1]:.4f}")

        # At checkpoint epochs, plot real vs. fake data in 2D
        if epoch in checkpoints:
            idx = checkpoints.index(epoch)
            ax = axes[idx]
            ax.clear()

            # We'll take a small sample of real_data from the last batch
            n_plot = min(200, batch_size)
            real_plot = real_data[:n_plot].detach()
            z_plot = torch.randn(n_plot, latent_dim, device=device)
            fake_plot = model.generate_samples(z_plot, class_idx)[:n_plot].detach()

            plot_data_fitting(real_plot, fake_plot, epoch, ax)

    # After training, save the multi-subplot figure
    fig.suptitle("Data Fitting Over Training (500 Epochs)")
    fig.tight_layout()
    fig.savefig("data_fitting_subplots.png", dpi=300)
    plt.close(fig)

    # Normalize metrics
    d_losses_norm = minmax_scale(d_losses)
    g_losses_norm = minmax_scale(g_losses)
    classifier_losses_norm = minmax_scale(classifier_losses)
    classifier_acc_norm = minmax_scale(classifier_accuracies)
    cos_sims_norm = minmax_scale(cos_sims)

    # Discriminator Loss
    plt.figure()
    plt.plot(d_losses_norm, label="D Loss (Norm)")
    plt.title("Discriminator Loss (Normalized)")
    plt.xlabel("Epoch")
    plt.ylabel("Scaled Loss")
    plt.legend()
    plt.savefig("discriminator_loss.png", dpi=300)
    plt.close()

    # Generator Loss
    plt.figure()
    plt.plot(g_losses_norm, label="G Loss (Norm)")
    plt.title("Generator Loss (Normalized)")
    plt.xlabel("Epoch")
    plt.ylabel("Scaled Loss")
    plt.legend()
    plt.savefig("generator_loss.png", dpi=300)
    plt.close()

    # Classifier Loss
    plt.figure()
    plt.plot(classifier_losses_norm, label="Classifier Loss (Norm)")
    plt.title("Classifier Loss (Normalized)")
    plt.xlabel("Epoch")
    plt.ylabel("Scaled Loss")
    plt.legend()
    plt.savefig("classifier_loss.png", dpi=300)
    plt.close()

    # Cosine Similarity
    plt.figure()
    plt.plot(cos_sims_norm, label="Cosine Similarity (Norm 1-cos)")
    plt.title("Cosine Similarity (Normalized)")
    plt.xlabel("Epoch")
    plt.ylabel("Scaled 1 - cos(sim)")
    plt.legend()
    plt.savefig("cosine_similarity.png", dpi=300)
    plt.close()

    # Classifier Accuracy
    plt.figure()
    plt.plot(classifier_acc_norm, label="Classifier Accuracy (Norm)")
    plt.title("Classifier Accuracy (Normalized)")
    plt.xlabel("Epoch")
    plt.ylabel("Scaled Accuracy")
    plt.legend()
    plt.savefig("classifier_accuracy.png", dpi=300)
    plt.close()

    return d_losses, g_losses, classifier_losses, classifier_accuracies, cos_sims

#############################################
# 8) Main
#############################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    csv_path = "normalized_unsw_nb15_with_labels_encoded_oversampled.csv"
    dataset = CSVIntrusionDataset(csv_file=csv_path)
    print(f"Loaded dataset: {dataset.num_samples} samples, "
          f"{dataset.num_features} features, {dataset.num_classes} classes.")

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    latent_dim = 100
    model = TMGGAN(latent_dim, dataset.num_classes, dataset.num_features).to(device)

    # 500 epochs run
    epochs = 500
    lr = 0.0002

    (d_losses,
     g_losses,
     class_losses,
     class_acc,
     cos_sims) = train_tmg_gan(
         model=model,
         dataloader=dataloader,
         latent_dim=latent_dim,
         epochs=epochs,
         lr=lr,
         device=device
     )
