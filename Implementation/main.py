import torch
from torch.utils.data import DataLoader

from architecture.tmg_gan import TMGGAN
from utils.dataset import LoadDataset
from train_tmg_gan import train_tmg_gan

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset
    csv_path = "normalized_unsw_nb15_with_labels_encoded_oversampled.csv"
    dataset = LoadDataset(csv_file=csv_path)
    print(f"Loaded dataset: {dataset.num_samples} samples, "
          f"{dataset.num_features} features, {dataset.num_classes} classes.")

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Initialize model
    latent_dim = 100
    model = TMGGAN(
        latent_dim=latent_dim,
        num_classes=dataset.num_classes,
        input_dim=dataset.num_features
    ).to(device)

    # Training configuration
    epochs = 50
    learning_rate = 0.0002

    # Start training
    train_tmg_gan(
        model=model,
        dataloader=dataloader,
        latent_dim=latent_dim,
        epochs=epochs,
        lr=learning_rate,
        device=device
    )
