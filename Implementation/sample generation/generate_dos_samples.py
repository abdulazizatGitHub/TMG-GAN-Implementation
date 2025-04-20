import torch

def generate_dos_samples(generator, classifier, class_idx=1, num_samples=1000, latent_dim=100, batch_size=128, device="cuda"):
    """
    Generate pseudo samples for class 1 (DoS) using trained TMG-GAN components.
    This follows the filtering mechanism described in Algorithm 2 using the classifier.

    Args:
        generator (nn.Module): Trained generator for DoS class.
        classifier (nn.Module): Trained classifier with shared + head structure.
        class_idx (int): Class index to generate for (default=1 for DoS).
        num_samples (int): Total desired high-quality samples.
        latent_dim (int): Dimension of input noise.
        batch_size (int): Batch size for each generation attempt.
        device (str): "cuda" or "cpu".

    Returns:
        torch.Tensor: Generated and filtered pseudo samples for class 1.
    """
    generator.eval()
    classifier.eval()
    generator.to(device)
    classifier.to(device)

    accepted_samples = []

    while len(accepted_samples) < num_samples:
        # Step 2: Sample latent vectors
        z = torch.randn(batch_size, latent_dim, device=device)

        # Step 3: Generate pseudo samples
        with torch.no_grad():
            fake_data = generator(z)
            class_preds = classifier(fake_data, mode='classifier')
            predicted_labels = torch.argmax(class_preds, dim=1)

        # Step 7–9: Filter correct class predictions
        valid_mask = predicted_labels == class_idx
        print(f"[Batch] Generated: {batch_size}, Accepted: {valid_mask.sum().item()}")
        selected = fake_data[valid_mask]

        # Step 10–11: Add accepted samples
        accepted_samples.extend(selected.cpu().split(1, dim=0))  # add individually

    # Step 13: Return filtered and trimmed samples
    generated_tensor = torch.cat(accepted_samples, dim=0)[:num_samples]
    return generated_tensor
