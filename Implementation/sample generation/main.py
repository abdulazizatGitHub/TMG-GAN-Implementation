import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch
import torch.nn.functional as F
import numpy as np

from generate_dos_samples import generate_dos_samples
from architecture.generator import Generator
from architecture.discriminator_classifier import DiscriminatorClassifier

if __name__ == "__main__":

    # Config
    latent_dim = 100
    input_dim = 42  # Replace accordingly
    num_classes = 5  # Replace accordingly
    class_idx = 1  # DoS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    generator = Generator(latent_dim, input_dim).to(device)
    generator.load_state_dict(torch.load("generator_class1_dos.pth", weights_only=True))

    classifier = DiscriminatorClassifier(input_dim, num_classes).to(device)
    classifier.shared.load_state_dict(torch.load("shared_backbone.pth", weights_only=True))
    classifier.classifier_head.load_state_dict(torch.load("classifier_head.pth", weights_only=True))

    # Generate samples
    dos_samples = generate_dos_samples(
        generator=generator,
        classifier=classifier,
        class_idx=class_idx,
        num_samples=1000,
        latent_dim=latent_dim,
        batch_size=128,
        device=device
    )

    # --- Evaluation of Generated Samples ---
    with torch.no_grad():
        # Get classifier predictions
        classifier.eval()
        dos_samples = dos_samples.to(device)
        preds = classifier(dos_samples, mode='classifier')
        predicted_labels = torch.argmax(preds, dim=1)

        # Compute how many were correctly classified as class 1 (DoS)
        correct = (predicted_labels == class_idx).sum().item()
        total = dos_samples.size(0)
        total = dos_samples.size(0)
        if total == 0:
            print("[Warning] No samples passed the filtering step. Classifier rejected all generated samples.")
        else:
            accuracy = correct / total
            probs = F.softmax(preds, dim=1)
            confidences = probs[:, class_idx].cpu().numpy()
            avg_conf = np.mean(confidences)
            min_conf = np.min(confidences)
            max_conf = np.max(confidences)

            print(f"[Evaluation] Accuracy on Generated DoS Samples: {accuracy:.4f}")
            print(f"[Evaluation] Confidence → Avg: {avg_conf:.4f}, Min: {min_conf:.4f}, Max: {max_conf:.4f}")



    # # Save as CSV
    # df = pd.DataFrame(dos_samples.cpu().numpy())
    # df.to_csv("generated_dos_samples.csv", index=False)
    # print("[Saved] generated_dos_samples.csv")
