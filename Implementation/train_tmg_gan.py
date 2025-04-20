import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from utils.losses import cosine_similarity  
from utils.plotting import plot_data_fitting

def train_tmg_gan(model, dataloader, latent_dim=100, epochs=500, 
                 t_d=1, t_g=1,  # Add parameters for iterations
                 lr=0.0002, beta1=0.5, beta2=0.999, device="cuda"):
    dc = model.dc
    generator = model.generator
    dc.to(device)
    generator.to(device)

    disc_params = list(dc.shared.parameters()) + list(dc.discriminator_head.parameters())
    gen_params = list(dc.classifier_head.parameters()) + list(generator.parameters())

    disc_optim = optim.Adam(disc_params, lr=lr, betas=(beta1, beta2))
    gen_optim = optim.Adam(gen_params, lr=lr, betas=(beta1, beta2))

    bce = nn.BCELoss()
    eps = 1e-8

    d_losses, g_losses, classifier_losses, classifier_accuracies, cos_sims = [], [], [], [], []
    checkpoints = [10, 20, 30, 40, 50]
    fig, axes = plt.subplots(1, len(checkpoints), figsize=(5 * len(checkpoints), 4))
    axes = np.array(axes).ravel()

    class_idx = 1  # DoS class

    for epoch in range(1, epochs + 1):
        epoch_d_loss, epoch_g_loss, epoch_class_loss = 0.0, 0.0, 0.0
        epoch_acc, epoch_cos_sim = 0.0, 0.0
        num_d_batches, num_g_batches = 0, 0

        # Collect data batches for this epoch
        epoch_batches = []
        for batch in dataloader:
            real_data, labels = batch
            real_data, labels = real_data.to(device), labels.to(device)
            
            # Filter for class_idx samples only
            mask = (labels == class_idx)
            if mask.sum() == 0:
                continue
                
            real_data = real_data[mask]
            labels = labels[mask]
            epoch_batches.append((real_data, labels))
        
        if len(epoch_batches) == 0:
            continue
            
        # Loop through batches for discriminator training
        for p in range(t_d):
            d_batch_losses = []
            
            for real_data, labels in epoch_batches:
                batch_size = real_data.size(0)
                num_d_batches += 1
                
                # Generate fake samples
                z = torch.randn(batch_size, latent_dim, device=device)
                fake_data = generator(z)
                
                # --- Discriminator ---
                disc_optim.zero_grad()
                real_lbls = torch.ones(batch_size, 1, device=device)
                fake_lbls = torch.zeros(batch_size, 1, device=device)
                
                pred_real = dc(real_data, mode='discriminator')
                pred_fake = dc(fake_data.detach(), mode='discriminator')
                
                features_real = dc(real_data, mode='features')
                logits_real = dc.classifier_head(features_real)
                probs_real = logits_real[:, class_idx]
                c_loss = -torch.log(probs_real + eps).mean()
                
                c_fake = dc(fake_data.detach(), mode='classifier')[:, class_idx]
                
                d_loss = bce(pred_real, real_lbls) + bce(pred_fake, fake_lbls) + c_fake.mean() + c_loss
                d_loss.backward()
                disc_optim.step()
                
                d_batch_losses.append(d_loss.item())
                epoch_class_loss += c_loss.item()
                
                # Track metrics for classifier
                with torch.no_grad():
                    acc = (probs_real > 0.5).float().mean().item()
                    epoch_acc += acc
            
            # Average loss for this discriminator iteration
            if d_batch_losses:
                epoch_d_loss += sum(d_batch_losses) / len(d_batch_losses)
        
        # Average discriminator loss across all t_d iterations
        if t_d > 0:
            epoch_d_loss /= t_d
        
        # Loop through batches for generator training
        for q in range(t_g):
            g_batch_losses = []
            cos_batch_sims = []
            
            for real_data, labels in epoch_batches:
                batch_size = real_data.size(0)
                num_g_batches += 1
                
                # Generate fake samples
                z = torch.randn(batch_size, latent_dim, device=device)
                fake_data = generator(z)
                
                # --- Generator + Classifier ---
                gen_optim.zero_grad()
                
                with torch.no_grad():
                    features_real = dc(real_data, mode='features')
                    probs_real = dc.classifier_head(features_real)[:, class_idx]
                
                features_fake = dc(fake_data, mode='features')
                pred_fake_d = dc.discriminator_head(features_fake)
                pred_fake_c = dc.classifier_head(features_fake)
                class_fake = pred_fake_c[:, class_idx]
                
                real_lbls_g = torch.ones(batch_size, 1, device=device)  # Ensure size matches pred_fake_d
                adv_loss = bce(pred_fake_d, real_lbls_g)
                
                # Calculate cosine similarity
                with torch.no_grad():
                    cos_intra = cosine_similarity(features_real, features_fake.detach())
                    z_other = torch.randn(batch_size, latent_dim, device=device)
                    fake_other = generator(z_other)
                    features_other = dc(fake_other, mode='features')
                    cos_inter = cosine_similarity(features_other, features_fake.detach())
                    cos_total = cos_inter.mean() - cos_intra.mean()
                    cos_batch_sims.append(cos_total.item())
                
                # Include cosine similarity in generator loss (as per algorithm)
                g_loss = adv_loss + class_fake.mean() + cos_total
                g_loss.backward()
                gen_optim.step()
                
                g_batch_losses.append(g_loss.item())
            
            # Average loss for this generator iteration
            if g_batch_losses:
                epoch_g_loss += sum(g_batch_losses) / len(g_batch_losses)
            if cos_batch_sims:
                epoch_cos_sim += sum(cos_batch_sims) / len(cos_batch_sims)
        
        # Average generator loss across all t_g iterations
        if t_g > 0:
            epoch_g_loss /= t_g
            epoch_cos_sim /= t_g
        
        # Scale metrics by the number of batches
        if num_d_batches > 0:
            epoch_class_loss /= num_d_batches
            epoch_acc /= num_d_batches
        
        # Append metrics for plotting
        d_losses.append(epoch_d_loss)
        g_losses.append(epoch_g_loss)
        classifier_losses.append(epoch_class_loss)
        classifier_accuracies.append(epoch_acc)
        cos_sims.append(epoch_cos_sim)
        
        print(f"[Epoch {epoch}/{epochs}] D Loss: {d_losses[-1]:.4f}, G Loss: {g_losses[-1]:.4f}, "
              f"Class Loss: {classifier_losses[-1]:.4f}, Clf Acc: {classifier_accuracies[-1]:.4f}, "
              f"Cos Sim: {cos_sims[-1]:.4f}",
              f"Avg Cosine Sim: {1.0 - np.mean(cos_sims):.4f}")
        
        # Visualization at checkpoints 
        if epoch in checkpoints:
            idx = checkpoints.index(epoch)
            ax = axes[idx]
            ax.clear()
            # Use the last batch from this epoch for visualization
            if len(epoch_batches) > 0:
                real_data, _ = epoch_batches[-1]
                n_plot = min(200, real_data.size(0))
                real_plot = real_data[:n_plot].detach()
                z_plot = torch.randn(n_plot, latent_dim, device=device)
                fake_plot = generator(z_plot)[:n_plot].detach()
                plot_data_fitting(real_plot, fake_plot, epoch, ax)

    # Final plots
    fig.suptitle("Data Fitting Over Training (Class 1 - DoS)")
    fig.tight_layout()
    fig.savefig("data_fitting_dos.png", dpi=300)
    plt.close(fig)

    def save_plot(data, title, filename, ylabel):
        plt.figure()
        plt.plot(data, label=title)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(filename, dpi=300)
        plt.close()

    save_plot(d_losses, "D Loss", "discriminator_loss.png", "Loss")
    save_plot(g_losses, "G Loss", "generator_loss.png", "Loss")
    save_plot(classifier_losses, "Classifier Loss", "classifier_loss.png", "Loss")
    save_plot(cos_sims, "Cosine Similarity (1-cos)", "cosine_similarity.png", "1 - cos(sim)")
    save_plot(classifier_accuracies, "Classifier Accuracy", "classifier_accuracy.png", "Accuracy")
    save_plot([1 - v for v in cos_sims], "Cosine Similarity", "cosine_similarity_actual.png", "cos(sim)")

    # Save models
    torch.save(generator.state_dict(), 'generator_class1_dos.pth')
    torch.save(dc.shared.state_dict(), 'shared_backbone.pth')
    torch.save(dc.discriminator_head.state_dict(), 'discriminator_head.pth')
    torch.save(dc.classifier_head.state_dict(), 'classifier_head.pth')

    return d_losses, g_losses, classifier_losses, classifier_accuracies, cos_sims