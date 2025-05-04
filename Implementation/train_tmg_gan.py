import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from utils.plotting import plot_data_fitting
import torch.nn.functional as F

class ModelTraining:
    def __init__(self, model, dataloader, latent_dim=100, epochs=500, lr=0.0002, beta1=0.5, beta2=0.999, 
                 t_d=5, t_g=1, device="cuda", validation_interval=10, checkpoint_interval=10):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.t_d = t_d
        self.t_g = t_g
        self.device = device
        self.validation_interval = validation_interval
        self.checkpoint_interval = checkpoint_interval

        # Single optimizer for shared backbone + heads
        dc_params = list(self.model.shared_backbone.parameters()) + \
                   list(self.model.discriminator.parameters()) + \
                   list(self.model.classifier.parameters())
        self.dc_optim = optim.Adam(dc_params, lr=lr, betas=(beta1, beta2))
        # Optimizers
        self.gen_optims = [optim.Adam(g.parameters(), lr=lr, betas=(beta1, beta2)) 
                          for g in self.model.generators]
        # Loss functions
        self.cross_entropy = nn.CrossEntropyLoss()

    def train(self):
        d_losses, g_losses, classifier_losses, cos_sims = [], [], [], []
        checkpoints = [10, 20, 30, 40, 50]
        fig, axes = plt.subplots(1, len(checkpoints), figsize=(15, 3))
        axes = np.array(axes).ravel()

        for epoch in range(1, self.epochs + 1):
            epoch_cos_intra, epoch_cos_inter = 0.0, 0.0
            epoch_d_loss, epoch_g_loss, epoch_class_loss = 0.0, 0.0, 0.0
            epoch_cos_sim = 0.0
            num_d_batches, num_g_batches = 0, 0

            # Store all generated samples for this epoch
            all_generated_samples = {}

            for class_idx in range(self.model.num_classes):
                epoch_batches = self.collect_epoch_batches(class_idx)
                if len(epoch_batches) == 0:
                    continue

                # Train Discriminator
                d_loss_sum, class_loss_sum, d_batches, gen_samples = self.train_discriminator(
                    epoch_batches, class_idx
                )
                
                all_generated_samples[class_idx] = gen_samples  # Store for generator phase
                epoch_d_loss += d_loss_sum
                epoch_class_loss += class_loss_sum
                num_d_batches += d_batches

                # Train Generator (pass stored samples)
                g_loss_sum, cos_sim_sum, g_batches, intra_sim_sum, inter_sim_sum = \
                    self.train_generator(epoch_batches, class_idx, all_generated_samples)
                
                epoch_g_loss += g_loss_sum
                epoch_cos_sim += cos_sim_sum
                epoch_cos_intra += intra_sim_sum
                epoch_cos_inter += inter_sim_sum
                num_g_batches += g_batches

            if num_d_batches > 0: epoch_d_loss /= num_d_batches
            if num_g_batches > 0: epoch_g_loss /= num_g_batches
            if num_d_batches > 0: epoch_class_loss /= num_d_batches
            if num_g_batches > 0: epoch_cos_sim /= num_g_batches
            if num_g_batches > 0:
                epoch_cos_intra /= num_g_batches
                epoch_cos_inter /= num_g_batches

            d_losses.append(epoch_d_loss)
            g_losses.append(epoch_g_loss)
            classifier_losses.append(epoch_class_loss)
            cos_sims.append(epoch_cos_sim)

            print(f"\n📘 Epoch {epoch}/{self.epochs}")
            print(f" - Discriminator Loss: {epoch_d_loss:.4f}")
            print(f" - Generator Loss: {epoch_g_loss:.4f}")
            print(f" - Classifier Loss: {epoch_class_loss:.4f}")
            print(f" - Cosine Similarity Loss (inter - intra): {epoch_cos_sim:.4f}")
            print(f"   • Intra-class Cosine Similarity: {epoch_cos_intra:.4f}")
            print(f"   • Inter-class Cosine Similarity: {epoch_cos_inter:.4f}")

            self.plot_checkpoints(epoch, checkpoints, axes, fig)

        self.final_plots(d_losses, g_losses, classifier_losses, cos_sims, fig)

    def collect_epoch_batches(self, class_idx):
        epoch_batches = []
        for batch in self.dataloader:
            real_data, labels = batch
            real_data, labels = real_data.to(self.device), labels.to(self.device)
            mask = (labels == class_idx)
            # Skip if no samples of this class in the batch
            if mask.sum() == 0:
                continue
            # Get samples for this class
            class_data = real_data[mask]
            class_labels = labels[mask]
            
            # Ensure batch size is valid for BatchNorm (at least 2 samples)
            if class_data.size(0) >= 2:
                epoch_batches.append((class_data, class_labels))
            # If not enough samples, we'll skip this batch in training
        
        return epoch_batches

    def train_discriminator(self, epoch_batches, class_idx):
        d_loss_sum, class_loss_sum = 0.0, 0.0
        d_batches = 0
        generated_samples = {}  # Store generated samples per class
        
        for real_data, labels in epoch_batches:
            if real_data.size(0) < 2:
                continue
                
            self.dc_optim.zero_grad()
            
            z = torch.randn(real_data.size(0), self.latent_dim, device=self.device)
            fake_data = self.model.generators[class_idx](z)
            
            # Store generated samples for this class
            generated_samples[class_idx] = fake_data.detach()
            
            d_real = self.model.discriminate(real_data).mean()
            d_fake = self.model.discriminate(fake_data.detach()).mean()
            c_real = self.model.classify(real_data)
            class_loss = self.cross_entropy(c_real, labels)

            # NEW: Diagnostic prints
            if d_batches == 0:  # Print first batch of each epoch
                print(f"\nClass {class_idx} Diagnostics:")
                print(f"  D(real): {d_real.item():.4f} (should increase)")
                print(f"  D(fake): {d_fake.item():.4f} (should decrease)")
                print(f"  C(x) confidence: {torch.softmax(c_real, 1).mean().item():.4f}")
                print(f"  Classifier loss: {class_loss.item():.4f} (should decrease)")
            
            d_loss = (d_real - d_fake - class_loss) / self.model.num_classes
            d_loss.backward()
            self.dc_optim.step()
            
            d_loss_sum += d_loss.item()
            class_loss_sum += class_loss.item()
            d_batches += 1
        
        return d_loss_sum, class_loss_sum, d_batches, generated_samples

    def train_generator(self, epoch_batches, class_idx, all_generated_samples):
        g_loss_sum, cos_sim_sum = 0.0, 0.0
        intra_sim_sum, inter_sim_sum = 0.0, 0.0
        g_batches = 0

        total_classes = self.model.num_classes
        
        for real_data, _ in epoch_batches:
            if real_data.size(0) < 2:
                continue
                
            self.gen_optims[class_idx].zero_grad()
            
            # Generate new fake data for this batch
            z = torch.randn(real_data.size(0), self.latent_dim, device=self.device)
            fake_data = self.model.generators[class_idx](z)

            eps = 2e-4
            real_features = self.model.get_features(real_data)
            fake_features = self.model.get_features(fake_data)
            
            real_features = F.normalize(real_features + eps, p=2, dim=1)
            fake_features = F.normalize(fake_features + eps, p=2, dim=1)
            
            g_adv = self.model.discriminate(fake_data).mean()
            pred_class = self.model.classify(fake_data)
            class_loss = self.cross_entropy(pred_class, 
                                        torch.full((len(fake_data),), class_idx,
                                        device=self.device))

            # Intra-class similarity
            intra_sim = F.cosine_similarity(fake_features, real_features).mean()

            # NEW: Generator diagnostics
            if g_batches == 0:
                print(f"\nClass {class_idx} Generator:")
                print(f"  D(fake): {g_adv.item():.4f}")
                print(f"  C(x̃) confidence: {torch.softmax(pred_class, 1).mean().item():.4f}")
                print(f"  Classifier loss: {class_loss.item():.4f}")
                
                # Add these right before inter-class calculation
                print(f"\nInter-Class Debug:")
                print(f"  Calculating inter-class sim for class {class_idx}")
                print(f"  Will compare with: {[k for k in range(total_classes) if k != class_idx]}")
                print(f"  Available classes: {[k for k in all_generated_samples if k in all_generated_samples and all_generated_samples[k]]}")
            
            # Inter-class similarity using stored samples from other classes
            # Calculate using exact paper formulation
            inter_sim = 0

            # Inter-class similarity calculation
            inter_sim = torch.tensor(0.0, device=self.device)

            for k in range(total_classes):
                if k != class_idx:
                    if k in all_generated_samples and all_generated_samples[k]:
                        other_fake_data = all_generated_samples[k][k]
                        if hasattr(other_fake_data, 'size') and other_fake_data.size(0) >= 2:
                            other_features = F.normalize(
                                self.model.get_features(other_fake_data), p=2, dim=1
                            )
                            inter_sim += F.cosine_similarity(
                                fake_features.unsqueeze(1), 
                                other_features.unsqueeze(0), 
                                dim=2
                            ).mean()

            # Normalize by (N-1) as in paper
            if total_classes > 1:
                inter_sim = inter_sim / (total_classes - 1)
                
            cos_loss = inter_sim - intra_sim
            g_loss = ((-g_adv + class_loss) / self.model.num_classes) + cos_loss

            # NEW: Print cosine metrics
            if g_batches == 0:
                print(f"  Intra-class sim: {intra_sim.item():.4f} (should → 1)")
                print(f"  Inter-class sim: {inter_sim.item():.4f} (should → 0)")
                print(f"  Cosine loss: {cos_loss.item():.4f}")

            g_loss.backward()
            self.gen_optims[class_idx].step()
            
            g_loss_sum += g_loss.item()
            cos_sim_sum += cos_loss.item()
            intra_sim_sum += intra_sim.item()
            inter_sim_sum += inter_sim.item()
            g_batches += 1
        
        return g_loss_sum, cos_sim_sum, g_batches, intra_sim_sum, inter_sim_sum

    def get_fresh_batch(self, class_idx):
        """Get a fresh batch of real data for visualization"""
        for batch in self.dataloader:
            real_data, labels = batch
            real_data, labels = real_data.to(self.device), labels.to(self.device)
            mask = (labels == class_idx)
            if mask.sum() > 0:
                return real_data[mask]
        return None

    def plot_checkpoints(self, epoch, checkpoints, axes, fig):
        if epoch in checkpoints:
            idx = checkpoints.index(epoch)
            ax = axes[idx]
            
            # Accumulate samples across ALL classes
            all_real, all_fake = [], []
            for class_idx in range(self.model.num_classes):
                # Get fresh batch to avoid data leakage
                real_data = self.get_fresh_batch(class_idx)
                if real_data is None:
                    continue
                    
                # Generate matching fake samples
                n_samples = min(200, len(real_data))
                # Ensure n_samples is at least 2 for BatchNorm
                if n_samples < 2:
                    continue
                    
                z = torch.randn(n_samples, self.latent_dim, device=self.device)
                fake_data = self.model.generators[class_idx](z).detach()
                
                all_real.append(real_data[:n_samples])
                all_fake.append(fake_data[:n_samples])
            
            if all_real:
                plot_data_fitting(torch.cat(all_real), torch.cat(all_fake), epoch, ax)
            fig.tight_layout()  # Ensure proper spacing

    def final_plots(self, d_losses, g_losses, classifier_losses, cos_sims, fig):
        fig.suptitle("Data Fitting Over Training (All Classes)")
        fig.tight_layout()
        fig.savefig("Implementation/results/data_fitting_all_classes.png", dpi=300)
        plt.close(fig)
        self.save_plot(d_losses, "D Loss", "Implementation/results/discriminator_loss.png", "Loss")
        self.save_plot(g_losses, "G Loss", "Implementation/results/generator_loss.png", "Loss")
        self.save_plot(classifier_losses, "Classifier Loss", "Implementation/results/classifier_loss.png", "Loss")
        self.save_plot(cos_sims, "Cosine Similarity (1-cos)", "Implementation/results/cosine_similarity.png", "1 - cos(sim)")
        self.save_plot([1 - v for v in cos_sims], "Cosine Similarity", "Implementation/results/cosine_similarity_actual.png", "cos(sim)")
        for k, gen in enumerate(self.model.generators):
            torch.save(gen.state_dict(), f'Implementation/models/generator_class{k}.pth')
        torch.save(self.model.shared_backbone.state_dict(), 'Implementation/models/shared_backbone.pth')
        torch.save(self.model.discriminator.state_dict(), 'Implementation/models/discriminator_head.pth')
        torch.save(self.model.classifier.state_dict(), 'Implementation/models/classifier_head.pth')

    def save_plot(self, data, title, filename, ylabel):
        plt.figure()
        
        # Raw data
        plt.plot(data, alpha=0.2, label=f'{title} (raw)')
        
        # Moving average (window=5)
        moving_avg = np.convolve(data, np.ones(5)/5, mode='valid')
        plt.plot(moving_avg, label=f'{title} (avg)', linewidth=2)
        
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()