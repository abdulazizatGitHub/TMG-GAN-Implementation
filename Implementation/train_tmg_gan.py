import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from utils.losses import cosine_similarity, pairwise_cosine_similarity
from utils.plotting import plot_data_fitting

class ModelTraining:
    def __init__(self, model, dataloader, latent_dim=100, epochs=500, lr=0.0002, beta1=0.5, beta2=0.999, 
                 t_d=1, t_g=1, device="cuda", validation_interval=10, checkpoint_interval=10):
        self.model = model
        self.dataloader = dataloader
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

        self.dc = self.model.dc
        self.generators = self.model.generators
        self.num_classes = self.model.num_classes

        self.dc.to(device)
        for g in self.generators:
            g.to(device)

        self.gen_optims = [optim.Adam(g.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)) for g in self.generators]
        self.disc_optim = optim.Adam(self.dc.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.cross_entropy = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()
        self.eps = 1e-8

    def train(self):
        d_losses, g_losses, classifier_losses, classifier_accuracies, cos_sims = [], [], [], [], []
        # 3. Set your checkpoints for 50-epoch test:
        checkpoints = [10, 20, 30, 40, 50]  # Evenly spaced for testing
        fig, axes = plt.subplots(1, len(checkpoints), figsize=(15, 3))
        axes = np.array(axes).ravel()

        for epoch in range(1, self.epochs + 1):
            epoch_cos_intra, epoch_cos_inter = 0.0, 0.0
            epoch_d_loss, epoch_g_loss, epoch_class_loss = 0.0, 0.0, 0.0
            epoch_acc, epoch_cos_sim = 0.0, 0.0
            num_d_batches, num_g_batches = 0, 0

            for class_idx in range(self.num_classes):
                epoch_batches = self.collect_epoch_batches(class_idx)
                if len(epoch_batches) == 0:
                    continue

                d_loss_sum, class_loss_sum, d_batches = self.train_discriminator(epoch_batches, class_idx)
                epoch_d_loss += d_loss_sum
                epoch_class_loss += class_loss_sum
                num_d_batches += d_batches

                g_loss_sum, cos_sim_sum, g_batches, intra_sim_sum, inter_sim_sum = self.train_generator(epoch_batches, class_idx)
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

        self.final_plots(d_losses, g_losses, classifier_losses, classifier_accuracies, cos_sims, fig)

    def collect_epoch_batches(self, class_idx):
        epoch_batches = []
        for batch in self.dataloader:
            real_data, labels = batch
            real_data, labels = real_data.to(self.device), labels.to(self.device)
            mask = (labels == class_idx)
            if mask.sum() == 0:
                continue
            real_data = real_data[mask]
            labels = labels[mask]
            epoch_batches.append((real_data, labels))
        return epoch_batches

    def train_discriminator(self, epoch_batches, class_idx):
        d_loss_sum, class_loss_sum, d_batches = 0.0, 0.0, 0

        for real_data, labels in epoch_batches:
            batch_size = real_data.size(0)
            d_batches += 1
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_data = self.generators[class_idx](z)
            self.disc_optim.zero_grad()
            real_lbls = torch.ones(batch_size, 1, device=self.device)
            fake_lbls = torch.zeros(batch_size, 1, device=self.device)
            pred_real = self.dc(real_data, mode='discriminator')
            pred_fake = self.dc(fake_data.detach(), mode='discriminator')
            features_real = self.dc(real_data, mode='features')
            logits_real = self.dc.classifier_head(features_real)
            c_loss_real = self.cross_entropy(logits_real, labels)
            features_fake = self.dc(fake_data.detach(), mode='features')
            logits_fake = self.dc.classifier_head(features_fake)
            c_loss_fake = self.cross_entropy(logits_fake, torch.full_like(labels, class_idx))
            d_loss = self.bce(pred_real, real_lbls) + self.bce(pred_fake, fake_lbls) + c_loss_real + c_loss_fake
            d_loss.backward()
            self.disc_optim.step()
            d_loss_sum += d_loss.item()
            class_loss_sum += c_loss_real.item()
        return d_loss_sum, class_loss_sum, d_batches

    def train_generator(self, epoch_batches, class_idx):
        g_loss_sum, cos_sim_sum, g_batches = 0.0, 0.0, 0
        intra_sim_sum, inter_sim_sum = 0.0, 0.0

        for real_data, labels in epoch_batches:
            batch_size = real_data.size(0)
            g_batches += 1
            z = torch.randn(batch_size, self.latent_dim, device=self.device) *2
            fake_data = self.generators[class_idx](z)
            self.gen_optims[class_idx].zero_grad()

            features_fake = self.dc(fake_data, mode='features')
            pred_fake_d = self.dc.discriminator_head(features_fake)
            pred_fake_c = self.dc.classifier_head(features_fake)

            # Generator Adversarial Loss (try to fool D)
            real_lbls_g = torch.ones(batch_size, 1, device=self.device)
            adv_loss = self.bce(pred_fake_d, real_lbls_g)

            # Classification Loss
            class_target = torch.full((batch_size,), class_idx, dtype=torch.long, device=self.device)
            class_loss = self.cross_entropy(pred_fake_c, class_target)

            # Intra-class Cosine Similarity (Eq. 2)
            with torch.no_grad():
                features_real = self.dc(real_data, mode='features')
                cos_intra = cosine_similarity(features_fake, features_real).mean()  # aligned, fine
                intra_sim_sum += cos_intra.item()

            # Inter-class Cosine Similarity (Eq. 3)
            cos_inter_total = 0.0
            num_other_classes = 0

            for other_class_idx in range(self.num_classes):
                if other_class_idx == class_idx:
                    continue
                z_other = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_other = self.generators[other_class_idx](z_other)
                features_other = self.dc(fake_other, mode='features')

                # ✅ All-to-all cosine matrix
                cos_inter_matrix = pairwise_cosine_similarity(features_fake, features_other)
                cos_inter_total += cos_inter_matrix.mean()
                num_other_classes += 1

            cos_inter_avg = cos_inter_total / (num_other_classes if num_other_classes > 0 else 1)
            inter_sim_sum += cos_inter_avg.item()

            # Final Cosine Similarity Loss (Eq. 4)
            cos_total = cos_inter_avg - cos_intra
            cos_sim_sum += cos_total.item()

            # Generator Loss (Eq. 8)
            g_loss = adv_loss + class_loss + cos_total
            g_loss.backward()
            self.gen_optims[class_idx].step()

            g_loss_sum += g_loss.item()

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
            for class_idx in range(self.num_classes):
                # Get fresh batch to avoid data leakage
                real_data = self.get_fresh_batch(class_idx)
                if real_data is None:
                    continue
                    
                # Generate matching fake samples
                n_samples = min(200, len(real_data))
                z = torch.randn(n_samples, self.latent_dim, device=self.device)
                fake_data = self.generators[class_idx](z).detach()
                
                all_real.append(real_data[:n_samples])
                all_fake.append(fake_data[:n_samples])
            
            if all_real:
                plot_data_fitting(torch.cat(all_real), torch.cat(all_fake), epoch, ax)
            fig.tight_layout()  # Ensure proper spacing

    def final_plots(self, d_losses, g_losses, classifier_losses, classifier_accuracies, cos_sims, fig):
        fig.suptitle("Data Fitting Over Training (All Classes)")
        fig.tight_layout()
        fig.savefig("Implementation/results/data_fitting_all_classes.png", dpi=300)
        plt.close(fig)
        self.save_plot(d_losses, "D Loss", "Implementation/results/discriminator_loss.png", "Loss")
        self.save_plot(g_losses, "G Loss", "Implementation/results/generator_loss.png", "Loss")
        self.save_plot(classifier_losses, "Classifier Loss", "Implementation/results/classifier_loss.png", "Loss")
        self.save_plot(cos_sims, "Cosine Similarity (1-cos)", "Implementation/results/cosine_similarity.png", "1 - cos(sim)")
        self.save_plot([1 - v for v in cos_sims], "Cosine Similarity", "Implementation/results/cosine_similarity_actual.png", "cos(sim)")
        for k, gen in enumerate(self.generators):
            torch.save(gen.state_dict(), f'Implementation/models/generator_class{k}.pth')
        torch.save(self.dc.shared.state_dict(), 'Implementation/models/shared_backbone.pth')
        torch.save(self.dc.discriminator_head.state_dict(), 'Implementation/models/discriminator_head.pth')
        torch.save(self.dc.classifier_head.state_dict(), 'Implementation/models/classifier_head.pth')

    def save_plot(self, data, title, filename, ylabel):
        plt.figure()
        plt.plot(data, label=title)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(filename, dpi=300)
        plt.close()
