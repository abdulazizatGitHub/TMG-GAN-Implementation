import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

sys.path.append('D:/8th Semester/FYP/Repository/TMG-GAN-Implementation/Implementation')
from architecture.tmg_gan import TMGGAN
from architecture.discriminator_classifier import ClassifierHead, SharedBackbone

class AttackSampleGeneration:
    def __init__(self, model_folder, device="auto", latent_dim=100, num_classes=5, num_samples_per_class=100):
        self.model_folder = model_folder
        self.device = self._get_device(device)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class
        self.batch_size = 256  # Must be >1 for BatchNorm

        # Load models with safety checks
        self.tmg_gan = self._load_tmg_gan()
        self.classifier = self._load_classifier()
        self._validate_models()

    def _get_device(self, device_str):
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def _load_tmg_gan(self):
        """Load TMG-GAN with BatchNorm workaround"""
        try:
            model = TMGGAN(latent_dim=self.latent_dim, 
                          input_dim=42,
                          num_classes=self.num_classes)
            
            for i in range(self.num_classes):
                gen_path = os.path.join(self.model_folder, f'generator_class{i}.pth')
                if not os.path.exists(gen_path):
                    raise FileNotFoundError(f"Generator {i} not found at {gen_path}")
                
                # Load with weights_only=True for security
                state_dict = torch.load(gen_path, map_location=self.device, weights_only=True)
                
                # Workaround for BatchNorm during inference
                for name, module in model.generators[i].named_modules():
                    if isinstance(module, torch.nn.BatchNorm1d):
                        module.track_running_stats = False
                
                model.generators[i].load_state_dict(state_dict)
                model.generators[i].eval()
            
            return model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"TMG-GAN loading failed: {str(e)}")

    def _load_classifier(self):
        """Load classifier with BatchNorm handling"""
        try:
            backbone = SharedBackbone(input_dim=42).to(self.device)
            backbone_path = os.path.join(self.model_folder, 'shared_backbone.pth')
            
            # Disable BatchNorm tracking for inference
            for module in backbone.modules():
                if isinstance(module, torch.nn.BatchNorm1d):
                    module.track_running_stats = False
            
            backbone.load_state_dict(
                torch.load(backbone_path, map_location=self.device, weights_only=True))
            
            classifier = ClassifierHead(backbone, self.num_classes).to(self.device)
            classifier_path = os.path.join(self.model_folder, 'classifier_head.pth')
            classifier.load_state_dict(
                torch.load(classifier_path, map_location=self.device, weights_only=True))
            
            return classifier.eval()
        except Exception as e:
            raise RuntimeError(f"Classifier loading failed: {str(e)}")

    def _validate_models(self):
        """Validate with batch_size > 1 to avoid BatchNorm issues"""
        test_z = torch.randn(2, self.latent_dim, device=self.device)  # Note: 2 samples
        for gen in self.tmg_gan.generators:
            sample = gen(test_z)
            assert sample.shape == (2, 42), f"Generator output shape mismatch: {sample.shape}"
        
        pred = self.classifier(sample)
        assert pred.shape == (2, self.num_classes), f"Classifier output shape mismatch: {pred.shape}"

    def generate_class_samples(self, class_idx):
        """Generate samples with batch processing"""
        valid_samples = []
        attempts = 0
        progress = tqdm(total=self.num_samples_per_class, desc=f"Class {class_idx}")
        
        while len(valid_samples) < self.num_samples_per_class:
            attempts += 1
            z = torch.randn(self.batch_size, self.latent_dim, device=self.device)
            
            with torch.no_grad():
                samples = self.tmg_gan.generators[class_idx](z)
                preds = torch.softmax(self.classifier(samples), dim=1)
                
                # Filter valid samples
                mask = (torch.argmax(preds, dim=1) == class_idx) & (preds[:, class_idx] > 0.4)
                valid_batch = samples[mask]
                
                if valid_batch.numel() > 0:
                    valid_samples.extend(valid_batch.cpu().unbind(0))  # Move to CPU early
                    progress.update(len(valid_batch))
            
            if attempts > 100 and len(valid_samples) == 0:
                progress.close()
                raise RuntimeError(f"Failed to generate valid samples for class {class_idx}")
        
        progress.close()
        return torch.stack(valid_samples[:self.num_samples_per_class]).to(self.device), attempts

    def generate_all_samples(self):
        results = {}
        print(f"\nGenerating {self.num_samples_per_class} samples per class:")
        
        for class_idx in range(self.num_classes):
            try:
                samples, attempts = self.generate_class_samples(class_idx)
                results[class_idx] = samples
                print(f"Class {class_idx}: {len(samples)} samples (after {attempts} batches)")
            except RuntimeError as e:
                print(f"Warning: Skipping class {class_idx} due to error: {str(e)}")
        
        return results

    def visualize_results(self, samples, save_path="generated_samples.png"):
        fig, axes = plt.subplots(1, self.num_classes, figsize=(15, 5))
        if self.num_classes == 1:
            axes = [axes]
        
        real_data = self._load_real_samples()
        
        for class_idx in range(self.num_classes):
            ax = axes[class_idx]
            class_samples = samples[class_idx].cpu().numpy()
            
            if len(class_samples) > 0:
                ax.scatter(class_samples[:, 0], class_samples[:, 1], 
                          c='red', alpha=0.5, label='Generated')
                
                if class_idx in real_data:
                    real_samples = real_data[class_idx][:, :2]
                    ax.scatter(real_samples[:, 0], real_samples[:, 1],
                              c='blue', alpha=0.3, label='Real')
            
            ax.set_title(f"Class {class_idx}")
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nVisualization saved to {save_path}")

    def _load_real_samples(self):
        try:
            df = pd.read_csv('Implementation/Dataset/preprocessed_Dataset/augmented_normalized_unsw_nb15.csv')
            return {
                class_idx: df[df['label'] == class_idx].drop(columns=['label']).values[:1000]
                for class_idx in range(self.num_classes)
            }
        except Exception as e:
            print(f"\nWarning: Could not load real data - {str(e)}")
            return {}

    def save_samples(self, samples, output_dir="generated_samples"):
        os.makedirs(output_dir, exist_ok=True)
        
        for class_idx, data in samples.items():
            if len(data) > 0:
                pd.DataFrame(data.cpu().numpy()).to_csv(
                    os.path.join(output_dir, f"class_{class_idx}.csv"),
                    index=False
                )
        print(f"\nSamples saved to {output_dir}/")

if __name__ == "__main__":
    try:
        config = {
            "model_folder": "Implementation/models",
            "device": "auto",
            "num_samples_per_class": 100,
            "latent_dim": 100,
            "num_classes": 5
        }
        
        print("Initializing sample generator...")
        generator = AttackSampleGeneration(**config)
        
        print("\nStarting sample generation...")
        samples = generator.generate_all_samples()
        
        generator.visualize_results(samples)
        generator.save_samples(samples)
        
    except Exception as e:
        print(f"\nFatal error: {str(e)}", file=sys.stderr)
        sys.exit(1)