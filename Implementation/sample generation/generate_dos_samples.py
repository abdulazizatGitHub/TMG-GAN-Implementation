import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('D:/8th Semester/FYP/Repository/TMG-GAN-Implementation/Implementation')
from architecture.tmg_gan import TMGGAN
from architecture.discriminator_classifier import DiscriminatorClassifier

class AttackSampleGeneration:
    def __init__(self, model_folder, classifier_path, device="cuda", latent_dim=100, num_classes=5, num_samples_per_class=100):
        self.model_folder = model_folder
        self.classifier_path = classifier_path
        self.device = device
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class

        # Load models
        self.tmg_gan_model = self.load_tmg_gan_models()
        # Change to load_full_classifier_model instead of load_classifier_model
        self.classifier_model = self.load_full_classifier_model()

        # Move models to the specified device
        self.tmg_gan_model.to(self.device)
        self.classifier_model.to(self.device)

    def load_tmg_gan_models(self):
        """
        Load the TMG-GAN model from saved generator state dictionaries.
        
        Returns:
            model (TMGGAN): The TMG-GAN model with loaded generators.
        """
        try:
            # Initialize the TMGGAN with proper dimensions
            tmg_gan_model = TMGGAN(latent_dim=self.latent_dim, input_dim=42, num_classes=self.num_classes)
            
            for i in range(self.num_classes):
                generator_path = os.path.join(self.model_folder, f'generator_class{i}.pth')
                if not os.path.exists(generator_path):
                    print(f"Missing model file: {generator_path}")
                    return None
                
                # Load state dict directly since that's what was saved
                # Set weights_only=True to address the warning
                generator_state_dict = torch.load(generator_path, map_location=self.device, weights_only=True)
                tmg_gan_model.generators[i].load_state_dict(generator_state_dict)
                
            return tmg_gan_model
        except Exception as e:
            print(f"Error loading TMG-GAN models: {e}")
            return None

    def load_full_classifier_model(self):
        """
        Load both shared backbone and classifier head to recreate the full classifier.
        """
        try:
            # Create an instance of the full DiscriminatorClassifier
            full_model = DiscriminatorClassifier(input_dim=42, num_classes=self.num_classes).to(self.device)
            
            # Load shared backbone
            shared_path = os.path.join(self.model_folder, 'shared_backbone.pth')
            if not os.path.exists(shared_path):
                print(f"Missing model file: {shared_path}")
                return None
                
            # Set weights_only=True to address the warning
            shared_state_dict = torch.load(shared_path, map_location=self.device, weights_only=True)
            full_model.shared.load_state_dict(shared_state_dict)
            
            # Load classifier head
            if not os.path.exists(self.classifier_path):
                print(f"Missing model file: {self.classifier_path}")
                return None
                
            # Set weights_only=True to address the warning  
            classifier_state_dict = torch.load(self.classifier_path, map_location=self.device, weights_only=True)
            full_model.classifier_head.load_state_dict(classifier_state_dict)
            
            full_model.eval()
            return full_model
        except Exception as e:
            print(f"Error loading full classifier model: {e}")
            # Print more details about the state dict for debugging
            try:
                temp_dict = torch.load(self.classifier_path, map_location=self.device, weights_only=True)
                print(f"Loaded classifier head state dict keys: {temp_dict.keys()}")
                
                temp_shared = torch.load(shared_path, map_location=self.device, weights_only=True)
                print(f"Loaded shared backbone state dict keys: {temp_shared.keys()}")
            except Exception as debug_e:
                print(f"Error inspecting state dict: {debug_e}")
            return None

    def generate_attack_samples(self):
        """
        Generate attack samples from the pre-trained TMG-GAN model and classifier.

        Returns:
            List: List of generated attack samples for each class.
        """
        attack_samples = []
        try:
            # Initialize the sample counters for each class t1, t2, ..., tN
            t = [0] * self.num_classes  # list for t_k for each class

            for class_idx in range(self.num_classes):
                generated_class_samples = []
                print(f"Generating samples for class {class_idx}...")
                attempts = 0
                max_attempts = self.num_samples_per_class * 10  # Limit total attempts

                # Generate attack samples for this class until reaching the required number
                while t[class_idx] < self.num_samples_per_class and attempts < max_attempts:
                    attempts += 1
                    z = torch.randn(1, self.latent_dim, device=self.device)  # Sample random latent vectors
                    
                    # Generate pseudo sample for the class using the generator from TMG-GAN
                    fake_sample = self.tmg_gan_model.generate_samples(z, class_idx)  # Using generate_samples method
                    
                    try:
                        # Classifier to judge if the sample belongs to the correct class
                        predicted_label = self.classifier_model(fake_sample, mode='classifier')
                        predicted_class = predicted_label.argmax(dim=1).item()
                        
                        # If predicted label matches the actual class, accept the sample
                        if predicted_class == class_idx:
                            generated_class_samples.append(fake_sample)
                            t[class_idx] += 1  # Increment valid sample counter
                            if t[class_idx] % 10 == 0 or t[class_idx] == self.num_samples_per_class:
                                print(f"  - Generated {t[class_idx]}/{self.num_samples_per_class} valid samples (after {attempts} attempts)")
                    except Exception as sample_e:
                        print(f"Error evaluating sample: {sample_e}")
                
                if generated_class_samples:
                    attack_samples.append(torch.cat(generated_class_samples))
                else:
                    print(f"Warning: Failed to generate any valid samples for class {class_idx}")
                    # Add empty tensor to keep indices consistent
                    attack_samples.append(torch.empty(0, 42, device=self.device))
        except Exception as e:
            print(f"Error generating attack samples: {e}")
            return []
        
        return attack_samples

    def plot_data_fitting(self, real_data, generated_data, epoch, ax):
        """
        Plot the data fitting between real and generated samples.

        Parameters:
            real_data (torch.Tensor): Real data samples.
            generated_data (torch.Tensor): Generated data samples.
            epoch (int): The current epoch number (for labeling).
            ax (matplotlib.axes.Axes): Axis to plot on.
        """
        try:
            # Check if we have valid generated data
            if generated_data.size(0) == 0:
                ax.text(0.5, 0.5, f"No valid samples for class {epoch}", 
                        horizontalalignment='center', verticalalignment='center')
                ax.set_title(f"Class {epoch} - No Valid Samples")
                return
            
            # Select the first two dimensions for visualization
            # This is a simplification - consider using dimensionality reduction for high-dimensional data
            real_data_flat = real_data.view(real_data.size(0), -1).detach().cpu().numpy()
            generated_data_flat = generated_data.view(generated_data.size(0), -1).detach().cpu().numpy()

            ax.scatter(real_data_flat[:, 0], real_data_flat[:, 1], label='Real Data', color='b', alpha=0.5)
            ax.scatter(generated_data_flat[:, 0], generated_data_flat[:, 1], label='Generated Data', color='r', alpha=0.5)
            ax.set_title(f"Data Fitting for Class {epoch}")
            ax.legend()
        except Exception as e:
            print(f"Error plotting data fitting: {e}")
            ax.text(0.5, 0.5, f"Error plotting class {epoch}", 
                    horizontalalignment='center', verticalalignment='center')

    def get_real_data_for_class(self, class_idx, num_samples, device):
        """
        Fetch real data samples for a given class from the UNSW-NB15 dataset.
        
        Parameters:
            class_idx (int): The class index for which real data is needed.
            num_samples (int): The number of samples to fetch.
            device (str): The device (CUDA or CPU).
        
        Returns:
            torch.Tensor: Real data samples for the class.
        """
        try:
            # Load the dataset (replace with your correct dataset path)
            dataset_path = 'Implementation/Dataset/preprocessed_Dataset/augmented_normalized_unsw_nb15.csv'
            if not os.path.exists(dataset_path):
                print(f"Dataset file not found: {dataset_path}")
                print(f"Current working directory: {os.getcwd()}")
                return torch.zeros(num_samples, 42, device=device)  # Return zeros as fallback
            
            df = pd.read_csv(dataset_path)
            print(f"Dataset loaded with {len(df)} rows and columns: {df.columns}")

            # Filter the data for the class
            class_data = df[df['label'] == class_idx]
            print(f"Found {len(class_data)} samples for class {class_idx}")
            
            if len(class_data) == 0:
                print(f"No samples found for class {class_idx}. Available classes: {df['class'].unique()}")
                return torch.zeros(num_samples, 42, device=device)
            
            # If there are not enough samples, sample with replacement
            class_data = class_data.sample(n=num_samples, replace=len(class_data) < num_samples)
            
            # Extract features (assumes all columns except 'class' are features)
            if 'class' in class_data.columns:
                real_data = class_data.drop(columns=['class']).values
            else:
                print("'class' column not found in DataFrame, using all columns")
                real_data = class_data.values
            
            # Convert the numpy array to a PyTorch tensor
            real_data_tensor = torch.tensor(real_data, dtype=torch.float32, device=device)
            
            return real_data_tensor
        except Exception as e:
            print(f"Error fetching real data for class {class_idx}: {e}")
            return torch.zeros(num_samples, 42, device=device)  # Return a tensor of zeros if there is an error

    def run(self):
        try:
            print("Starting attack sample generation...")
            attack_samples = self.generate_attack_samples()

            if not attack_samples:
                raise ValueError("Attack samples generation failed.")

            print(f"Generated samples for {len(attack_samples)} classes")
            
            # Plot the generated samples vs real data for data fitting
            fig, axes = plt.subplots(1, self.num_classes, figsize=(15, 5))
            if self.num_classes == 1:
                axes = [axes]  # Convert to list if only one class

            for class_idx in range(self.num_classes):
                # Get real data for class_idx
                print(f"Preparing visualization for class {class_idx}...")
                real_data = self.get_real_data_for_class(class_idx, self.num_samples_per_class, self.device)
                
                if class_idx < len(attack_samples):
                    generated_data = attack_samples[class_idx]
                    print(f"Class {class_idx}: {len(generated_data)} generated samples")
                else:
                    print(f"No attack samples available for class {class_idx}")
                    generated_data = torch.empty(0, 42, device=self.device)
                
                self.plot_data_fitting(real_data, generated_data, class_idx, axes[class_idx])

            print("Saving plot...")
            plt.tight_layout()
            plt.savefig("attack_samples_visualization.png")
            print("Plot saved to attack_samples_visualization.png")
            plt.show()

        except Exception as e:
            print(f"Error in running attack sample generation process: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        # Set the paths to the model folder and individual model files
        model_folder = 'Implementation/models'
        classifier_path = os.path.join(model_folder, 'classifier_head.pth')

        print(f"Using model folder: {model_folder}")
        print(f"Classifier path: {classifier_path}")
        
        # Check if files exist
        if not os.path.exists(model_folder):
            print(f"Model folder not found: {model_folder}")
        if not os.path.exists(classifier_path):
            print(f"Classifier model not found: {classifier_path}")

        # Set the device to either CUDA or CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Initialize the AttackSampleGeneration with the models
        attack_sample_gen = AttackSampleGeneration(model_folder, classifier_path, device=device)

        # Run the attack sample generation and plotting
        attack_sample_gen.run()

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()