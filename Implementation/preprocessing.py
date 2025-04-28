import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
from sklearn.utils import resample

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        print(f"✅ Data loaded from {self.file_path}")
        return self.data

class DataPreprocessor:
    def __init__(self, data):
        self.data = data
        self.processed_df = None
        self.X = None
        self.y = None
        self.X_columns = None
        self.scaler = MinMaxScaler()
        
        # Target counts from the table
        self.target_counts = {
            0: 56000,  # Normal
            1: 12264,  # DoS
            2: 10491,  # Reconnaissance
            3: 1133,   # Shellcode
            4: 130     # Worms
        }

    def preprocess_data(self):
        # 1. Label mapping for selected classes
        attack_cat_mapping = {
            'Normal': 0,
            'DoS': 1,
            'Reconnaissance': 2,
            'Shellcode': 3,
            'Worms': 4
        }

        # 2. Map attack_cat to label
        self.data['label'] = self.data['attack_cat'].map(attack_cat_mapping)

        # 3. Keep only selected classes
        selected_classes = [0, 1, 2, 3, 4]
        df = self.data[self.data['label'].isin(selected_classes)].copy()

        # 4. Drop unwanted columns
        for col in ['id', 'attack_cat']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # 5. Encode categorical columns
        categorical_columns = ['proto', 'service', 'state']
        label_encoder = LabelEncoder()
        for column in categorical_columns:
            if column in df.columns:
                df[column] = label_encoder.fit_transform(df[column].astype(str))

        # 6. Convert all to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # 7. Scale the features (except label)
        features = df.drop(columns=['label'])
        self.X = self.scaler.fit_transform(features)
        self.y = df['label'].copy()
        self.X_columns = features.columns
        self.processed_df = pd.DataFrame(self.X, columns=self.X_columns)
        self.processed_df['label'] = self.y  # 👈 label is added at the end

    def augment_data(self):
        """Augment the data to reach target class counts"""
        augmented_dfs = []
        
        # Process each class separately
        for class_idx in self.target_counts.keys():
            # Get the data for this class
            class_data = self.processed_df[self.processed_df['label'] == class_idx]
            current_count = len(class_data)
            target_count = self.target_counts[class_idx]
            
            print(f"Class {class_idx}: Current count = {current_count}, Target count = {target_count}")
            
            if current_count < target_count:
                # We need to oversample
                samples_needed = target_count - current_count
                print(f"  - Adding {samples_needed} samples through augmentation")
                
                # Use SMOTE-like approach: resample with randomization
                noise_factor = 0.05  # For adding small variations
                
                # Basic resampling with replacement
                oversampled_df = resample(
                    class_data,
                    replace=True,
                    n_samples=samples_needed,
                    random_state=42
                )
                
                # Add small Gaussian noise to create variations (except to label column)
                feature_cols = oversampled_df.columns.drop('label')
                
                # Add noise to continuous features to create unique samples
                for col in feature_cols:
                    std_dev = class_data[col].std() * noise_factor
                    noise = np.random.normal(0, std_dev, size=samples_needed)
                    # Ensure we don't go outside of [0,1] range for normalized data
                    oversampled_df[col] = np.clip(oversampled_df[col] + noise, 0, 1)
                
                # Add the original and oversampled data
                augmented_dfs.append(class_data)
                augmented_dfs.append(oversampled_df)
            
            elif current_count > target_count:
                # We need to undersample
                samples_to_keep = target_count
                print(f"  - Reducing by {current_count - target_count} samples through undersampling")
                
                # Undersample by random selection without replacement
                undersampled_df = resample(
                    class_data,
                    replace=False,
                    n_samples=samples_to_keep,
                    random_state=42
                )
                augmented_dfs.append(undersampled_df)
            
            else:
                # Count is already correct
                print("  - Count is already at target level")
                augmented_dfs.append(class_data)
        
        # Combine all the augmented classes
        self.processed_df = pd.concat(augmented_dfs, axis=0).reset_index(drop=True)
        print(f"\n✅ Data augmentation completed. New dataset shape: {self.processed_df.shape}")
        
        return self.processed_df

    def save_scaler(self, path='Implementation/models/minmax_scaler.pkl'):
        joblib.dump(self.scaler, path)
        print(f"✅ MinMaxScaler saved to {path}")

class DataCleaner:
    def __init__(self, dataframe):
        self.df = dataframe

    def clean(self):
        # Convert empty strings and whitespace to NaN
        self.df['label'] = self.df['label'].replace(r'^\s*$', np.nan, regex=True)

        # Remove rows with NaN values
        initial_rows = len(self.df)
        self.df = self.df[self.df['label'].notna()]
        cleaned_rows = len(self.df)
        print(f"🧹 Cleaned NaNs from label column: {initial_rows - cleaned_rows} rows removed")

        # Convert label to int after cleanup
        self.df['label'] = self.df['label'].astype(int)

        return self.df

class DataSaver:
    def __init__(self, resampled_data, output_file):
        self.resampled_data = resampled_data
        self.output_file = output_file

    def save(self):
        self.resampled_data.to_csv(self.output_file, index=False)
        print(f"✅ Preprocessed data saved to {self.output_file}")

def main():
    file_path = 'Implementation/Dataset/UNSW_NB15_training-set.csv'
    data_loader = DataLoader(file_path)
    raw_data = data_loader.load_data()

    # Preprocessing
    preprocessor = DataPreprocessor(raw_data)
    preprocessor.preprocess_data()

    # Cleaning
    cleaner = DataCleaner(preprocessor.processed_df)
    cleaned_data = cleaner.clean()
    
    # Update preprocessor with cleaned data
    preprocessor.processed_df = cleaned_data
    
    # Print original class distribution
    print("\n📊 Original class distribution:")
    print(cleaned_data['label'].value_counts())
    
    # Augment data to achieve target class counts
    augmented_data = preprocessor.augment_data()
    
    # Class stats after augmentation
    print("\n📊 Class distribution after augmentation:")
    print(augmented_data['label'].value_counts().sort_index())

    print("\n⚖️ Imbalance Ratios (IR) after augmentation:")
    normal_count = augmented_data['label'].value_counts().get(0, 1)
    for label, count in sorted(augmented_data['label'].value_counts().items()):
        if label != 0:
            print(f"Class {label}: IR = {round(normal_count / count, 2)}")

    print("\n📌 Final summary:")
    print({
        'Total samples': augmented_data.shape[0],
        'Number of classes': augmented_data['label'].nunique(),
        'Number of features': augmented_data.shape[1] - 1
    })

    # Save final dataset
    saver = DataSaver(augmented_data, 'Implementation/Dataset/preprocessed_Dataset/augmented_normalized_unsw_nb15.csv')
    saver.save()
    preprocessor.save_scaler()

if __name__ == "__main__":
    main()