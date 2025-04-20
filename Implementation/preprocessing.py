import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        print(f"Data loaded from {self.file_path}")
        return self.data

class DataPreprocessor:
    def __init__(self, data):
        self.data = data
        self.X = None
        self.y = None
        self.X_columns = None

    def preprocess_data(self):
        attack_cat_mapping = {
            'Normal': 0,
            'DoS': 1,
            'Reconnaissance': 2,
            'Shellcode': 3,
            'Worms': 4
        }

        self.data['label'] = self.data['attack_cat'].map(attack_cat_mapping)
        selected_classes = [0, 1, 2, 3, 4]
        data_filtered = self.data[self.data['label'].isin(selected_classes)]
        data_filtered.dropna(subset=['label'], inplace=True)

        if 'id' in data_filtered.columns:
            data_filtered.drop(columns=['id'], inplace=True)
        if 'attack_cat' in data_filtered.columns:
            data_filtered.drop(columns=['attack_cat'], inplace=True)

        categorical_columns = ['proto', 'service', 'state']
        label_encoder = LabelEncoder()

        for column in categorical_columns:
            if column in data_filtered.columns:
                data_filtered[column] = label_encoder.fit_transform(data_filtered[column])

        self.X = data_filtered.drop(columns=['label'])
        self.y = data_filtered['label']
        self.X_columns = self.X.columns

        self.X = self.X.apply(pd.to_numeric, errors='coerce')
        scaler = MinMaxScaler()
        self.X = scaler.fit_transform(self.X)

class SMOTEHandler:
    def __init__(self, X, y, original_columns, target_sample_counts):
        self.X = X
        self.y = y
        self.X_resampled = None
        self.y_resampled = None
        self.original_columns = original_columns
        self.target_sample_counts = target_sample_counts

    def apply_oversampling(self):
        sampling_strategy = {}
        class_counts = self.y.value_counts()

        for label, desired_count in self.target_sample_counts.items():
            current_count = class_counts.get(label, 0)
            if current_count < desired_count:
                sampling_strategy[label] = desired_count

        print("\nFinal SMOTE Sampling Strategy:")
        for k, v in sampling_strategy.items():
            print(f"Class {k}: {v} samples")

        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        self.X_resampled, self.y_resampled = smote.fit_resample(self.X, self.y)

    def get_resampled_data(self):
        return pd.DataFrame(self.X_resampled, columns=self.original_columns), self.y_resampled

class DataSaver:
    def __init__(self, resampled_data, output_file):
        self.resampled_data = resampled_data
        self.output_file = output_file

    def save(self):
        self.resampled_data.to_csv(self.output_file, index=False)
        print(f"Resampled data saved to {self.output_file}")

def main():
    file_path = 'UNSW_NB15_training-set.csv'
    data_loader = DataLoader(file_path)
    data = data_loader.load_data()

    data_preprocessor = DataPreprocessor(data)
    data_preprocessor.preprocess_data()

    print("\nClass distribution before SMOTE:")
    print(data_preprocessor.y.value_counts())

    target_sample_counts = {
        0: 56000,
        1: 22264,
        2: 20491,
        3: 11133,
        4: 10130
    }

    smote_handler = SMOTEHandler(
        data_preprocessor.X,
        data_preprocessor.y,
        data_preprocessor.X_columns,
        target_sample_counts
    )
    smote_handler.apply_oversampling()

    X_resampled, y_resampled = smote_handler.get_resampled_data()
    print("\nClass distribution after SMOTE:")
    print(pd.Series(y_resampled).value_counts())

    data_after_summary = {
        'Total samples': X_resampled.shape[0],
        'Number of classes (after)': len(set(y_resampled)),
        'Number of features (after)': X_resampled.shape[1],
        'Head (after)': X_resampled[:5]
    }
    print("\nSummary after oversampling:")
    print(data_after_summary)

    resampled_data = pd.DataFrame(X_resampled, columns=data_preprocessor.X_columns)
    resampled_data['label'] = y_resampled
    data_saver = DataSaver(resampled_data, 'normalized_unsw_nb15_with_labels_encoded_oversampled.csv')
    data_saver.save()

if __name__ == "__main__":
    main()