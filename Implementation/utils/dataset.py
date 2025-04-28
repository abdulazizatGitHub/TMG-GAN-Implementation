import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class LoadDataset(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        df = pd.read_csv(csv_file)

        # 🔍 Separate label and features explicitly
        self.labels = torch.tensor(df['label'].values, dtype=torch.long)
        self.features = torch.tensor(df.drop(columns=['label']).values.astype(np.float32), dtype=torch.float32)

        self.num_samples, self.num_features = self.features.shape
        self.num_classes = len(np.unique(self.labels.numpy()))

        # ✅ Debug print (optional)
        print("\n✅ LoadDataset initialized:")
        print(f" - Samples: {self.num_samples}")
        print(f" - Features: {self.num_features}")
        print(f" - Classes: {self.num_classes}")
        print(" - Class distribution:", dict(zip(*np.unique(self.labels.numpy(), return_counts=True))))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
