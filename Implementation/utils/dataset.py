import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class LoadDataset(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        df = pd.read_csv(csv_file).values.astype(np.float32)
        self.features = torch.tensor(df[:, :-1], dtype=torch.float32)
        self.labels = torch.tensor(df[:, -1], dtype=torch.long)
        self.num_samples, self.num_features = self.features.shape
        self.num_classes = len(np.unique(self.labels.numpy()))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
