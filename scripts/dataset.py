import torch
from torch.utils.data import Dataset

class MarketDataset(Dataset):
    def __init__(self, X, y_class, y_ret):
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_class = torch.tensor(y_class, dtype=torch.float32)
        self.y_ret = torch.tensor(y_ret, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx], self.y_ret[idx]