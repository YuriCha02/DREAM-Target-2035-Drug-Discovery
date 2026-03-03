import os
import torch    
from torch_geometric.data import Dataset

class GraphDataset(Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None):
        self.root_dir = root_dir
        self.graph_files = sorted([f for f in os.listdir(root_dir) if f.endswith(".pt")])
        super().__init__(root_dir, transform, pre_transform)

    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        path = os.path.join(self.root_dir, self.graph_files[idx])
        data = torch.load(path, weights_only=False)
        return data
