import torch
import torch.nn as nn

class MACCSEmbedder(nn.Module):
    def __init__(self, in_dim=167, out_dim=8, dropout=0.5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, maccs):
        return self.proj(maccs)
