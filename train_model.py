import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from utils.evaluate import f1_score as f1_score
from dataset.GraphDataset import GraphDataset
from utils.train import train
from utils.build_optimizer import build_optimizer
from model.GAT import GNNStack

class objectview:
    def __init__(self, d):
        self.__dict__ = d

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)

def main():
    for args in [
        {'model_type': 'GAT', 'dataset': 'train', 'num_layers': 2, 'heads': 16,
        'batch_size': 1024, 'hidden_dim': 512, 'dropout': 0.1, 'epochs': 100, 'output_dim': 1,
        'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 1e-4, 
        'use_maccs': False, 'lr': 0.001, 'num_workers': 8, 'model_path': 'GAT2_full_512.pt'},
    ]:
        args = objectview(args)

        if not args.model_type == 'GAT':
            args.heads = 1

        if args.dataset == 'train':
            train_dataset = GraphDataset(root_dir="train_full")
            valid_dataset = GraphDataset(root_dir="valid_full") if os.path.exists("valid_full") else None

        else:
            raise NotImplementedError("Unknown dataset")

        losses, best_model = train(train_dataset, valid_dataset, args)

        print("Minimum loss: {:.4f}".format(min(losses)))

        del train_dataset  # Free memory
        print("Training complete. Evaluating on validation set...")

        if valid_dataset is None:
            valid_dataset = GraphDataset(root_dir="valid_full")
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

            f1_test = f1_score(valid_loader, best_model, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        save_model_preds=True, model_type=args.model_type)
            print(f"Final Test F1@1: {f1_test:.4f}")

        plt.title("Training Curve: " + args.model_type)
        plt.plot(losses, label="Training Loss")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()