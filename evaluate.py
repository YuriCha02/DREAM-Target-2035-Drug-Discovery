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
from sklearn.metrics import classification_report


class objectview:
    def __init__(self, d):
        self.__dict__ = d

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)

def evaluate(loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data)
            probs = torch.sigmoid(logits.view(-1))           # sigmoid scores
            preds = (probs > 0.5).long()                      # binary predictions
            labels = data.y.long()                            # ground truth labels

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    report = classification_report(
        all_labels, all_preds, target_names=['Class 0', 'Class 1'], zero_division=0
    )
    print(report)

def main():
    for args in [
        {'model_type': 'GAT', 'dataset': 'train', 'num_layers': 2, 'heads': 8,
        'batch_size': 1024, 'hidden_dim': 512, 'dropout': 0.1, 'epochs': 50, 'output_dim': 1,
        'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 1e-4, 
        'use_maccs': False, 'lr': 0.01, 'model_path': 'GAT_full_512.pt'},
    ]:
        args = objectview(args)

        if not args.model_type == 'GAT':
            args.heads = 1

        train_dataset = GraphDataset(root_dir="train_full")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        valid_dataset = GraphDataset(root_dir="valid_full")
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        model = GNNStack(train_dataset.num_node_features, args.hidden_dim, args, use_maccs=args.use_maccs)
        model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if os.path.exists(args.model_path):
            print(f"Loading existing model weights from {args.model_path}")
            model.load_state_dict(torch.load(args.model_path))
    
        model.eval()

        evaluate(train_loader, model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        evaluate(valid_loader, model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        plt.title("Training Curve: " + args.model_type)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()