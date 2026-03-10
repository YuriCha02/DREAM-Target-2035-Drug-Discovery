import os
import copy
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
from model.GAT import GNNStack
from utils.evaluate import evaluate_f1 
from utils.build_optimizer import build_optimizer

def train(train_dataset, valid_dataset, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # for actural dataset
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) # when using GraphDataset

    if valid_dataset:
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    model = GNNStack(train_dataset.num_node_features, args.hidden_dim, args, use_maccs=args.use_maccs)
    model.to(device)

    if os.path.exists(args.model_path):
        print(f"Loading existing model weights from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path))

    scheduler, opt = build_optimizer(args, model.parameters())  # define build_optimizer elsewhere

    losses = []
    best_model_state = None
    best_f1 = 0

    for epoch in range(args.epochs):
        
        all_labels, all_preds = [], []

        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):

            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch)

            labels = batch.y.float()
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels)
            loss.backward()

            opt.step()

            total_loss += loss.item() * batch.num_graphs

            preds = (torch.sigmoid(logits.view(-1)) > 0.5).long()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        total_loss /= len(train_loader.dataset)
        losses.append(total_loss)

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        if valid_loader:
            if epoch % 5 == 0:
                f1_pos = evaluate_f1(valid_loader, model, device, save_model_preds=True, model_type=args.model_type)
                print(f"Validation F1 (pos) at epoch {epoch}: {f1_pos:.4f}")
                if f1_pos > best_f1:
                    best_f1 = f1_pos
                    best_model_state = copy.deepcopy(model.state_dict())
                    torch.save(best_model_state, f'best_model_{args.model_path}')

        model_state = copy.deepcopy(model.state_dict())
        torch.save(model_state, args.model_path)

        f1_pos = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
        print(f"Epoch {epoch:>3d} — Loss {total_loss:.4f} — F1 (pos) {f1_pos:.4f}")

    return losses, best_model_state