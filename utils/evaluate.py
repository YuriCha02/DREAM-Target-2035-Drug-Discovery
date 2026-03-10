import os
import torch
import pandas as pd
from sklearn.metrics import f1_score

def evaluate_f1(loader, model, device, save_model_preds=False, model_type=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data)
            preds = (torch.sigmoid(logits.view(-1)) > 0.5).long()
            labels = data.y.long()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    if save_model_preds:
        print(f"Saving model predictions for {model_type}")
        df = pd.DataFrame({'pred': all_preds, 'label': all_labels})
        df.to_csv(f'predictions_{model_type}.csv', index=False)

    f1_pos = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    return f1_pos