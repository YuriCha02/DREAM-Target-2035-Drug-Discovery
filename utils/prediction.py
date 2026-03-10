import torch

def prediction(loader, model, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data)
            preds = torch.sigmoid(logits.view(-1))
            all_preds.append(preds.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()

    return all_preds
