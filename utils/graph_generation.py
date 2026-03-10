import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from joblib import Parallel, delayed
from tqdm import tqdm

def add_value_to_graphs(graph_dir, MW_dict, ALOGP_dict, MACCS_dict, use_MACCS=True):
    for filename in tqdm(os.listdir(graph_dir)):
        if not filename.endswith(".pt"):
            continue

        # Extract index from filename
        idx = int(filename.split("_")[1].split(".")[0])
        MW = MW_dict[idx]
        ALOGP = ALOGP_dict[idx]

        if use_MACCS:
            MACCS = MACCS_dict[idx]

        # Load and update graph
        path = os.path.join(graph_dir, filename)
        data = torch.load(path)
        data.MW = torch.tensor([MW], dtype=torch.float32)
        data.ALOGP = torch.tensor([ALOGP], dtype=torch.float32)
        if use_MACCS:
            data.MACCS = torch.tensor(MACCS, dtype=torch.float32).view(1, -1) # ensure MACCS is 2D

        # Overwrite graph with activity added
        torch.save(data, path)


# Create fullly connected graphs with MACCS
def generate_and_save_graphs(fps, labels, indices, MACCS, out_dir, n_jobs=8):
    os.makedirs(out_dir, exist_ok=True)

    def worker(fp, lbl, idx, MACCS):
        graph_path = os.path.join(out_dir, f"graph_{idx}.pt")
        if os.path.exists(graph_path):
            return
        try:
            graph = build_fully_connected_graph_with_maccs(fp, MACCS, lbl)
            #graph.MW = MW[idx]
            #graph.ALOGP = ALOGP[idx]
            torch.save(graph, graph_path)
        except Exception as e:
            print(f"[!] Failed at index {idx}: {e}")

    Parallel(n_jobs=n_jobs)(
        delayed(worker)(fps[i], labels[i], indices[i], MACCS[i]) for i in range(fps.shape[0])
    )

def vstack_fingerprints(row):
    """Concatenate 8 fingerprint arrays into a single vector."""
    return np.vstack([
        row['ECFP4'], row['ECFP6'], row['FCFP4'], row['FCFP6'],
        row['RDK'], row['AVALON'], row['ATOMPAIR'], row['TOPTOR'],
    ]).astype(np.float32)

def build_fully_connected_graph_with_maccs(fp_array, maccs, label):
    """
    fp_array: (8, 2048) numpy array
    maccs: (167,) numpy array
    """
    # Pad MACCS to 2048
    maccs_padded = np.zeros(2048, dtype=np.float32)
    maccs_padded[:167] = maccs

    # Stack all 9 nodes
    x = torch.tensor(np.vstack([fp_array, maccs_padded]), dtype=torch.float32)  # (9, 2048)

    # Fully connected edge_index
    edge_index = torch.tensor(
        [[i, j] for i in range(9) for j in range(9) if i != j],
        dtype=torch.long
    ).T  # shape: (2, num_edges)

    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

def add_value_to_graphs(graph_dir, MW_dict, ALOGP_dict, MACCS_dict, use_MACCS=True):
    for filename in tqdm(os.listdir(graph_dir)):
        if not filename.endswith(".pt"):
            continue

        # Extract index from filename
        idx = int(filename.split("_")[1].split(".")[0])
        MW = MW_dict[idx]
        ALOGP = ALOGP_dict[idx]

        if use_MACCS:
            MACCS = MACCS_dict[idx]

        # Load and update graph
        path = os.path.join(graph_dir, filename)
        try:
            data = torch.load(path)
        except Exception as e:
            print(f"Skipping {path} due to error: {e}")
            continue
        data.MW = torch.tensor([MW], dtype=torch.float32)
        data.ALOGP = torch.tensor([ALOGP], dtype=torch.float32)
        if use_MACCS:
            data.MACCS = torch.tensor(MACCS, dtype=torch.float32).view(1, -1) # ensure MACCS is 2D

        # Overwrite graph with activity added
        torch.save(data, path)