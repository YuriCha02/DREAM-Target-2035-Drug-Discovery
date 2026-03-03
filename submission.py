import os
import numpy as np
import pyarrow.parquet as pq
from utils.graph_generation import generate_and_save_graphs, vstack_fingerprints, add_value_to_graphs
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import torch
from torch_geometric.loader import DataLoader
from dataset.GraphDataset import GraphDataset
from utils.build_optimizer import build_optimizer
from model.GAT import GNNStack
from utils.prediction import prediction

class objectview:
    def __init__(self, d):
        self.__dict__ = d

def main():

    # Output directories
    submission = "submission_full"

    # Create output dirs
    os.makedirs(submission, exist_ok=True)

    # Read only required columns
    cols = ['ECFP4','ECFP6','FCFP4','FCFP6','RDK','AVALON','ATOMPAIR','TOPTOR', 'MACCS']
    parquet_file = pq.ParquetFile("Step2_TestData_Target2035.parquet")

    batch_size = 4096  # adjust to fit your memory

    global_idx_offset = 0

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=cols):
        df = batch.to_pandas()
        for col in cols:
            df[col] = df[col].apply(lambda x: np.fromstring(x, sep=','))

        test_fps = np.stack(df.apply(vstack_fingerprints, axis=1).values)
        test_MACCS = np.stack(df['MACCS'].values)
        test_labels = np.full(test_fps.shape[0], -1, dtype=np.float32)

        # Use absolute index for naming
        indices = np.arange(global_idx_offset, global_idx_offset + test_fps.shape[0])
        generate_and_save_graphs(
            test_fps,
            test_labels,
            indices,
            test_MACCS,
            out_dir=submission,
            n_jobs=8
        )

        global_idx_offset += test_fps.shape[0]

    # We add MW, ALOGP, and MACCS to the graphs separately due to memory constraints
    df = pd.read_parquet("Step2_TestData_Target2035.parquet", columns=["MW", "AlogP"])

    # Check if standard scaler exists, if not create it
    if not os.path.exists("scaler_MW.pkl") or not os.path.exists("scaler_ALOGP.pkl"):

        scaler_MW = StandardScaler()
        scaler_MW.fit(df["MW"].values.reshape(-1, 1))
        dump(scaler_MW, "scaler_MW.pkl")
        
        scaler_ALOGP = StandardScaler()
        scaler_ALOGP.fit(df["ALOGP"].values.reshape(-1, 1))
        dump(scaler_ALOGP, "scaler_ALOGP.pkl")

    # Load standard Scaler for MW and ALOGP
    scaler_MW = load("scaler_MW.pkl")
    scaler_ALOGP = load("scaler_ALOGP.pkl")

    # Transform MW and ALOGP
    df["MW"] = scaler_MW.transform(df["MW"].values.reshape(-1, 1)).flatten()
    df["AlogP"] = scaler_ALOGP.transform(df["AlogP"].values.reshape(-1, 1)).flatten()

    MW_dict = df["MW"].to_dict()  # {index: activity}
    ALOGP_dict = df["AlogP"].to_dict()

    add_value_to_graphs("submission_full", MW_dict, ALOGP_dict, None, use_MACCS=False)

    for args in [
        {'model_type': 'GAT', 'dataset': 'train', 'num_layers': 2, 'heads': 8,
        'batch_size': 1024, 'hidden_dim': 512, 'dropout': 0.1, 'epochs': 50, 'output_dim': 1,
        'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 1e-4, 
        'use_maccs': False, 'lr': 0.01, 'num_workers': 8, 'model_path': 'GAT_full_512.pt'},
    ]:
        args = objectview(args)

        test_dataset = GraphDataset(root_dir="submission_full")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = GNNStack(test_dataset.num_node_features, args.hidden_dim, args, use_maccs=args.use_maccs)
        model.load_state_dict(torch.load(args.model_path))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        all_preds = prediction(test_loader, model, device)

        # load the submission form
        submission_df = pd.read_csv("Step2_submission_file.csv")
        submission_df['Score'] = all_preds

        # In the column `Sel_50` we will put the top 50 of the scores as 1, rest as 0
        submission_df['Sel_50'] = 0
        top_50_indices = submission_df['Score'].nlargest(50).index
        submission_df.loc[top_50_indices, 'Sel_50'] = 1

        # Save the submission file
        submission_df.to_csv("Step2_submission_file.csv", index=False)

if __name__ == "__main__":
    main()