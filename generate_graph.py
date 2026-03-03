import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.graph_generation import generate_and_save_graphs, vstack_fingerprints

def main():

    # Output directories
    train = "train_full"
    valid = "valid_full"

    # Create output dirs
    os.makedirs(train, exist_ok=True)
    os.makedirs(valid, exist_ok=True)

    # Read only required columns
    cols = ['ECFP4','ECFP6','FCFP4','FCFP6','RDK','AVALON','ATOMPAIR','TOPTOR', 'MACCS', 'LABEL']
    df = pd.read_parquet('WDR91.parquet', columns=cols)

    # Train/test split (stratified)
    train_df, valid_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['LABEL']
    )

    # stack fingerprints
    train_fps = np.stack(train_df.apply(vstack_fingerprints, axis=1).values)
    train_MACCS = np.stack(train_df['MACCS'].values)
    train_labels = train_df['LABEL'].values
    valid_fps = np.stack(valid_df.apply(vstack_fingerprints, axis=1).values)
    valid_MACCS = np.stack(valid_df['MACCS'].values)
    valid_labels = valid_df['LABEL'].values

    # Generate graphs with labels
    generate_and_save_graphs(
        train_fps,
        train_labels,
        train_df.index.values,
        train_MACCS,
        out_dir=train,
        n_jobs=9
    )
    generate_and_save_graphs(
        valid_fps,
        valid_labels,
        valid_df.index.values,
        valid_MACCS,
        out_dir=valid,
        n_jobs=9
    )

if __name__ == "__main__":
    main()