import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from utils.graph_generation import add_value_to_graphs

def main():
    # We add MW, ALOGP, and MACCS to the graphs separately due to memory constraints
    df = pd.read_parquet("WDR91.parquet", columns=["LABEL", "MW", "ALOGP", "MACCS"])

    # Train/test split (stratified)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['LABEL']
    )

    # Check if standard scaler exists, if not create it
    if not os.path.exists("scaler_MW.pkl") or not os.path.exists("scaler_ALOGP.pkl"):

        # Train/test split (stratified)
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['LABEL']
        )

        scaler_MW = StandardScaler()
        scaler_MW.fit(train_df["MW"].values.reshape(-1, 1))
        dump(scaler_MW, "scaler_MW.pkl")
        
        scaler_ALOGP = StandardScaler()
        scaler_ALOGP.fit(train_df["ALOGP"].values.reshape(-1, 1))
        dump(scaler_ALOGP, "scaler_ALOGP.pkl")

    # Load standard Scaler for MW and ALOGP
    scaler_MW = load("scaler_MW.pkl")
    scaler_ALOGP = load("scaler_ALOGP.pkl")

    # Transform MW and ALOGP
    df["MW"] = scaler_MW.transform(df["MW"].values.reshape(-1, 1)).flatten()
    df["ALOGP"] = scaler_ALOGP.transform(df["ALOGP"].values.reshape(-1, 1)).flatten()

    MW_dict = df["MW"].to_dict()  # {index: activity}
    ALOGP_dict = df["ALOGP"].to_dict()
    MACCS_dict = df["MACCS"].to_dict()

    add_value_to_graphs("train_full", MW_dict, ALOGP_dict, MACCS_dict, use_MACCS=False)
    add_value_to_graphs("valid_full", MW_dict, ALOGP_dict, MACCS_dict, use_MACCS=False)

if __name__ == "__main__":
    main()