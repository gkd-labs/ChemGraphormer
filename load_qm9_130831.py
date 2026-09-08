"""
Load and save the clean QM9 dataset with 130,831 molecules.

Fetches from HuggingFace (n0w0f/qm9-csv), removes the 3054 uncharacterized
molecules using the official Figshare exclusion list, and saves to CSV.

Usage:
    python load_qm9_130831.py
"""

import re
import urllib.request
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


UNCHARACTERIZED_URL = "https://ndownloader.figshare.com/files/3195404"
OUTPUT_FILE = "qm9_130831.csv"


def fetch_uncharacterized_indices(url: str) -> set:
    """Download and parse the official list of 3054 uncharacterized molecule indices (0-based)."""
    urllib.request.urlretrieve(url, "uncharacterized.txt")
    with open("uncharacterized.txt") as f:
        lines = f.readlines()

    indices = set()
    for line in lines:
        parts = line.strip().split()
        if parts and parts[0].isdigit():
            indices.add(int(parts[0]) - 1)  # convert 1-based to 0-based
    return indices


def load_qm9() -> pd.DataFrame:
    """Load full QM9 dataset (133,885 molecules) from HuggingFace.

    Downloads the raw CSV directly from the Hub.
    """
    url = "https://huggingface.co/datasets/n0w0f/qm9-csv/resolve/main/qm9_dataset.csv"
    return pd.read_csv(url)


def main():
    print("Loading QM9 from HuggingFace...")
    df = load_qm9()
    print(f"Full dataset: {df.shape}")

    print("Fetching uncharacterized indices...")
    unchar = fetch_uncharacterized_indices(UNCHARACTERIZED_URL)
    print(f"Uncharacterized count: {len(unchar)}")

    df_clean = df.drop(index=list(unchar)).reset_index(drop=True)
    print(f"Clean dataset: {df_clean.shape}")

    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")
    print(df_clean[["smiles", "zero_point_energy"]].head())

    print("Splitting into train/val/test (110,000 : 10,000 : 10,831)...")
    training_dataset, validation_dataset, test_dataset = prepare_qm9_splits(csv_path=OUTPUT_FILE)

    train_smiles, train_labels = training_dataset
    val_smiles, val_labels = validation_dataset
    test_smiles, test_labels = test_dataset

    print(f"Train: {train_smiles.shape[0]} molecules, labels {train_labels.shape}")
    print(f"Val:   {val_smiles.shape[0]} molecules, labels {val_labels.shape}")
    print(f"Test:  {test_smiles.shape[0]} molecules, labels {test_labels.shape}")

    # Save each split to disk (smiles + label) so the work isn't lost when the process exits
    train_out = train_smiles.reset_index(drop=True).copy()
    train_out["zero_point_energy"] = train_labels.numpy().ravel()
    train_out.to_csv("qm9_train.csv", index=False)

    val_out = val_smiles.reset_index(drop=True).copy()
    val_out["zero_point_energy"] = val_labels.numpy().ravel()
    val_out.to_csv("qm9_val.csv", index=False)

    test_out = test_smiles.reset_index(drop=True).copy()
    test_out["zero_point_energy"] = test_labels.numpy().ravel()
    test_out.to_csv("qm9_test.csv", index=False)

    print("Saved splits to qm9_train.csv, qm9_val.csv, qm9_test.csv")

    return training_dataset, validation_dataset, test_dataset


def prepare_qm9_splits(csv_path: str = "/content/qm9_130831.csv"):
    """
    Loads the clean 130,831-molecule QM9 CSV and splits it into
    train : val : test = 110,000 : 10,000 : 10,831.

    Args:
        csv_path (str): Path to the qm9_130831.csv file produced by main().

    Returns:
        tuple: (training_dataset, validation_dataset, test_dataset), where
               each dataset is a (smiles_df, labels) pair. smiles_df is a
               pandas DataFrame with a 'smiles' column, and labels is a
               torch.float32 tensor of 'zero_point_energy' values.
    """
    qm9_v = pd.read_csv(csv_path)

    # train : val : test = 110,000 : 10,000 : 10,831
    train_val, test = train_test_split(
        qm9_v[["smiles", "zero_point_energy"]], test_size=10831, random_state=42
    )
    train, val = train_test_split(train_val, test_size=10000, random_state=42)

    # Extract SMILES and labels independently
    # Train split
    smiles_df = pd.DataFrame(train["smiles"])
    labels = pd.DataFrame(train["zero_point_energy"])
    labels = torch.tensor(labels.values, dtype=torch.float32)
    training_dataset = (smiles_df, labels)

    # Validation split
    smiles_df = pd.DataFrame(val["smiles"])
    labels = pd.DataFrame(val["zero_point_energy"])
    labels = torch.tensor(labels.values, dtype=torch.float32)
    validation_dataset = (smiles_df, labels)

    # Test split
    smiles_df = pd.DataFrame(test["smiles"])
    labels = pd.DataFrame(test["zero_point_energy"])
    labels = torch.tensor(labels.values, dtype=torch.float32)
    test_dataset = (smiles_df, labels)

    return training_dataset, validation_dataset, test_dataset

if __name__ == "__main__":
    main()
