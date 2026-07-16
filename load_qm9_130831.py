"""
Load and save the clean QM9 dataset with 130,831 molecules.

Fetches from HuggingFace (n0w0f/qm9-csv), removes the 3054 uncharacterized
molecules using the official Figshare exclusion list, and saves to CSV.

Usage:
    python load_qm9_130831.py
"""

import re
import urllib.request
from datasets import load_dataset
import pandas as pd


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
    """Load full QM9 dataset (133,885 molecules) from HuggingFace."""
    ds = load_dataset("n0w0f/qm9-csv", split="train")
    return ds.to_pandas()


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


if __name__ == "__main__":
    main()
