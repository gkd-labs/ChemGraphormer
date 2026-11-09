"""
Attach labels to sharded graph files produced by run_sharded.py and save one
labeled PyG dataset list per shard.

Labels are matched using each shard's "indices" (row numbers in the original
CSV), so molecules dropped for invalid SMILES do not shift the alignment.

Usage:
    python utils/add_labels_sharded.py --shards-dir dataset --labels-csv dataset.csv \
        --label-col gap_val --out-dir labels

IMPORTANT: --labels-csv must be the SAME file, in the SAME row order, that was
passed to run_sharded.py (no sorting / shuffling / filtering beforehand).
"""
import glob
import os

import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm


def label_shard(shard, labels_all):
    node_features = shard["node_features"]
    edge_features = shard["edge_features"]
    edge_indices = shard["edge_pairs"]
    laplacian_pos = shard["laplacian_pe"]
    indices = shard["indices"]

    n = len(node_features)
    if not (len(edge_features) == len(edge_indices) == len(laplacian_pos) == len(indices) == n):
        raise ValueError("Shard lists have inconsistent lengths.")

    labels = labels_all[torch.as_tensor(indices, dtype=torch.long)]  # aligned via original row numbers

    out = []
    for i in range(n):
        pairs = edge_indices[i]
        if len(pairs) > 0:
            edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        out.append(Data(
            x=node_features[i],
            edge_index=edge_index,
            edge_attr=edge_features[i],
            lap_pos=laplacian_pos[i],
            y=labels[i].unsqueeze(0),
        ))
    return out


def add_graph_labels_sharded(shards_dir, labels_csv, out_dir, label_col="gap_val"):
    os.makedirs(out_dir, exist_ok=True)

    labels_all = torch.tensor(
        pd.read_csv(labels_csv, usecols=[label_col])[label_col].values, dtype=torch.float32
    )

    paths = sorted(glob.glob(os.path.join(shards_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt files in {shards_dir}")

    total = 0
    for path in tqdm(paths, desc="Labeling shards"):
        out_path = os.path.join(out_dir, "labeled_" + os.path.basename(path))
        if os.path.exists(out_path):
            continue  # resumable

        shard = torch.load(path, weights_only=False)
        if max(shard["indices"], default=-1) >= len(labels_all):
            raise ValueError(
                f"{path}: indices exceed label rows ({len(labels_all)}). "
                "Is --labels-csv the same file that was passed to run_sharded.py?"
            )

        data_list = label_shard(shard, labels_all)
        tmp = out_path + ".tmp"
        torch.save(data_list, tmp)
        os.replace(tmp, out_path)
        total += len(data_list)
        del shard, data_list

    print(f"Labeled {total} graphs this run -> {out_dir}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--shards-dir", required=True, help="Folder with shard_*.pt from run_sharded.py")
    p.add_argument("--labels-csv", required=True, help="Same CSV, same row order, as given to run_sharded.py")
    p.add_argument("--label-col", default="gap_val")
    p.add_argument("--out-dir", default="labeled_shards")
    args = p.parse_args()

    add_graph_labels_sharded(args.shards_dir, args.labels_csv, args.out_dir, args.label_col)
