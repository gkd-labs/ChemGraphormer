import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data


def add_graph_labels(graphs_path: str, labels_csv: str, label_col: str = "gap_val"):
    """
    Loads previously generated graph tensors (node/edge features, edge indices,
    Laplacian PE) and attaches per-graph labels, producing a list of
    torch_geometric.data.Data objects ready for training/validation.

    Args:
        graphs_path (str): Path to the .pt file produced by graph_generator_ogb.py
                            (contains 'node_features', 'edge_features',
                            'edge_pairs', 'laplacian_pe').
        labels_csv (str): Path to a CSV file with labels in the same row order
                           as the SMILES list used to generate the graphs.
        label_col (str): Name of the label column in labels_csv (default: 'gap_val').

    Returns:
        list[Data]: A list of PyG Data objects, one per graph, each with
                    x, edge_index, edge_attr, lap_pos, and y set.
    """
    graphs = torch.load(graphs_path, weights_only=False)
    node_features = graphs['node_features']
    edge_features = graphs['edge_features']
    edge_indices = graphs['edge_pairs']
    laplacian_pos = graphs['laplacian_pe']

    labels_df = pd.read_csv(labels_csv)
    if len(labels_df) != len(node_features):
        raise ValueError(
            f"Row count mismatch: {len(labels_df)} labels vs {len(node_features)} graphs. "
            "Labels must be in the same order as the SMILES list used to build the graphs."
        )
    labels = torch.tensor(labels_df[label_col].values, dtype=torch.float32)

    # Attach labels — required for training and validation sets
    # Labels must be tensors
    graph_dataset_list = []
    for i in tqdm(range(len(node_features)), desc="Adding graph labels..."):
        pairs = edge_indices[i]
        if len(pairs) > 0:
            edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        data = Data(
            x=node_features[i],
            edge_index=edge_index,
            edge_attr=edge_features[i],
            lap_pos=laplacian_pos[i],
            y=labels[i].unsqueeze(0)
        )
        graph_dataset_list.append(data)

    return graph_dataset_list


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Attach labels to previously generated graphs and save a PyG dataset list."
    )
    parser.add_argument(
        "--graphs", type=str, required=True,
        help="Path to the .pt file from graph_generator_ogb.py (node_features, edge_features, edge_pairs, laplacian_pe)."
    )
    parser.add_argument(
        "--labels-csv", type=str, required=True,
        help="Path to a CSV with labels in the same row order as the SMILES used to build the graphs."
    )
    parser.add_argument(
        "--label-col", type=str, default="gap_val",
        help="Name of the label column in --labels-csv (default: 'gap_val')."
    )
    parser.add_argument(
        "--output", type=str, default="graphs.pt",
        help="Path to save the labeled PyG dataset list (default: 'graphs.pt')."
    )
    args = parser.parse_args()

    graph_dataset_list = add_graph_labels(args.graphs, args.labels_csv, args.label_col)
    torch.save(graph_dataset_list, args.output)
    print(f"Saved {len(graph_dataset_list)} labeled graphs to {args.output}")
