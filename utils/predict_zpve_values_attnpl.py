import sys
import os
import argparse
import importlib
import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Attention-pooling ablation variants -> model module (each defines a differently-structured
# ChemGraphormerClassifier). Use --variant to pick the one matching your checkpoint.
VARIANT_MODEL_MAP = {
    "gate_zero":             "model.chemgraphormer_ablate_gate_init_zero",
    "gate_one":               "model.chemgraphormer_ablate_gate_init_one",
    "gate_two":               "model.chemgraphormer_ablate_gate_init_two",
    "edge":                   "model.chemgraphormer_ablate_edge",
    "rpe":                    "model.chemgraphormer_ablate_rpe",
    "rpe_edge":                "model.chemgraphormer_ablate_rpe_edge",
    "no_sinusoidal_rpe":       "model.chemgraphormer_ablate_no_sinusoidal_rpe",
    "no_edge_msg_and_gate":    "model.chemgraphormer_ablate_no_edge_msg_and_gate",
    "static_edge_flow":        "model.chemgraphormer_ablate_static_edge_flow",
}


def load_model_class(variant):
    """Dynamically import the ChemGraphormerClassifier matching the requested variant."""
    if variant not in VARIANT_MODEL_MAP:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(VARIANT_MODEL_MAP.keys())}")
    module = importlib.import_module(VARIANT_MODEL_MAP[variant])
    return module.ChemGraphormerClassifier

def load_model_from_checkpoint(checkpoint_path, model_params, device, model_class):
    """Initializes architecture with custom params and loads weights."""
    model = model_class(**model_params).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model

def make_prediction_attnpl(dataset_path, checkpoint_path, model_params=None, device="cuda", batch_size=128, model_class=None):
    """
    Inference entry point.
    :param model_params: Dictionary of model architecture settings. 
                         If None, defaults below are used.
    :param model_class: The ChemGraphormerClassifier class matching the checkpoint's
                         ablation variant. If None, defaults to the gate_one variant
                         (backward-compatible with earlier direct callers of this function).
    """
    if model_class is None:
        model_class = load_model_class("gate_one")

    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    
    if model_params is None:
        model_params = {}

    # Dynamically add d_node, d_edge, and lap_dim if not already present
    data_sample = dataset[0]
    if 'd_node' not in model_params:
        model_params['d_node'] = data_sample.x.size(1)
    if 'd_edge' not in model_params:
        model_params['d_edge'] = data_sample.edge_attr.size(1)
    if 'lap_dim' not in model_params:
        model_params['lap_dim'] = data_sample.lap_pos.size(1) if hasattr(data_sample, 'lap_pos') else 0

    # Fill in any other default model parameters if they are missing
    default_params = {
        'num_classes': 1,
        'd_model': 512,
        'n_heads': 8,
        'num_layers': 6,
        'd_ff': 1024,
        'num_freqs': 8,
        'dropout': 0.0
    }
    for key, value in default_params.items():
        if key not in model_params:
            model_params[key] = value

    model = load_model_from_checkpoint(checkpoint_path, model_params, device, model_class)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Running Inference...", leave=False):
            batch = batch.to(device)
            preds = model(batch)
            all_predictions.append(preds.cpu().view(-1))

            if hasattr(batch, "y") and batch.y is not None:
                all_targets.append(batch.y.view(-1).cpu())

    predictions = torch.cat(all_predictions, dim=0)
    mae = None
    if len(all_targets) > 0:
        targets = torch.cat(all_targets, dim=0)
        mae = F.l1_loss(predictions, targets).item()
        
    return pd.DataFrame(predictions.cpu().numpy().reshape(-1, 1), columns=['Predictions']), mae


def parse_args():
    parser = argparse.ArgumentParser(description="Run ChemGraphormer ablation (attention pooling) inference and save predictions to disk.")
    parser.add_argument("--dataset_path",    type=str,   required=True, help="Path to preprocessed dataset (.pt)")
    parser.add_argument("--checkpoint_path", type=str,   required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--variant",         type=str,   required=True, choices=list(VARIANT_MODEL_MAP.keys()),
                         help="Which ablation variant's architecture to load (must match how the checkpoint was trained).")
    parser.add_argument("--num_classes",     type=int,   default=1,     help="Specify 1 for regression tasks, 2 for binary classification tasks (default: 1)")
    parser.add_argument("--d_model",         type=int,   default=512,   help="Model hidden dimension (default: 512)")
    parser.add_argument("--n_heads",         type=int,   default=8,     help="Number of attention heads (default: 8)")
    parser.add_argument("--num_layers",      type=int,   default=6,     help="Number of encoder layers (default: 6)")
    parser.add_argument("--d_ff",            type=int,   default=1024,  help="Feed-forward hidden dimension (default: 1024)")
    parser.add_argument("--num_freqs",       type=int,   default=8,     help="Number of sinusoidal RPE frequencies (default: 8)")
    parser.add_argument("--dropout",         type=float, default=0.0,   help="Dropout rate (default: 0.0)")
    parser.add_argument("--batch_size",      type=int,   default=128,   help="Inference batch size (default: 128)")
    parser.add_argument("--device",          type=str,   default="cuda", help="Target device: 'cuda' or 'cpu' (default: cuda)")
    parser.add_argument("--output_dir",      type=str,   required=True, help="Directory to save prediction outputs")
    parser.add_argument("--split",           type=str,   default="test", help="Split name used in output filenames (default: test)")

    return parser.parse_args()


def save_predictions(preds_df, mae, output_dir, split):
    """
    Save predictions to CSV and print MAE (if available).

    Args:
        preds_df: DataFrame with a 'Predictions' column
        mae: Mean absolute error against targets, or None if the dataset had no labels
        output_dir: Directory to write output files
        split: Dataset split name used in the output filename
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{split}_predictions.csv")
    preds_df.to_csv(csv_path, index=False)
    print(f"Predictions saved to: {csv_path}")
    if mae is not None:
        print(f"Test MAE: {mae:.5f}")
    else:
        print("No labels found in dataset — MAE not computed.")


if __name__ == "__main__":
    args = parse_args()

    print(f"Dataset: {args.dataset_path}")
    dataset_size = len(torch.load(args.dataset_path, map_location="cpu", weights_only=False))
    print(f"Test dataset size: {dataset_size}")

    model_params = {
        'num_classes': args.num_classes,
        'd_model':     args.d_model,
        'n_heads':     args.n_heads,
        'num_layers':  args.num_layers,
        'd_ff':        args.d_ff,
        'num_freqs':   args.num_freqs,
        'dropout':     args.dropout,
    }

    model_class = load_model_class(args.variant)

    preds_df, mae = make_prediction_attnpl(
        dataset_path    = args.dataset_path,
        checkpoint_path = args.checkpoint_path,
        model_params    = model_params,
        device          = args.device,
        batch_size      = args.batch_size,
        model_class     = model_class,
    )

    save_predictions(preds_df, mae, args.output_dir, split=args.split)