
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

from torch_geometric.loader import DataLoader
from ogb.lsc import PCQM4Mv2Evaluator
from tqdm import tqdm

from model.chemgraphormer_meanp_gate0 import ChemGraphormerClassifier

#-----
# Args
#-----
def parse_args():
    parser = argparse.ArgumentParser(description="Single model prediction with ChemGraphormer — saves outputs to disk.")
    parser.add_argument("--dataset_path",     type=str,   required=True,      help="Path to preprocessed validation dataset (.pt)")
    parser.add_argument("--checkpoint_path", type=str,   required=True,      help="Path to model checkpoint state dict (.pt)")
    parser.add_argument("--num_classes",      type=int,   default=1,              help="Specify 1 for regression tasks, 2 for binary classification tasks (default: 1)")
    parser.add_argument("--d_model",          type=int,   default=512,         help="Model hidden dimension (default: 512)")
    parser.add_argument("--n_heads",          type=int,   default=16,          help="Number of attention heads (default: 16)")
    parser.add_argument("--num_layers",       type=int,   default=8,           help="Number of encoder layers (default: 8)")
    parser.add_argument("--d_ff",             type=int,   default=2048,        help="Feed-forward hidden dimension (default: 2048)")
    parser.add_argument("--num_freqs",        type=int,   default=16,          help="Number of sinusoidal RPE frequencies (default: 16)")
    parser.add_argument("--dropout",          type=float, default=0.15,         help="Dropout rate (default: 0.15)")
    parser.add_argument("--batch_size",       type=int,   default=128,         help="Inference batch size (default: 128)")
    parser.add_argument("--device",           type=str,   default="cuda",      help="Target device: 'cuda' or 'cpu' (default: cuda)")
    parser.add_argument("--output_dir",       type=str,   required=True,       help="Directory to save prediction outputs")
    parser.add_argument("--split",            type=str,   default="test",      help="Split name used in output filenames (default: test)")

    return parser.parse_args()

def save_predictions(preds, output_dir, split):
    """
    Save predictions in two formats:
      1. CSV with index and predicted value columns
      2. OGB submission format (.pt dict with y_pred key)

    Args:
        preds: Flat prediction tensor [N]
        output_dir: Directory to write all output files
        split: Dataset split name used in filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    preds_np = preds.numpy()

    # 1. CSV
    csv_path = os.path.join(output_dir, f"{split}_predictions.csv")
    pd.DataFrame({"idx": np.arange(len(preds_np)), "y_pred": preds_np}).to_csv(csv_path, index=False)
    print(f"CSV saved to:            {csv_path}")

    # 2. OGB submission format
    ogb_path = os.path.join(output_dir, f"{split}_submission.pt")
    torch.save({"y_pred": preds}, ogb_path)
    print(f"OGB submission saved to: {ogb_path}")

def move_batch_to_device(batch, device):
    """
    Move all relevant attributes of a PyG batch to the specified device.

    Args:
        batch: PyG batch object
        device: Target device

    Returns:
        batch: PyG batch with tensors on device
    """
    for attr in ['x', 'lap_pos', 'edge_index', 'edge_attr', 'batch']:
        if hasattr(batch, attr):
            tensor = getattr(batch, attr)
            if isinstance(tensor, torch.Tensor):
                setattr(batch, attr, tensor.to(device))
    return batch

def build_model_params(data_sample, args):
    """
    Build model parameter dict from a sample graph and parsed args.

    Args:
        data_sample: A single PyG Data object from the dataset
        args: Parsed argparse.Namespace

    Returns:
        dict: Model parameter dict
    """
    return {
        'd_node':      data_sample.x.size(1),
        'd_edge':      data_sample.edge_attr.size(1),
        'lap_dim':     data_sample.lap_pos.size(1),
        'num_classes': args.num_classes,
        'd_model':     args.d_model,
        'n_heads':     args.n_heads,
        'num_layers':  args.num_layers,
        'd_ff':        args.d_ff,
        'num_freqs':   args.num_freqs,
        'dropout':     args.dropout,
    }


def load_model(checkpoint_path, model_params, device):
    """
    Instantiate and load a ChemGraphormerClassifier from a state dict checkpoint.

    Args:
        checkpoint_path: Path to saved model state dict (.pt)
        model_params: Dict of model hyperparameters
        device: Target device

    Returns:
        model: Loaded model in eval mode
    """
    model = ChemGraphormerClassifier(**model_params).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))  
    model.eval()
    return model


def load_dataset_and_loader(dataset_path, batch_size):
    """
    Load a preprocessed dataset and return it with a DataLoader.

    Args:
        dataset_path: Path to preprocessed dataset (.pt)
        batch_size: Batch size for the DataLoader

    Returns:
        dataset: Loaded PyG dataset
        loader: DataLoader over the dataset
    """
    print("Loading dataset...")
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)  # custom PyG dataset objects require full unpickling
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataset, loader

def run_prediction_meanp(args):
    """
    Run prediction and save outputs.

    Args:
        args: Parsed argparse.Namespace
    """
    dataset, loader = load_dataset_and_loader(args.dataset_path, args.batch_size)
    model_params = build_model_params(dataset[0], args)
    model = load_model(args.checkpoint_path, model_params, args.device)
    model.eval() 

    all_preds = []
    print("\nRunning prediction...")
    with torch.no_grad(): 
        for batch in tqdm(loader, desc="Single Model Prediction", leave=False, dynamic_ncols=True):
            batch = move_batch_to_device(batch, args.device)
            preds = model(batch) 
            all_preds.append(preds.cpu())

    final_preds = torch.cat(all_preds, dim=0).squeeze() # Concatenate and remove singleton dimensions
    save_predictions(final_preds, args.output_dir, split=args.split)

    print("\nDone.")


if __name__ == "__main__":
    args = parse_args()
    run_prediction_meanp(args)
