import sys
import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model.chemgraphormer_ablate_use_mean_pooling_gate_one import ChemGraphormerClassifier

def load_model_from_checkpoint(checkpoint_path, model_params, device):
    """Initializes architecture with custom params and loads weights."""
    model = ChemGraphormerClassifier(**model_params).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model

def make_prediction_meanpl(dataset_path, checkpoint_path, model_params=None, device="cuda", batch_size=128):
    """
    Inference entry point.
    :param model_params: Dictionary of model architecture settings. 
                         If None, defaults below are used.
    """
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

    model = load_model_from_checkpoint(checkpoint_path, model_params, device)
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