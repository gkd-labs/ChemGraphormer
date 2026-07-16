import os
import sys
import csv
import time
import argparse
import math
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model.chemgraphormer_ablate_rpe import ChemGraphormerClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="RPE ablation training for ChemGraphormer with warmup + cosine LR decay.")
    parser.add_argument("--train_path",              type=str,   required=True,  help="Path to preprocessed training dataset (.pt)")
    parser.add_argument("--valid_path",              type=str,   required=True,  help="Path to preprocessed validation dataset (.pt)")
    parser.add_argument("--ckpt_dir",                type=str,   required=True,  help="Directory to save checkpoints")
    parser.add_argument("--epoch_log_path",          type=str,   required=True,  help="CSV file path for logging epoch metrics")
    parser.add_argument("--epochs",                  type=int,   default=100,    help="Total number of training epochs (default: 100)")
    parser.add_argument("--warmup_epochs",           type=int,   default=10,     help="Number of linear warmup epochs (default: 10)")
    parser.add_argument("--batch_size",              type=int,   default=512,    help="Batch size (default: 512)")
    parser.add_argument("--lr",                      type=float, default=1e-4,   help="Peak learning rate after warmup (default: 1e-4)")
    parser.add_argument("--min_lr",                  type=float, default=1e-6,   help="Minimum LR at end of cosine decay (default: 1e-6)")
    parser.add_argument("--weight_decay",            type=float, default=5e-4,   help="AdamW weight decay (default: 5e-4)")
    parser.add_argument("--early_stopping_patience", type=int,   default=15,     help="Epochs without improvement before early stopping (default: 15)")
    parser.add_argument("--device",                  type=str,   default="cuda", help="Training device: 'cuda' or 'cpu' (default: cuda)")
    parser.add_argument("--num_classes",             type=int,   default=1,      help="Specify 1 for regression tasks, 2 for binary tasks, 3 for multi-label tasks and so on")
    parser.add_argument("--d_model",                 type=int,   default=512,    help="Model hidden dimension (default: 512)")
    parser.add_argument("--n_heads",                 type=int,   default=16,     help="Number of attention heads (default: 16)")
    parser.add_argument("--num_layers",              type=int,   default=8,      help="Number of encoder layers (default: 8)")
    parser.add_argument("--d_ff",                    type=int,   default=1536,   help="Feed-forward hidden dimension (default: 1536)")
    parser.add_argument("--num_freqs",               type=int,   default=16,     help="Number of sinusoidal RPE frequencies (default: 16)")
    parser.add_argument("--dropout",                 type=float, default=0.2,    help="Dropout rate (default: 0.2)")

    return parser.parse_args()


def move_batch_to_device(batch, device):
    """
    Move all relevant attributes of a PyG batch to the specified device.

    Args:
        batch: PyG batch object containing x, lap_pos, edge_index, edge_attr, batch, y
        device: Target device ("cuda" or "cpu")

    Returns:
        batch: PyG batch with all tensor attributes moved to device
    """
    for attr in ['x', 'lap_pos', 'edge_index', 'edge_attr', 'batch', 'y']:
        if hasattr(batch, attr):
            tensor = getattr(batch, attr)
            if isinstance(tensor, torch.Tensor):
                setattr(batch, attr, tensor.to(device))
    return batch


def count_parameters(model):
    """
    Print total and trainable parameters of a model.

    Args:
        model: torch.nn.Module
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total:,} total | {trainable:,} trainable")


def build_scheduler(optimizer, epochs, warmup_epochs, min_lr):
    """
    Build a scheduler with linear warmup followed by cosine decay.

    Linear warmup: LR rises from 0 to peak LR over warmup_epochs.
    Cosine decay: LR decays from peak LR to min_lr over remaining epochs.

    Args:
        optimizer: Optimizer to schedule
        epochs: Total number of training epochs
        warmup_epochs: Number of linear warmup epochs
        min_lr: Minimum LR at end of cosine decay

    Returns:
        scheduler: torch.optim.lr_scheduler.LambdaLR instance
    """
    peak_lr = optimizer.param_groups[0]['lr']
    decay_epochs = epochs - warmup_epochs

    def lr_lambda(epoch):
        # epoch is 0-indexed inside LambdaLR
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, decay_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine*(1-min_lr/peak_lr)+(min_lr/peak_lr)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, device, scaler):
    """
    Perform one epoch of training.

    Args:
        model: ChemGraphormerClassifier instance
        loader: DataLoader for training data
        optimizer: Optimizer
        device: Training device
        scaler: AMP GradScaler for mixed precision

    Returns:
        avg_loss: Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Train", leave=False, dynamic_ncols=True):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda"):
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), batch.y.view(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * batch.num_graphs
        del batch, pred, loss
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_one_epoch(model, loader, device):
    """
    Validate model for one epoch.

    Args:
        model: ChemGraphormerClassifier instance
        loader: DataLoader for validation data
        device: Validation device

    Returns:
        val_mae: Mean absolute error on validation set
    """
    model.eval()
    total_mae = 0
    num_samples = 0
    for batch in tqdm(loader, desc="Val", leave=False, dynamic_ncols=True):
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type="cuda"):
            pred = model(batch)
        total_mae += F.l1_loss(pred.view(-1), batch.y.view(-1), reduction='sum').item()
        num_samples += batch.y.numel()
        del batch, pred
    return total_mae / num_samples


def get_resume_checkpoint(ckpt_dir):
    """
    Return path to the last checkpoint if it exists.

    Args:
        ckpt_dir: Directory containing checkpoints

    Returns:
        path: Full path to last_checkpoint.pt or None
    """
    path = os.path.join(ckpt_dir, "last_checkpoint.pt")
    return path if os.path.exists(path) else None


def run_rpe_ablation(args):
    """
    Full model with no RPE ablation training loop for ChemGraphormer with linear warmup + cosine LR decay,
    early stopping, mixed precision, checkpointing, and CSV logging.

    Args:
        args: Parsed argparse.Namespace containing all training hyperparameters
    """
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print("Loading datasets...")
    train_dataset = torch.load(args.train_path, map_location="cpu", weights_only=False)  # custom PyG dataset objects require full unpickling
    valid_dataset = torch.load(args.valid_path, map_location="cpu", weights_only=False)  # custom PyG dataset objects require full unpickling

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    sample_graph = train_dataset[0]
    model = ChemGraphormerClassifier(
        d_node=sample_graph.x.size(1),
        d_edge=sample_graph.edge_attr.size(1),
        lap_dim=sample_graph.lap_pos.size(1),
        num_classes=args.num_classes,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        num_freqs=args.num_freqs,
        dropout=args.dropout,
    ).to(args.device)
    count_parameters(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs, args.min_lr)
    scaler = torch.amp.GradScaler('cuda')

    latest_ckpt = get_resume_checkpoint(args.ckpt_dir)
    start_epoch, best_val_mae = 1, float('inf')
    no_improve_epochs = 0
    best_ckpt_path = os.path.join(args.ckpt_dir, "chemgraphormer_rpe_ablation_best_model.pt")

    loaded_ckpt = None
    if latest_ckpt:
        loaded_ckpt = torch.load(latest_ckpt, map_location=args.device)
        model.load_state_dict(loaded_ckpt["model_state"])
        optimizer.load_state_dict(loaded_ckpt["optimizer_state"])
        scheduler.load_state_dict(loaded_ckpt["scheduler_state"])
        scaler.load_state_dict(loaded_ckpt.get("scaler_state", scaler.state_dict()))
        start_epoch = loaded_ckpt["epoch"] + 1
        best_val_mae = loaded_ckpt.get("best_val_mae", float('inf'))
        no_improve_epochs = loaded_ckpt.get("no_improve_epochs", 0)
        loaded_ckpt = None

    log_dir = os.path.dirname(args.epoch_log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    if start_epoch == 1:
        with open(args.epoch_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val MAE', 'Epoch Time (min)', 'LR'])

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, args.device, scaler)
        val_mae = validate_one_epoch(model, valid_loader, args.device)
        scheduler.step()
        epoch_minutes = (time.time() - start_time) / 60

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Train Loss={train_loss:.5f}, Val MAE={val_mae:.5f}, Time={epoch_minutes:.2f} min, LR={current_lr:.2e}")
        with open(args.epoch_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.5f}", f"{val_mae:.5f}", f"{epoch_minutes:.2f}", f"{current_lr:.2e}"])

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val MAE={best_val_mae:.5f}")
                break

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_mae": best_val_mae,
            "no_improve_epochs": no_improve_epochs,
        }, os.path.join(args.ckpt_dir, "last_checkpoint.pt"))

    print(f"Training complete. Best Val MAE: {best_val_mae:.5f} | Saved to: {best_ckpt_path}")


if __name__ == "__main__":
    args = parse_args()
    run_rpe_ablation(args)