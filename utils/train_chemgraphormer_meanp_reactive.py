import os
import sys
import csv
import time
import argparse
import math
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from ogb.lsc import PCQM4Mv2Evaluator
from tqdm import tqdm

from model.chemgraphormer_meanp_gate0 import ChemGraphormerClassifier

"""
Args
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain ChemGraphormer with reactive LR, per-layer LR scaling, and dynamic batch scheduling.")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--valid_path", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--epoch_log_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--lr_factor", type=float, default=0.8)
    parser.add_argument("--bs_phase1", type=int, default=2048)
    parser.add_argument("--bs_phase2", type=int, default=512)
    parser.add_argument("--bs_phase3", type=int, default=256)
    parser.add_argument("--bs_phase4", type=int, default=128)
    parser.add_argument("--lr_mult_phase1", type=float, default=4.0)
    parser.add_argument("--lr_mult_phase2", type=float, default=1.0)
    parser.add_argument("--lr_mult_phase3", type=float, default=0.5)
    parser.add_argument("--lr_mult_phase4", type=float, default=0.25)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1536)
    parser.add_argument("--num_freqs", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)

    return parser.parse_args()


def move_batch_to_device(batch, device):
    """Move batch tensors to device."""
    for attr in ['x', 'lap_pos', 'edge_index', 'edge_attr', 'batch', 'y']:
        if hasattr(batch, attr):
            tensor = getattr(batch, attr)
            if isinstance(tensor, torch.Tensor):
                setattr(batch, attr, tensor.to(device))
    return batch


def count_parameters(model):
    """Print model parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total:,} total | {trainable:,} trainable")


def build_optimizer(model, lr_multiplier, base_lr, weight_decay):
    """Build per-layer AdamW optimizer."""
    return torch.optim.AdamW([
        {"params": model.encoder.embed.parameters(), "lr": base_lr * lr_multiplier * 1.0},
        {"params": model.encoder.layers[0].parameters(), "lr": base_lr * lr_multiplier * 0.9},
        {"params": model.encoder.layers[1].parameters(), "lr": base_lr * lr_multiplier * 0.8},
        {"params": model.encoder.layers[2].parameters(), "lr": base_lr * lr_multiplier * 0.7},
        {"params": model.encoder.layers[3].parameters(), "lr": base_lr * lr_multiplier * 0.6},
        {"params": model.encoder.layers[4].parameters(), "lr": base_lr * lr_multiplier * 0.5},
        {"params": model.encoder.layers[5].parameters(), "lr": base_lr * lr_multiplier * 0.4},
        {"params": model.encoder.layers[6].parameters(), "lr": base_lr * lr_multiplier * 0.3},
        {"params": model.encoder.layers[7].parameters(), "lr": base_lr * lr_multiplier * 0.2},
        {"params": model.head.parameters(), "lr": base_lr * lr_multiplier * 10.0},
    ], weight_decay=weight_decay)


def build_scheduler(optimizer, patience, factor, min_lr):
    """Build ReduceLROnPlateau scheduler."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=factor,
        patience=patience,
        min_lr=min_lr
    )


def train_one_epoch(model, loader, optimizer, device, scaler):
    """Run one training epoch."""
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Train", leave=False, dynamic_ncols=True):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda"):
            pred = model(batch)
            loss = F.l1_loss(pred.squeeze(), batch.y.squeeze())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * batch.num_graphs
        del batch, pred, loss
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_one_epoch(model, loader, device, evaluator):
    """Run validation epoch."""
    model.eval()
    preds, labels = [], []
    for batch in tqdm(loader, desc="Val", leave=False, dynamic_ncols=True):
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type="cuda"):
            pred = model(batch)
        preds.append(pred.cpu())
        labels.append(batch.y.cpu())
        del batch, pred
    preds = torch.cat(preds, dim=0).view(-1)
    labels = torch.cat(labels, dim=0).view(-1)
    return evaluator.eval({"y_pred": preds, "y_true": labels})["mae"]


def get_resume_checkpoint(ckpt_dir):
    """Return last checkpoint path if exists."""
    path = os.path.join(ckpt_dir, "last_checkpoint.pt")
    return path if os.path.exists(path) else None


def get_all_lrs(optimizer):
    """Return list of LRs for all param groups."""
    return [g["lr"] for g in optimizer.param_groups]


def train_with_meanp_gate0_reactive(args):
    """Main training loop."""
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print("Loading datasets...")
    train_dataset = torch.load(args.train_path, map_location="cpu", weights_only=False)
    valid_dataset = torch.load(args.valid_path, map_location="cpu", weights_only=False)

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
    evaluator = PCQM4Mv2Evaluator()

    scaler = torch.amp.GradScaler('cuda')
    current_batch_size, current_lr_multiplier = 0, 0
    start_epoch, best_val_mae = 1, float('inf')
    loaded_ckpt = None

    latest_ckpt = get_resume_checkpoint(args.ckpt_dir)
    if latest_ckpt:
        print(f"Resuming from last checkpoint: {latest_ckpt}")
        loaded_ckpt = torch.load(latest_ckpt, map_location=args.device)
        model.load_state_dict(loaded_ckpt["model_state"])
        start_epoch = loaded_ckpt["epoch"] + 1
        best_val_mae = loaded_ckpt.get("best_val_mae", float('inf'))
        scaler.load_state_dict(loaded_ckpt.get("scaler_state", scaler.state_dict()))
    else:
        print("No saved checkpoint found, training will start from scratch.")

    log_dir = os.path.dirname(args.epoch_log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    lr_col_names = [
        'LR_embed',
        'LR_layer0', 'LR_layer1', 'LR_layer2', 'LR_layer3',
        'LR_layer4', 'LR_layer5', 'LR_layer6', 'LR_layer7',
        'LR_head'
    ]

    if start_epoch == 1:
        with open(args.epoch_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val MAE', 'Epoch Time (min)'] + lr_col_names)
    else:
        print(f"Appending to existing log file at: {args.epoch_log_path}")

    for epoch in range(start_epoch, args.epochs + 1):

        if 1 <= epoch <= 30:
            new_batch_size, lr_multiplier = args.bs_phase1, args.lr_mult_phase1
        elif 31 <= epoch <= 70:
            new_batch_size, lr_multiplier = args.bs_phase2, args.lr_mult_phase2
        elif 71 <= epoch <= 120:
            new_batch_size, lr_multiplier = args.bs_phase3, args.lr_mult_phase3
        else:
            new_batch_size, lr_multiplier = args.bs_phase4, args.lr_mult_phase4

        if new_batch_size != current_batch_size or lr_multiplier != current_lr_multiplier or epoch == start_epoch:
            current_batch_size, current_lr_multiplier = new_batch_size, lr_multiplier

            train_loader = DataLoader(train_dataset, batch_size=new_batch_size, shuffle=True, pin_memory=True)
            valid_loader = DataLoader(valid_dataset, batch_size=new_batch_size, shuffle=False, pin_memory=True)

            # Always build fresh optimizer and scheduler at correct phase LR
            optimizer = build_optimizer(model, lr_multiplier, args.lr, args.weight_decay)
            scheduler = build_scheduler(optimizer, args.lr_patience, args.lr_factor, args.min_lr)

            # Only restore optimizer/scheduler state if saved phase matches current phase
            if loaded_ckpt and 'optimizer_state' in loaded_ckpt:
                saved_cfg = loaded_ckpt.get("train_cfg", {})
                if (saved_cfg.get("lr_multiplier") == current_lr_multiplier and
                        saved_cfg.get("batch_size") == current_batch_size):
                    print(f"Phase matched, restoring optimizer and scheduler state from previous checkpoint.")
                    optimizer.load_state_dict(loaded_ckpt["optimizer_state"])
                    scheduler.load_state_dict(loaded_ckpt["scheduler_state"])
                else:
                    print(f"Phase change detected, building fresh optimizer at lr_mult={lr_multiplier}, bs={new_batch_size}.")
                loaded_ckpt = None  # Only attempt restore once

        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, args.device, scaler)
        val_mae = validate_one_epoch(model, valid_loader, args.device, evaluator)
        scheduler.step(val_mae)

        epoch_minutes = (time.time() - start_time) / 60
        all_lrs = get_all_lrs(optimizer)

        print(f"Epoch {epoch}: Train Loss={train_loss:.5f}, Val MAE={val_mae:.5f}, Time={epoch_minutes:.2f} min, LR={all_lrs[0]:.2e}")

        with open(args.epoch_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch, f"{train_loss:.5f}", f"{val_mae:.5f}", f"{epoch_minutes:.2f}"]
                + [f"{lr:.2e}" for lr in all_lrs]
            )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "chemgraphormer_best_model.pt"))

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_mae": best_val_mae,
            "train_cfg": {"batch_size": current_batch_size, "lr_multiplier": current_lr_multiplier},
        }, os.path.join(args.ckpt_dir, "last_checkpoint.pt"))

    print(f"Training complete. Best Val MAE: {best_val_mae:.5f}")


if __name__ == "__main__":
    args = parse_args()
    train_with_meanp_gate0_reactive(args)