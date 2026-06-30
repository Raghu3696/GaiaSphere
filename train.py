"""
GaiaSphere — Training Entry Point
===================================
Train FloodNet, CropHealthNet, or WildfireNet using the GaiaNet backbone.

Usage:
    python train.py --config configs/flood.yaml --task flood
    python train.py --config configs/crop.yaml  --task crop
    python train.py --config configs/wildfire.yaml --task wildfire

Author: GaiaSphere Project
"""

import os
import yaml
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from src.models.gaianet import GaiaNet
from src.utils.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger("GaiaSphere")


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class FloodLoss(nn.Module):
    """Binary cross-entropy + Dice loss for flood segmentation."""

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss()

    def dice_loss(self, pred, target, smooth=1.0):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice


class CropLoss(nn.Module):
    """Huber loss for NDVI regression (robust to outliers)."""

    def __init__(self):
        super().__init__()
        self.loss = nn.HuberLoss(delta=0.1)

    def forward(self, pred, target):
        return self.loss(pred, target)


class WildfireLoss(nn.Module):
    """Focal loss for fire detection (handles severe class imbalance)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


LOSS_MAP = {
    'flood': FloodLoss,
    'crop': CropLoss,
    'wildfire': WildfireLoss,
}


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, scaler, device, task):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader):
        images  = batch['images'].to(device)
        weather = batch.get('weather')
        if weather is not None:
            weather = weather.to(device)
        dem    = batch['dem'].to(device)
        coords = batch.get('coords')
        if coords is not None:
            coords = coords.to(device)
        target = batch['target'].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images, weather, dem, coords)
            pred = outputs.get(task) or outputs.get('fire_prob')
            loss = criterion(pred, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if step % 50 == 0:
            log.info(f"  Step {step:04d} | loss: {loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, task):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for batch in loader:
        images  = batch['images'].to(device)
        weather = batch.get('weather')
        if weather is not None:
            weather = weather.to(device)
        dem    = batch['dem'].to(device)
        coords = batch.get('coords')
        if coords is not None:
            coords = coords.to(device)
        target = batch['target'].to(device)

        outputs = model(images, weather, dem, coords)
        pred = outputs.get(task) or outputs.get('fire_prob')
        loss = criterion(pred, target)
        total_loss += loss.item()

        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics     = compute_metrics(all_preds, all_targets, task)

    return total_loss / max(len(loader), 1), metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GaiaSphere Training")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config")
    parser.add_argument('--task', type=str, required=True, choices=['flood', 'crop', 'wildfire'])
    parser.add_argument('--resume', type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    task = args.task
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device} | Task: {task}")

    # Output directory
    out_dir = Path(cfg.get('output_dir', f'outputs/{task}'))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────────────
    model = GaiaNet(
        img_size=cfg.get('img_size', 256),
        patch_size=cfg.get('patch_size', 16),
        in_channels=cfg.get('in_channels', 13),
        vit_embed_dim=cfg.get('vit_embed_dim', 768),
        vit_depth=cfg.get('vit_depth', 12),
        vit_heads=cfg.get('vit_heads', 12),
        st_depth=cfg.get('st_depth', 6),
        weather_dim=cfg.get('weather_dim', 16),
        gnn_hidden=cfg.get('gnn_hidden', 256),
        gnn_layers=cfg.get('gnn_layers', 3),
        task=task,
        dropout=cfg.get('dropout', 0.1),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {total_params / 1e6:.1f}M")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        log.info(f"Resumed from {args.resume}")

    # ── Data ───────────────────────────────────────────────────────────────
    # NOTE: Replace with your actual Dataset class
    # from src.data.dataset import GaiaDataset
    # train_set = GaiaDataset(cfg['train_dir'], task=task, split='train')
    # val_set   = GaiaDataset(cfg['val_dir'],   task=task, split='val')
    # train_loader = DataLoader(train_set, batch_size=cfg.get('batch_size', 4), shuffle=True,  num_workers=4, pin_memory=True)
    # val_loader   = DataLoader(val_set,   batch_size=cfg.get('batch_size', 4), shuffle=False, num_workers=4, pin_memory=True)
    log.info("NOTE: Using placeholder data. Plug in GaiaDataset for real training.")

    # ── Optimizer & Scheduler ──────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.get('lr', 1e-4),
        weight_decay=cfg.get('weight_decay', 1e-2),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.get('epochs', 50),
        eta_min=cfg.get('min_lr', 1e-6),
    )
    scaler = GradScaler()
    criterion = LOSS_MAP[task]()

    # ── Training ───────────────────────────────────────────────────────────
    best_metric = -float('inf')
    epochs = cfg.get('epochs', 50)

    log.info(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        log.info(f"\n── Epoch {epoch}/{epochs} ──────────────────────────────")

        # Uncomment when dataset is ready:
        # train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device, task)
        # val_loss, metrics = eval_epoch(model, val_loader, criterion, device, task)
        # log.info(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        # log.info(f"Metrics: {metrics}")

        # Primary metric (F1 for segmentation, R² for regression)
        # primary = metrics.get('f1', metrics.get('r2', 0.0))
        # if primary > best_metric:
        #     best_metric = primary
        #     torch.save({'epoch': epoch, 'model_state': model.state_dict()}, out_dir / f'{task}_best.pth')
        #     log.info(f"  ★ New best: {primary:.4f} → saved checkpoint")

        scheduler.step()

    log.info(f"\nTraining complete. Best: {best_metric:.4f}")
    log.info(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main()
