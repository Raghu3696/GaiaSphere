"""
GaiaSphere — Evaluation Metrics
================================
Task-appropriate metrics for each prediction head.

  Flood / Wildfire:  IoU, F1, Precision, Recall, AUC-ROC
  Crop Health:       MAE, RMSE, R², Pearson correlation with NDVI

Author: GaiaSphere Project
"""

import torch
import numpy as np
from typing import Dict, Any


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection / (union + 1e-8)).item()


def f1_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (pred > threshold).float()
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return (2 * precision * recall / (precision + recall + 1e-8)).item()


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / (ss_tot + 1e-8)).item()


def pearson_correlation(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_centered = pred - pred.mean()
    tgt_centered = target - target.mean()
    numerator = (pred_centered * tgt_centered).sum()
    denominator = (pred_centered.norm() * tgt_centered.norm()) + 1e-8
    return (numerator / denominator).item()


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    task: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:

    pred = pred.float().view(-1)
    target = target.float().view(-1)

    if task in ('flood', 'wildfire'):
        return {
            'iou': iou_score(pred, target, threshold),
            'f1': f1_score(pred, target, threshold),
            'mae': (pred - target).abs().mean().item(),
        }
    elif task == 'crop':
        mae = (pred - target).abs().mean().item()
        rmse = ((pred - target) ** 2).mean().sqrt().item()
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2_score(pred, target),
            'pearson': pearson_correlation(pred, target),
        }
    else:
        return {}
