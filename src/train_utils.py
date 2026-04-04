"""
src/train_utils.py
==================
Training, evaluation, and persistence utilities for BRICSLiquiditySNN.

All functions are model-agnostic where possible — they accept any
nn.Module that outputs raw logits from input tensors.

Usage
-----
from src.train_utils import train_one_epoch, evaluate, save_model, load_model
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from typing import Tuple, Dict, Optional, List

from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score, roc_curve,
    confusion_matrix, precision_score, recall_score
)
from spikingjelly.activation_based import functional


# ─────────────────────────────────────────────
# SECTION 1 — Training
# ─────────────────────────────────────────────

def train_one_epoch(model     : nn.Module,
                    loader    : torch.utils.data.DataLoader,
                    criterion : nn.Module,
                    optimizer : torch.optim.Optimizer,
                    device    : torch.device,
                    clip_norm : float = 1.0) -> float:
    """
    Run one full training epoch over a DataLoader.

    Resets LIF membrane voltages before each batch via
    functional.reset_net(). This is critical for SNN correctness —
    voltage must not carry over between independent sequences.

    Parameters
    ----------
    model     : nn.Module — SNN or any model outputting raw logits
    loader    : DataLoader providing (X_batch, y_batch) tuples
    criterion : loss function (use BCEWithLogitsLoss for SNN)
    optimizer : torch.optim.Optimizer
    device    : torch.device
    clip_norm : float — gradient clipping max norm (default 1.0)
                Prevents exploding gradients with surrogate gradients.

    Returns
    -------
    float — mean loss per sample across all batches
    """
    model.train()
    total_loss = 0.0
    total_n    = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        functional.reset_net(model)   # ← CRITICAL for SNN

        logits = model(X_batch).squeeze()
        loss   = criterion(logits, y_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        total_n    += len(y_batch)

    return total_loss / total_n


# ─────────────────────────────────────────────
# SECTION 2 — Evaluation
# ─────────────────────────────────────────────

def evaluate(model     : nn.Module,
             loader    : torch.utils.data.DataLoader,
             criterion : nn.Module,
             device    : torch.device) -> Dict:
    """
    Evaluate model on a DataLoader and return full metrics dict.

    Uses Youden's J statistic (argmax of TPR - FPR) to find the
    optimal decision threshold rather than hardcoding 0.5.
    This is important for imbalanced datasets where the model's
    output probabilities may not be centred at 0.5.

    Parameters
    ----------
    model     : nn.Module
    loader    : DataLoader
    criterion : loss function
    device    : torch.device

    Returns
    -------
    dict with keys:
        loss, accuracy, precision, recall, f1, auc,
        optimal_threshold, confusion_matrix,
        all_probs, all_labels, all_preds
    """
    model.eval()
    all_probs, all_labels = [], []
    total_loss, total_n   = 0.0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            functional.reset_net(model)
            logits = model(X_batch).squeeze()
            loss   = criterion(logits, y_batch)
            probs  = torch.sigmoid(logits)

            total_loss += loss.item() * len(y_batch)
            total_n    += len(y_batch)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(y_batch.cpu().numpy().tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Optimal threshold via Youden's J
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    opt_idx   = int(np.argmax(tpr - fpr))
    opt_thresh = float(thresholds[opt_idx])

    all_preds = (all_probs >= opt_thresh).astype(int)

    return {
        "loss"             : round(total_loss / total_n, 5),
        "accuracy"         : round(accuracy_score(all_labels, all_preds), 4),
        "precision"        : round(precision_score(all_labels, all_preds,
                                                   zero_division=0), 4),
        "recall"           : round(recall_score(all_labels, all_preds,
                                                zero_division=0), 4),
        "f1"               : round(f1_score(all_labels, all_preds,
                                            zero_division=0), 4),
        "auc"              : round(roc_auc_score(all_labels, all_probs), 4),
        "optimal_threshold": round(opt_thresh, 4),
        "confusion_matrix" : confusion_matrix(all_labels, all_preds).tolist(),
        "all_probs"        : all_probs,
        "all_labels"       : all_labels,
        "all_preds"        : all_preds,
    }


def get_optimal_threshold(all_labels : np.ndarray,
                           all_probs  : np.ndarray) -> float:
    """
    Find optimal classification threshold via Youden's J statistic.

    Youden's J = TPR - FPR, maximised at the threshold that best
    separates positive and negative classes on the ROC curve.

    Parameters
    ----------
    all_labels : np.ndarray — true binary labels
    all_probs  : np.ndarray — predicted probabilities

    Returns
    -------
    float — optimal threshold in [0, 1]
    """
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    return float(thresholds[int(np.argmax(tpr - fpr))])


# ─────────────────────────────────────────────
# SECTION 3 — Model Persistence
# ─────────────────────────────────────────────

def save_model(model      : nn.Module,
               config     : dict,
               path       : str,
               extra_meta : Optional[dict] = None) -> None:
    """
    Save model weights, config, and optional metadata to a .pt file.

    The saved file contains everything needed to reload the model
    without access to the original training notebook:
    - model_state: weights and biases
    - config: architecture hyperparameters
    - meta: performance metrics, feature columns, thresholds

    Parameters
    ----------
    model      : trained nn.Module
    config     : dict — architecture parameters (n_features, hidden1, etc.)
    path       : str — save path (e.g. 'outputs/snn_best.pt')
    extra_meta : dict, optional — any additional info to store
                 (e.g. val_auc, feature_cols, optimal_threshold)

    Examples
    --------
    >>> save_model(model, config, "outputs/snn_best.pt",
    ...            extra_meta={"val_auc": 0.555, "val_f1": 0.563})
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                exist_ok=True)

    payload = {
        "model_state": model.state_dict(),
        "config"     : config,
    }
    if extra_meta:
        payload["meta"] = extra_meta

    torch.save(payload, path)
    size_kb = os.path.getsize(path) / 1024
    print(f"✅ Model saved: {path}  ({size_kb:.1f} KB)")


def load_model(path        : str,
               model_class ,
               device      : torch.device) -> Tuple[nn.Module, dict, dict]:
    """
    Load a model from a .pt checkpoint file.

    Reconstructs the model architecture from the saved config,
    loads weights, and sets the model to eval mode.

    Parameters
    ----------
    path        : str — path to .pt file saved by save_model()
    model_class : class — the nn.Module class to instantiate
                  (e.g. BRICSLiquiditySNN)
    device      : torch.device

    Returns
    -------
    model  : nn.Module — loaded model in eval mode on device
    config : dict — architecture config used to build the model
    meta   : dict — any extra metadata (empty dict if not saved)

    Examples
    --------
    >>> model, config, meta = load_model(
    ...     "outputs/snn_best.pt", BRICSLiquiditySNN, device)
    >>> print(meta["val_auc"])
    0.555
    """
    checkpoint = torch.load(path, map_location=device)
    config     = checkpoint["config"]
    meta       = checkpoint.get("meta", {})

    model = model_class(**{
        k: config[k] for k in
        ["n_features", "hidden1", "hidden2", "tau", "v_threshold"]
        if k in config
    }).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"✅ Model loaded: {path}")
    print(f"   Config : {config}")
    if meta:
        print(f"   Meta   : { {k:v for k,v in meta.items() if k != 'feature_cols'} }")

    return model, config, meta


def save_config_json(config : dict, path : str) -> None:
    """
    Save a configuration dictionary as a formatted JSON file.

    Used to export model config for FastAPI to load at startup
    without importing any PyTorch or SNN dependencies.

    Parameters
    ----------
    config : dict — must be JSON-serialisable (no tensors)
    path   : str  — output path (e.g. 'outputs/snn_config.json')
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Config saved: {path}")


def print_metrics(metrics : dict, label : str = "Metrics") -> None:
    """
    Pretty-print a metrics dictionary from evaluate().

    Parameters
    ----------
    metrics : dict — output of evaluate()
    label   : str  — header label for the printout
    """
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    skip = {"all_probs", "all_labels", "all_preds", "confusion_matrix"}
    for k, v in metrics.items():
        if k not in skip:
            print(f"  {k:<22}: {v}")

    cm = metrics.get("confusion_matrix")
    if cm:
        print(f"\n  Confusion Matrix:")
        print(f"                 Pred DOWN  Pred UP")
        print(f"  Actual DOWN  :    {cm[0][0]:>5}     {cm[0][1]:>5}")
        print(f"  Actual UP    :    {cm[1][0]:>5}     {cm[1][1]:>5}")
    print(f"{'='*50}")