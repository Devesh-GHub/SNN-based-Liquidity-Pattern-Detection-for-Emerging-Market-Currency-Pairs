"""
src/lstm_model.py
=================
BRICSLiquidityLSTM — LSTM baseline for INR/BRL FX direction prediction.

Used as the comparison baseline against BRICSLiquiditySNN in the BRICS
settlement project. Deliberately kept simple — one LSTM layer, one
dropout, one linear output — to serve as an honest, fair benchmark.

Design principle
----------------
Every architectural and training choice mirrors the SNN where possible:
- Same input format: (batch, timesteps, n_features)
- Same lookback window: T=10
- Same feature set and scaler
- Same loss function: BCEWithLogitsLoss with pos_weight
- Same optimiser: Adam(lr=0.001, weight_decay=1e-4)
- Same evaluation: Youden J optimal threshold

The only difference is the temporal processing mechanism:
- LSTM: continuous-valued gated recurrent cell (dense computation)
- SNN:  binary spike-based LIF neurons (sparse computation)

This isolation ensures that any performance difference is attributable
to the architecture, not to data or training asymmetries.

Key difference from SNN during training
----------------------------------------
LSTM does NOT require functional.reset_net() between batches.
PyTorch initialises (h_0, c_0) to zeros automatically for each
new forward pass when not explicitly provided — the hidden state
does not persist across batches by default. The SNN requires
explicit reset because LIF membrane voltage is stored as a module
attribute that persists until cleared.

References
----------
- Hochreiter & Schmidhuber (1997): "Long Short-Term Memory." Neural Computation.
- Greff et al. (2017): "LSTM: A Search Space Odyssey." IEEE TNNLS.
  (Justification for forget gate bias initialisation to 1.0)
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from typing import Optional, Tuple


class BRICSLiquidityLSTM(nn.Module):
    """
    LSTM baseline for INR/BRL settlement direction prediction.

    Predicts whether the synthetic INR/BRL cross-rate will be higher
    (UP=1) or lower (DOWN=0) on the next trading day, given a lookback
    window of normalised daily features.

    Architecture
    ------------
    Input: (batch, timesteps, n_features)
        ↓
    LSTM(input_size=n_features, hidden_size=hidden_size,
         num_layers=1, batch_first=True)
        ↓  outputs (batch, T, hidden_size) — one state per timestep
    Take last timestep: lstm_out[:, -1, :]  → (batch, hidden_size)
        ↓
    Dropout(dropout)
        ↓
    Linear(hidden_size → 1)
        ↓
    Raw logit → BCEWithLogitsLoss during training
              → sigmoid(logit) > optimal_threshold for inference

    Design decisions
    ----------------
    - hidden_size=32: matches SNN's hidden2 output dimension for
      parameter-comparable final readout layer.
    - num_layers=1: single layer prevents overfitting on ~720 training
      sequences. Dataset too small for stacked LSTM.
    - Last timestep only (lstm_out[:, -1, :]): standard for sequence
      classification. Preserves architectural distinction from SNN's
      spike accumulator (mean over all timesteps).
    - Forget gate bias = 1.0: standard LSTM trick (Greff et al., 2017)
      that prevents catastrophic forgetting at initialisation.
    - No attention, no bidirectional: this is a baseline, not SOTA.
      Adding complexity would make the comparison unfair.

    Parameters
    ----------
    n_features  : int   — number of input features per timestep. Default 9.
    hidden_size : int   — LSTM hidden state dimension. Default 32.
    dropout     : float — dropout rate applied after LSTM. Default 0.2.
    num_layers  : int   — number of stacked LSTM layers. Default 1.

    Input / Output
    --------------
    forward(x) input  : torch.Tensor, shape (batch, timesteps, n_features)
    forward(x) output : torch.Tensor, shape (batch, 1) — raw logit
                        Apply sigmoid for probability, compare to
                        optimal_threshold for binary prediction.

    Comparison with SNN
    -------------------
    | Aspect          | LSTM              | SNN                    |
    |-----------------|-------------------|------------------------|
    | Temporal memory | Gated cell state  | Membrane voltage decay |
    | Computation     | Dense (always on) | Sparse (spike-gated)   |
    | Parameters      | ~5,537            | ~2,945                 |
    | Output type     | Last h_t          | Spike rate (mean/T)    |
    | Reset needed    | No                | Yes (reset_net)        |

    Examples
    --------
    >>> model = BRICSLiquidityLSTM(n_features=9)
    >>> x     = torch.randn(32, 10, 9)    # batch=32, T=10, F=9
    >>> logit = model(x)                   # shape: (32, 1)
    >>> prob  = torch.sigmoid(logit)       # shape: (32, 1) in [0,1]
    >>> pred  = (prob >= 0.53).int()       # binary prediction
    """

    def __init__(self,
                 n_features : int   = 9,
                 hidden_size: int   = 32,
                 dropout    : float = 0.2,
                 num_layers : int   = 1):
        super().__init__()

        self.n_features  = n_features
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout_rate = dropout

        # ── Layers ────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,   # input: (batch, T, features)
            dropout     = 0.0,    # inter-layer dropout (only >1 layer)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        """
        Initialise LSTM and linear layer weights.

        LSTM weights:
        - Input-hidden (weight_ih): Xavier uniform — standard for
          feed-forward connections.
        - Hidden-hidden (weight_hh): Orthogonal — standard for
          recurrent connections, prevents vanishing gradients.
        - Bias: zeros, except forget gate bias = 1.0 (Greff et al., 2017).

        Linear weights: Xavier uniform, bias = 0.
        """
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                # Forget gate bias = 1.0
                # LSTM bias vector is [i, f, g, o] gates, each size hidden
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.

        Note: No reset_net() needed. PyTorch initialises (h_0, c_0)
        to zeros by default on each forward call when not explicitly
        provided — hidden state does not persist between batches.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, timesteps, n_features)
            Normalised feature sequences from DataLoader.
            Values should be in approximately [0, 1] after MinMaxScaler.

        Returns
        -------
        torch.Tensor, shape (batch, 1) — raw logit.
            Use BCEWithLogitsLoss (not BCELoss) as training criterion.
            Apply sigmoid for probability; compare to optimal_threshold.
        """
        # lstm_out : (batch, T, hidden_size)
        # _        : (h_n, c_n) — not used for classification
        lstm_out, _ = self.lstm(x)

        # Last timestep: (batch, hidden_size)
        # Captures the LSTM's summary of the full T-step sequence
        out = lstm_out[:, -1, :]

        out = self.dropout(out)
        out = self.fc(out)       # (batch, 1) raw logit
        return out

    # ── Utility methods ───────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)

    def flops_per_inference(self, lookback: int = 10) -> int:
        """
        Estimate FLOPs per inference for one sequence.

        LSTM FLOPs per timestep:
            4 gates × 2 ops × (hidden×features + hidden×hidden)
            Factor 2: one multiply + one add per connection.

        Parameters
        ----------
        lookback : int — sequence length (default 10)

        Returns
        -------
        int — approximate FLOPs for one sequence
        """
        flops_per_step = (
            4 * 2 * self.hidden_size * self.n_features +
            4 * 2 * self.hidden_size * self.hidden_size
        )
        return flops_per_step * lookback

    def estimate_energy(self,
                        lookback  : int   = 10,
                        e_op_pj   : float = 0.9) -> dict:
        """
        Estimate inference energy in picojoules.

        Uses the same 0.9 pJ/operation benchmark as the SNN estimate
        (Blouw et al., 2019) for a fair comparison. Note that FLOPs
        and SynOps are not directly equivalent — this is an indicative
        comparison, not an exact measurement.

        Parameters
        ----------
        lookback : int   — sequence length
        e_op_pj  : float — energy per operation in pJ

        Returns
        -------
        dict with flops, energy_pj
        """
        flops     = self.flops_per_inference(lookback)
        energy_pj = flops * e_op_pj / 1000   # convert to pJ scale
        return {
            "flops"    : flops,
            "energy_pj": round(energy_pj, 3),
        }


def load_lstm(path        : str,
              device      : torch.device,
              config      : Optional[dict] = None) -> Tuple[
                  "BRICSLiquidityLSTM", dict]:
    """
    Load a BRICSLiquidityLSTM from a checkpoint file.

    Parameters
    ----------
    path   : str — path to .pt checkpoint or .pth weights file
    device : torch.device
    config : dict, optional — if None, loads config from checkpoint

    Returns
    -------
    model  : BRICSLiquidityLSTM in eval mode
    meta   : dict — metadata from checkpoint (empty if .pth only)
    """
    checkpoint = torch.load(path, map_location=device)

    # Handle both full checkpoint (.pt) and weights-only (.pth)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state  = checkpoint["model_state"]
        cfg    = config or checkpoint.get("config", {})
        meta   = {k: v for k, v in checkpoint.items()
                  if k not in ("model_state", "config")}
    else:
        state = checkpoint
        cfg   = config or {}
        meta  = {}

    model = BRICSLiquidityLSTM(
        n_features  = cfg.get("n_features",  9),
        hidden_size = cfg.get("hidden_size", 32),
        dropout     = cfg.get("dropout",     0.2),
        num_layers  = cfg.get("num_layers",  1),
    ).to(device)

    model.load_state_dict(state)
    model.eval()
    print(f"✅ LSTM loaded: {path}  ({model.count_parameters()} params)")
    return model, meta

