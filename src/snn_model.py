"""
src/snn_model.py
================
BRICSLiquiditySNN — Spiking Neural Network for INR/BRL FX direction prediction.

This module contains the full SNN architecture used in the BRICS settlement
project. It is imported by training notebooks, the FastAPI prediction endpoint,
and the dashboard backend.

Architecture summary
--------------------
Input (batch, T, F) → FC1 → BN1 → LIF1 → FC2 → BN2 → LIF2 → FC3 → logit

Spike accumulation readout: spikes are summed across T timesteps and divided
by T to produce a spike rate vector, which is then mapped to a binary
settlement direction probability via FC3.

References
----------
- Fang et al. (2021): "Incorporating Learnable Membrane Time Constants to
  Enhance Learning of Spiking Neural Networks." ICCV 2021.
- Blouw et al. (2019): "Benchmarking Keyword Spotting Efficiency on
  Neuromorphic Hardware." ACM CF 2019. (Energy benchmark: 0.9 pJ/SynOp)
- Cont (2001): "Empirical properties of asset returns: stylized facts."
  Quantitative Finance. (Justification for bursty spike encoding)
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate
import numpy as np
from typing import Optional


class BRICSLiquiditySNN(nn.Module):
    """
    Spiking Neural Network for INR/BRL settlement direction prediction.

    Predicts whether the synthetic INR/BRL cross-rate will be higher
    (UP=1) or lower (DOWN=0) on the next trading day, given a lookback
    window of normalised daily features.

    Architecture
    ------------
    Input: (batch, timesteps, n_features)
        ↓
    FC1(n_features → hidden1) + BatchNorm1d + LIF1
        ↓  [spike output: 0 or 1 per neuron]
    FC2(hidden1 → hidden2) + BatchNorm1d + LIF2
        ↓  [spike output: 0 or 1 per neuron]
    Spike accumulator (sum over T, divide by T → spike rate)
        ↓
    FC3(hidden2 → 1)
        ↓
    Raw logit  →  BCEWithLogitsLoss during training
               →  sigmoid(logit) > optimal_threshold for inference

    Design decisions
    ----------------
    - BatchNorm before LIF: normalises fc output to consistent range
      so LIF threshold is meaningful regardless of input scale.
      BatchNorm BETWEEN two LIF layers would disrupt spike dynamics.
    - Xavier initialisation: prevents fc outputs from being too small
      to cross the LIF threshold at initialisation.
    - ATan surrogate gradient: stable, widely cited (Fang et al., 2021).
    - detach_reset=True: prevents gradient flow through the reset
      operation, improving training stability.
    - Spike rate readout: rate coding over T timesteps provides a
      continuous [0,1] signal from binary spikes.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep. Default 9.
    hidden1 : int
        Number of neurons in first hidden layer. Default 64.
    hidden2 : int
        Number of neurons in second hidden layer. Default 32.
    tau : float
        LIF membrane time constant. Controls how fast voltage decays.
        tau=2.0: ~50% decay per step (recent inputs dominate).
        tau=5.0: ~20% decay per step (longer memory).
        Default 2.0 — appropriate for 10-day lookback window.
    v_threshold : float
        LIF firing threshold. Neurons fire when membrane voltage
        exceeds this value. Default 0.1 — calibrated to produce
        ~15-50% spike rate on normalised INR/BRL features.

    Input / Output
    --------------
    forward(x) input  : torch.Tensor, shape (batch, timesteps, n_features)
    forward(x) output : torch.Tensor, shape (batch, 1) — raw logit
                        Apply sigmoid for probability, then compare to
                        optimal_threshold for binary prediction.

    Examples
    --------
    >>> model = BRICSLiquiditySNN(n_features=9, tau=2.0, v_threshold=0.1)
    >>> x = torch.randn(32, 10, 9)       # batch=32, T=10, F=9
    >>> logit = model(x)                  # shape: (32, 1)
    >>> prob  = torch.sigmoid(logit)      # shape: (32, 1), values in [0,1]
    >>> pred  = (prob >= 0.52).int()      # binary prediction
    """

    def __init__(self,
                 n_features : int   = 9,
                 hidden1    : int   = 64,
                 hidden2    : int   = 32,
                 tau        : float = 2.0,
                 v_threshold: float = 0.1):
        super().__init__()

        self.n_features  = n_features
        self.hidden1     = hidden1
        self.hidden2     = hidden2
        self.tau         = tau
        self.v_threshold = v_threshold

        # ── Layer definitions ─────────────────────────────────────────
        self.fc1  = nn.Linear(n_features, hidden1, bias=True)
        self.bn1  = nn.BatchNorm1d(hidden1)
        self.lif1 = neuron.LIFNode(
            tau               = tau,
            v_threshold       = v_threshold,
            surrogate_function= surrogate.ATan(),
            detach_reset      = True,
        )

        self.fc2  = nn.Linear(hidden1, hidden2, bias=True)
        self.bn2  = nn.BatchNorm1d(hidden2)
        self.lif2 = neuron.LIFNode(
            tau               = tau,
            v_threshold       = v_threshold,
            surrogate_function= surrogate.ATan(),
            detach_reset      = True,
        )

        self.fc3 = nn.Linear(hidden2, 1, bias=True)

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────
    def _init_weights(self):
        """
        Xavier uniform initialisation for all linear layers.

        Xavier initialisation scales weights by sqrt(2 / (fan_in + fan_out)),
        ensuring fc outputs span a range that LIF neurons can threshold on.
        Without this, fc outputs are too small to trigger spikes at init.
        """
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    # ── Forward pass ──────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process a batch of sequences through the SNN.

        Each timestep is processed independently through fc+bn+lif.
        Spike outputs are accumulated across all T timesteps.
        The final spike rate (accumulated / T) is fed to fc3.

        CRITICAL: functional.reset_net(self) must be called at the
        start of each forward pass to zero LIF membrane voltages.
        Without this, voltage from the previous batch contaminates
        the current batch — the #1 SpikingJelly training bug.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, timesteps, n_features)
            Normalised feature sequences. Values should be in [0, 1]
            (MinMaxScaler with clip=True applied upstream).

        Returns
        -------
        torch.Tensor, shape (batch, 1)
            Raw logit. Apply torch.sigmoid() for probability output.
            Use BCEWithLogitsLoss (not BCELoss) as the training criterion.
        """
        # Reset LIF membrane voltages — MUST be called every forward pass
        functional.reset_net(self)

        T = x.shape[1]

        # Accumulates spike outputs from lif2 across all timesteps
        spike_accumulator = torch.zeros(
            x.shape[0], self.hidden2,
            device=x.device, dtype=x.dtype
        )

        for t in range(T):
            x_t = x[:, t, :]                    # (batch, n_features)

            out = self.fc1(x_t)                 # (batch, hidden1)
            out = self.bn1(out)                 # normalise for LIF
            out = self.lif1(out)                # spike: 0 or 1 per neuron

            out = self.fc2(out)                 # (batch, hidden2)
            out = self.bn2(out)
            out = self.lif2(out)                # spike: 0 or 1 per neuron

            spike_accumulator = spike_accumulator + out

        # Spike rate: fraction of timesteps each neuron fired
        spike_rate = spike_accumulator / T      # (batch, hidden2) in [0,1]

        return self.fc3(spike_rate)             # (batch, 1) raw logit

    # ── Utility methods ───────────────────────────────────────────────
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_spike_rate(self,
                       loader     : torch.utils.data.DataLoader,
                       device     : torch.device,
                       n_batches  : int = 10) -> float:
        """
        Measure mean spike rate across n_batches of a DataLoader.

        Spike rate = fraction of (neuron × timestep) pairs that fire,
        averaged over both LIF layers.

        Parameters
        ----------
        loader    : DataLoader providing (X_batch, y_batch) tuples
        device    : torch.device
        n_batches : int — number of batches to sample

        Returns
        -------
        float — mean spike rate across both layers, in [0, 1]
        """
        self.eval()
        rates = []

        with torch.no_grad():
            for i, (X_batch, _) in enumerate(loader):
                if i >= n_batches:
                    break
                X_batch = X_batch.to(device)
                functional.reset_net(self)
                T = X_batch.shape[1]

                acc1 = torch.zeros(X_batch.shape[0], self.hidden1,
                                   device=device)
                acc2 = torch.zeros(X_batch.shape[0], self.hidden2,
                                   device=device)

                for t in range(T):
                    x_t  = X_batch[:, t, :]
                    s1   = self.lif1(self.bn1(self.fc1(x_t)))
                    s2   = self.lif2(self.bn2(self.fc2(s1)))
                    acc1 += s1
                    acc2 += s2

                r1 = (acc1 / T).mean().item()
                r2 = (acc2 / T).mean().item()
                rates.append((r1 + r2) / 2)

        return float(np.mean(rates))

    def estimate_energy(self,
                        spike_rate   : Optional[float] = None,
                        e_syn_pj     : float = 0.9) -> dict:
        """
        Estimate inference energy based on synaptic operations (SynOps).

        Uses the benchmark of 0.9 pJ per synaptic operation on 45nm CMOS
        hardware (Blouw et al., 2019). This is a theoretical lower bound —
        actual energy depends on hardware implementation.

        Parameters
        ----------
        spike_rate : float, optional
            Mean spike rate from get_spike_rate(). If None, uses 0.478
            (measured baseline value).
        e_syn_pj : float
            Energy per synaptic operation in picojoules. Default 0.9.

        Returns
        -------
        dict with keys:
            snn_synops, dense_synops, snn_energy_pj,
            dense_energy_pj, efficiency_pct
        """
        if spike_rate is None:
            spike_rate = 0.478   # measured baseline

        lookback = 10   # T timesteps

        # SNN: only fired neurons propagate
        synops_l1 = spike_rate * self.hidden1 * self.hidden2 * lookback
        synops_l2 = spike_rate * self.hidden2 * 1            * lookback
        snn_synops = synops_l1 + synops_l2

        # Dense equivalent: all neurons fire every timestep
        dense_synops = (self.hidden1 * self.hidden2 * lookback +
                        self.hidden2 * 1            * lookback)

        snn_energy   = snn_synops   * e_syn_pj
        dense_energy = dense_synops * e_syn_pj
        efficiency   = (1 - snn_energy / dense_energy) * 100

        return {
            "snn_synops"    : round(snn_synops,   1),
            "dense_synops"  : round(dense_synops, 1),
            "snn_energy_pj" : round(snn_energy,   3),
            "dense_energy_pj": round(dense_energy, 3),
            "efficiency_pct": round(efficiency,   1),
        }