"""
api/predictor.py
================
SNNPredictor — loads BRICSLiquiditySNN once and serves predictions.

Singleton pattern: the module-level `predictor` object is instantiated
once when the module is first imported. FastAPI's lifespan imports this
module at startup, so the model is loaded exactly once and reused for
every subsequent request.

Reloading the model per-request would add ~200-500ms latency and
defeat the purpose of a low-latency settlement signal API.

Usage
-----
from api.predictor import predictor

result = predictor.predict([16.2, 16.3, ..., 16.5])   # 10 prices
print(result["recommendation"])   # "DIRECT" or "USD_FALLBACK"
"""

import torch
import torch.nn as nn
import numpy as np
import joblib
import json
import os
import sys
import time
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────
# Works whether called as `python api/predictor.py` or imported by FastAPI
_HERE        = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
# sys.path.insert(0, _PROJECT_ROOT)

# Support both local (outputs/) and Render deployment
# Render serves from repo root, so outputs/ is at /opt/render/project/src/outputs/
def _find_outputs_dir() -> str:
    candidates = [
        os.path.join(_PROJECT_ROOT, "outputs"),  # local dev
        os.path.join(os.getcwd(), "outputs"),    # Render / uvicorn cwd
        "outputs",                                # relative fallback
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    # Create it if missing (Render cold start)
    os.makedirs(candidates[0], exist_ok=True)
    return candidates[0]


from src.snn_model import BRICSLiquiditySNN
from spikingjelly.activation_based import functional



# ── File paths ─────────────────────────────────────────────────────────

OUTPUTS_DIR = _find_outputs_dir()
MODEL_PATH  = os.path.join(OUTPUTS_DIR, "snn_model.pth")
CONFIG_PATH = os.path.join(OUTPUTS_DIR, "snn_config.json")
SCALER_PATH = os.path.join(OUTPUTS_DIR, "scaler.pkl")

# ── Decision thresholds ────────────────────────────────────────────────
_PROB_THRESHOLD_DEFAULT = 0.52   # fallback only — overridden by config optimal_threshold at load
RATE_THRESHOLD = 0.10   # minimum spike rate for DIRECT routing
SPIKE_THRESHOLD = 0.003  # return threshold for spike encoding




class SNNPredictor:
    """
    Singleton predictor for BRICSLiquiditySNN inference.

    Loads the trained model, scaler, and config once at instantiation.
    Exposes preprocess() and predict() for the FastAPI endpoint.

    All inference runs on CPU — the model is small enough (~3K params)
    that CPU inference is faster than GPU transfer overhead.

    Attributes
    ----------
    model        : BRICSLiquiditySNN in eval mode
    scaler       : fitted MinMaxScaler (same as training)
    config       : dict from snn_config.json
    feature_cols : list of feature column names (matches training order)
    lookback     : int — sequence length (10)
    n_features   : int — number of features (9)
    loaded       : bool — True if all components loaded successfully
    load_time_s  : float — seconds taken to load
    """

    def __init__(self):
        self.model          = None
        self.scaler         = None
        self.config         = None
        self.feature_cols   = None
        self.lookback       = 10
        self.n_features     = 9
        self.loaded         = False
        self.load_time_s    = 0.0
        self._device        = torch.device("cpu")
        self.prob_threshold = _PROB_THRESHOLD_DEFAULT   # overwritten from config in _load()

        self._load()

    def _load(self) -> None:
        """
        Load model, scaler, and config from outputs/ directory.
        Sets self.loaded = True on success, False on any failure.
        """
        t0 = time.time()

        # ── Check all files exist ─────────────────────────────────────
        missing = [p for p in [MODEL_PATH, CONFIG_PATH, SCALER_PATH]
                   if not os.path.exists(p)]
        if missing:
            print(f"❌ Missing files: {missing}")
            print(f"   Run training notebooks first.")
            self.loaded = False
            return

        try:
            # ── Load config ───────────────────────────────────────────
            with open(CONFIG_PATH) as f:
                self.config = json.load(f)

            self.feature_cols   = self.config.get("feature_cols", [])
            self.n_features     = self.config["n_features"]
            self.lookback       = self.config.get("lookback", 10)
            self.prob_threshold = self.config.get("optimal_threshold", 0.52)

            # ── Instantiate model ─────────────────────────────────────
            self.model = BRICSLiquiditySNN(
                n_features  = self.config["n_features"],
                hidden1     = self.config["hidden1"],
                hidden2     = self.config["hidden2"],
                tau         = self.config["tau"],
                v_threshold = self.config["v_threshold"],
            ).to(self._device)

            # ── Load weights ──────────────────────────────────────────
            state_dict = torch.load(MODEL_PATH, map_location=self._device)
            self.model.load_state_dict(state_dict)
            self.model.eval()

            # ── Load scaler ───────────────────────────────────────────
            self.scaler = joblib.load(SCALER_PATH)

            self.load_time_s = round(time.time() - t0, 3)
            self.loaded      = True

            print(f"✅ SNN model loaded successfully")
            print(f"   Architecture : FC({self.n_features}→{self.config['hidden1']})"
                  f"→BN→LIF→FC(→{self.config['hidden2']})→BN→LIF→FC(→1)")
            print(f"   Parameters   : {self.model.count_parameters()}")
            print(f"   Val AUC      : {self.config.get('val_auc', 'N/A')}")
            print(f"   Prob thresh  : {self.prob_threshold} (from config optimal_threshold)")
            print(f"   Lookback     : {self.lookback} days")
            print(f"   Features     : {self.n_features}")
            print(f"   Load time    : {self.load_time_s}s")

        except Exception as e:
            print(f"❌ Model load failed: {type(e).__name__}: {e}")
            self.loaded = False

    def _build_feature_sequence(self,
                                  prices: np.ndarray) -> np.ndarray:
        """
        Build a (lookback, n_features) feature matrix from raw prices.

        Computes all 9 training features at each of the 10 timesteps.
        Matches the exact feature engineering from notebook 06.

        Feature order (must match FEATURE_COLS from training):
        0: daily_return       — % change from previous close
        1: log_return         — log(P_t / P_{t-1})
        2: rolling_mean_7d    — 7-day rolling mean (normalised as deviation)
        3: rolling_std_7d     — 7-day rolling std
        4: price_momentum_5d  — (P_t - P_{t-5}) / P_{t-5}
        5: spike_signal       — 1 if |return| > 0.003 else 0
        6: spike_intensity    — |return| if spike, else 0
        7: inter_spike_interval — steps since last spike
        8: india_repo_rate    — latest RBI rate (forward-filled)

        Parameters
        ----------
        prices : np.ndarray, shape (10,) — 10 raw INR/BRL closes

        Returns
        -------
        np.ndarray, shape (10, 9) — raw (unscaled) feature matrix
        """
        T          = len(prices)
        n_feat     = self.n_features
        feat_matrix = np.zeros((T, n_feat), dtype=np.float32)

        # Pre-compute returns for all timesteps
        log_returns   = np.concatenate([[0.0], np.log(prices[1:] / prices[:-1])])
        daily_returns = np.concatenate([[0.0],
                                         (prices[1:] - prices[:-1]) / prices[:-1]])

        # Spike history for ISI calculation
        spike_history = [abs(r) > SPIKE_THRESHOLD for r in daily_returns]

        for t in range(T):
            # ── Feature 0: daily return ───────────────────────────────
            daily_ret = float(daily_returns[t])

            # ── Feature 1: log return ─────────────────────────────────
            log_ret   = float(log_returns[t])

            # ── Feature 2: rolling mean — raw value, matches training ────
            # Training: price.rolling(7).mean() → raw price level (~15–17)
            # Scaler was fitted on these raw values, so send raw here too.
            window         = prices[max(0, t-6) : t+1]
            roll_mean_feat = float(np.mean(window))

            # ── Feature 3: rolling std — ddof=1 matches pandas .std() ──
            roll_std = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0

            # ── Feature 4: price momentum 5d ──────────────────────────
            if t >= 5:
                momentum = float((prices[t] - prices[t-5]) / prices[t-5])
            else:
                momentum = 0.0

            # ── Feature 5: spike signal ───────────────────────────────
            spike_sig = 1.0 if abs(daily_ret) > SPIKE_THRESHOLD else 0.0

            # ── Feature 6: spike intensity ────────────────────────────
            spike_int = abs(daily_ret) if spike_sig == 1.0 else 0.0

            # ── Feature 7: inter-spike interval ──────────────────────
            isi = 0.0
            for j in range(t - 1, -1, -1):
                if spike_history[j]:
                    break
                isi += 1.0

            # ── Feature 8: India repo rate ────────────────────────────
            india_repo = 6.5   # latest RBI rate, forward-filled

            feat_matrix[t] = [
                daily_ret, log_ret, roll_mean_feat, roll_std,
                momentum, spike_sig, spike_int, isi, india_repo
            ]

        return feat_matrix   # (10, 9) raw features

    def preprocess(self, price_sequence: list) -> torch.Tensor:
        """
        Convert raw INR/BRL price sequence to a model-ready tensor.

        Steps
        -----
        1. Validate input (10 prices, all positive)
        2. Build (10, 9) feature matrix from prices
        3. Apply saved MinMaxScaler (fitted on training data)
        4. Clip to [0, 1] (handles val values outside train range)
        5. Reshape to (1, 10, 9) PyTorch tensor

        Parameters
        ----------
        price_sequence : list of 10 float — INR/BRL closing prices

        Returns
        -------
        torch.Tensor, shape (1, 10, n_features), dtype float32

        Raises
        ------
        ValueError : if sequence length != 10 or prices invalid
        RuntimeError: if scaler not loaded
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded")

        if len(price_sequence) != self.lookback:
            raise ValueError(
                f"Expected {self.lookback} prices, got {len(price_sequence)}"
            )

        prices = np.array(price_sequence, dtype=np.float64)

        if np.any(prices <= 0):
            raise ValueError("All prices must be positive")
        if np.any(prices > 100) or np.any(prices < 5):
            raise ValueError(
                "Prices out of expected INR/BRL range [5, 100]. "
                "Check you are sending INR/BRL rates."
            )

        # Build feature matrix
        feat_matrix = self._build_feature_sequence(prices)  # (10, 9)

        # Scale using training scaler
        feat_scaled = self.scaler.transform(feat_matrix)    # (10, 9)
        feat_scaled = np.clip(feat_scaled, 0.0, 1.0)

        # Convert to tensor: (1, 10, 9)
        tensor = torch.tensor(
            feat_scaled[np.newaxis, :, :], dtype=torch.float32
        )
        return tensor

    def predict(self, price_sequence: list) -> dict:
        """
        Run full SNN inference on a 10-step price sequence.

        Parameters
        ----------
        price_sequence : list of 10 float — INR/BRL daily closes

        Returns
        -------
        dict with keys:
            prob            float  — raw sigmoid output [0,1]
            direction       str    — "UP" or "DOWN"
            confidence      float  — |prob - 0.5| × 2, in [0,1]
            spike_rate      float  — mean LIF spike rate [0,1]
            spike_rate_pct  float  — spike rate as percentage
            recommendation  str    — "DIRECT" or "USD_FALLBACK"
            decision        str    — same as recommendation
            lif1_rate       float  — layer 1 mean spike rate
            lif2_rate       float  — layer 2 mean spike rate
            latency_ms      float  — inference time in milliseconds

        Raises
        ------
        RuntimeError : if model not loaded
        ValueError   : if price_sequence is invalid
        """
        if not self.loaded:
            raise RuntimeError(
                "SNN model not loaded. Check outputs/ directory."
            )

        t_start = time.perf_counter()

        # ── Preprocess ────────────────────────────────────────────────
        x = self.preprocess(price_sequence)   # (1, 10, 9)
        x = x.to(self._device)
        T = x.shape[1]

        # ── Forward pass with spike monitoring ────────────────────────
        self.model.eval()
        functional.reset_net(self.model)

        acc1 = torch.zeros(1, self.model.hidden1, device=self._device)
        acc2 = torch.zeros(1, self.model.hidden2, device=self._device)

        with torch.no_grad():
            for t in range(T):
                x_t  = x[:, t, :]
                s1   = self.model.lif1(self.model.bn1(self.model.fc1(x_t)))
                s2   = self.model.lif2(self.model.bn2(self.model.fc2(s1)))
                acc1 += s1
                acc2 += s2

            lif1_rate  = float((acc1 / T).mean().item())
            lif2_rate  = float((acc2 / T).mean().item())
            spike_rate = (lif1_rate + lif2_rate) / 2

            logit = self.model.fc3(acc2 / T)
            prob  = float(torch.sigmoid(logit).squeeze().item())

        latency_ms = (time.perf_counter() - t_start) * 1000

        # ── Decision ──────────────────────────────────────────────────
        direction  = "UP"   if prob >= 0.5 else "DOWN"
        confidence = abs(prob - 0.5) * 2

        if prob >= self.prob_threshold and spike_rate >= RATE_THRESHOLD:
            recommendation = "DIRECT"
            rec_text       = (
                f"Route via direct INR/BRL settlement. "
                f"Prob={prob:.3f} ≥ {self.prob_threshold}, "
                f"spike_rate={spike_rate:.3f} ≥ {RATE_THRESHOLD}."
            )
        elif prob < self.prob_threshold:
            recommendation = "USD_FALLBACK"
            rec_text       = (
                f"Use USD route. Model probability {prob:.3f} "
                f"below threshold {self.prob_threshold}."
            )
        else:
            recommendation = "USD_FALLBACK"
            rec_text       = (
                f"Use USD route. Spike rate {spike_rate:.3f} "
                f"below threshold {RATE_THRESHOLD} — "
                f"insufficient signal richness."
            )

        return {
            "prob"            : round(prob,        4),
            "direction"       : direction,
            "confidence"      : round(confidence,  4),
            "spike_rate"      : round(spike_rate,  4),
            "spike_rate_pct"  : round(spike_rate * 100, 2),
            "lif1_rate"       : round(lif1_rate,   4),
            "lif2_rate"       : round(lif2_rate,   4),
            "recommendation"  : recommendation,
            "decision"        : recommendation,
            "recommendation_text": rec_text,
            "prob_threshold_used": self.prob_threshold,
            "rate_threshold_used": RATE_THRESHOLD,
            "latency_ms"      : round(latency_ms,  3),
        }

    def get_status(self) -> dict:
        """Return predictor status for the /health endpoint."""
        return {
            "loaded"         : self.loaded,
            "model_version"  : "BRICSLiquiditySNN v1.0",
            "n_features"     : self.n_features,
            "lookback"       : self.lookback,
            "parameters"     : self.model.count_parameters() if self.model else 0,
            "val_auc"        : self.config.get("val_auc", 0.0) if self.config else 0.0,
            "load_time_s"    : self.load_time_s,
            "prob_threshold" : self.prob_threshold,
            "rate_threshold" : RATE_THRESHOLD,
        }


# ── Singleton instantiation ────────────────────────────────────────────
# Loaded once when the module is first imported.
# FastAPI imports this module at startup via lifespan.
# All subsequent requests reuse this instance.
predictor = SNNPredictor()


# ── Module-level compatibility wrappers ───────────────────────────────
# api.main imports this module and expects function-style helpers.
def load_model() -> bool:
    """
    Ensure model is loaded for API startup.

    Returns
    -------
    bool
        True if model/scaler/config are available, else False.
    """
    if not predictor.loaded:
        predictor._load()
    return predictor.loaded


def is_loaded() -> bool:
    """Return whether the singleton predictor is ready."""
    return predictor.loaded


def get_config() -> dict:
    """Return loaded model config (empty dict if unavailable)."""
    return predictor.config if predictor.config is not None else {}


def predict(price_sequence: list) -> dict:
    """Run inference through the singleton predictor."""
    return predictor.predict(price_sequence)


# ── Standalone test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("SNNPredictor standalone test")
    print("=" * 55)

    if not predictor.loaded:
        print("❌ Model not loaded — check outputs/ directory")
        exit(1)

    # Test 1: normal sequence (realistic INR/BRL rates)
    test_prices = [16.2, 16.3, 16.1, 16.4, 16.5,
                   16.3, 16.6, 16.4, 16.7, 16.5]

    print(f"\nTest 1 — Normal INR/BRL sequence")
    print(f"  Input prices : {test_prices}")
    result = predictor.predict(test_prices)

    print(f"\n  Results:")
    for k, v in result.items():
        if k != "recommendation_text":
            print(f"    {k:<22}: {v}")
    print(f"    recommendation_text: {result['recommendation_text']}")

    # Test 2: upward trending sequence
    up_prices = [16.0, 16.1, 16.2, 16.3, 16.4,
                  16.5, 16.6, 16.7, 16.8, 16.9]
    print(f"\nTest 2 — Upward trend")
    result2 = predictor.predict(up_prices)
    print(f"  direction={result2['direction']}  "
          f"prob={result2['prob']:.4f}  "
          f"spike_rate={result2['spike_rate']:.4f}  "
          f"→ {result2['recommendation']}")

    # Test 3: volatile sequence (should trigger spikes)
    volatile = [16.0, 16.8, 15.9, 17.1, 16.2,
                 17.3, 15.8, 17.0, 16.1, 16.9]
    print(f"\nTest 3 — Volatile sequence")
    result3 = predictor.predict(volatile)
    print(f"  direction={result3['direction']}  "
          f"prob={result3['prob']:.4f}  "
          f"spike_rate={result3['spike_rate']:.4f}  "
          f"→ {result3['recommendation']}")

    # Test 4: validate error handling
    print(f"\nTest 4 — Error handling")
    try:
        predictor.predict([16.0] * 5)   # wrong length
        print("  ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✅ ValueError caught: {e}")

    try:
        predictor.predict([16.0] * 9 + [-1.0])   # negative price
        print("  ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✅ ValueError caught: {e}")

    # Preprocess shape check
    print(f"\nTest 5 — Preprocess shape")
    tensor = predictor.preprocess(test_prices)
    print(f"  Output tensor shape : {tuple(tensor.shape)}")
    print(f"  Expected            : (1, 10, {predictor.n_features})")
    assert tuple(tensor.shape) == (1, 10, predictor.n_features), "Shape mismatch!"
    print(f"  ✅ Shape correct")
    print(f"  Value range         : [{tensor.min():.4f}, {tensor.max():.4f}]")

    print(f"\n{'='*55}")
    print(f"✅ All tests passed — predictor ready for FastAPI")
    print(f"{'='*55}")