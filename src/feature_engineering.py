"""
src/feature_engineering.py
===========================
Feature engineering pipeline for the BRICS SNN project.

Converts clean OHLCV price data into model-ready feature matrices
containing price-based, spike-based, and macro features.

All functions follow the no-leakage rule: every feature uses only
information available at or before the current timestep t.
The target variable uses t+1 data (tomorrow's direction) — this is
intentional and correct for next-day prediction.

Usage
-----
from src.feature_engineering import build_feature_matrix, create_sequences

feat = build_feature_matrix(price_series, repo_series)
X, y, dates = create_sequences(feat, FEATURE_COLS, 'target', lookback=10)
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from .data_utils import encode_to_spikes, compute_inter_spike_interval


# ─────────────────────────────────────────────
# SECTION 1 — Price Features
# ─────────────────────────────────────────────

def build_price_features(price: pd.Series,
                          rolling_window: int = 7,
                          momentum_window: int = 5) -> pd.DataFrame:
    """
    Build price-based features from a close price series.

    Features created:
    - daily_return      : % change from previous timestep (decimal)
    - log_return        : log(P_t / P_{t-1}), more stable for ML
    - rolling_mean      : rolling mean over `rolling_window` periods
    - rolling_std       : rolling std (local volatility proxy)
    - price_momentum    : (P_t - P_{t-k}) / P_{t-k} over momentum_window

    All features use only past data (no look-ahead).
    rolling_window uses min_periods=5 to handle early rows gracefully.

    Parameters
    ----------
    price : pd.Series
        Close price series with DatetimeIndex.
        For daily data: rolling_window=7, momentum_window=5
        For hourly data: rolling_window=24, momentum_window=8
    rolling_window : int, default 7
        Window size for rolling mean and std.
    momentum_window : int, default 5
        Lookback period for momentum calculation.

    Returns
    -------
    pd.DataFrame
        DataFrame with 5 price feature columns and same index as price.

    Examples
    --------
    >>> price_feat = build_price_features(price, rolling_window=7)
    >>> price_feat.columns.tolist()
    ['daily_return', 'log_return', 'rolling_mean', 'rolling_std',
     'price_momentum']
    """
    feat = pd.DataFrame(index=price.index)

    # 1. Daily/hourly % return
    feat["daily_return"] = price.pct_change()

    # 2. Log return — additive across time, closer to normally distributed
    feat["log_return"] = np.log(price / price.shift(1))

    # 3. Rolling mean — trend context
    feat["rolling_mean"] = price.rolling(
        window=rolling_window, min_periods=5
    ).mean()

    # 4. Rolling std — local volatility
    feat["rolling_std"] = price.rolling(
        window=rolling_window, min_periods=5
    ).std()

    # 5. Momentum — medium-term direction
    feat["price_momentum"] = (
        (price - price.shift(momentum_window)) / price.shift(momentum_window)
    )

    return feat


# ─────────────────────────────────────────────
# SECTION 2 — Spike Features
# ─────────────────────────────────────────────

def build_spike_features(returns: pd.Series,
                          threshold: float = 0.003,
                          isi_unit: str = "days") -> pd.DataFrame:
    """
    Build neuromorphic spike-based features from a return series.

    Features created:
    - spike_signal          : binary 0/1 rate-coded spike (from encode_to_spikes)
    - spike_intensity       : abs(return) on spike days, 0 otherwise
    - inter_spike_interval  : time since last spike (ISI)

    The ISI feature is unique to this project — no standard finance ML
    paper uses it. It encodes volatility clustering: a long ISI suggests
    the market has been calm, which may amplify the next spike's significance.

    Parameters
    ----------
    returns : pd.Series
        Daily or hourly return series in decimal form.
    threshold : float, default 0.003
        Spike threshold (0.003 for daily, 0.001 for hourly).
    isi_unit : str, default 'days'
        Unit for ISI values ('days' for daily, 'hours' for hourly).

    Returns
    -------
    pd.DataFrame
        DataFrame with 3 spike feature columns.

    Examples
    --------
    >>> spike_feat = build_spike_features(returns, threshold=0.003)
    >>> spike_feat['spike_signal'].mean() * 100
    14.4
    """
    feat = pd.DataFrame(index=returns.index)

    # 1. Binary spike signal
    feat["spike_signal"] = encode_to_spikes(returns, threshold=threshold)

    # 2. Spike intensity — magnitude preserved on spike days only
    feat["spike_intensity"] = np.where(
        feat["spike_signal"] == 1,
        returns.abs(),
        0.0
    )

    # 3. Inter-spike interval
    feat["inter_spike_interval"] = compute_inter_spike_interval(
        feat["spike_signal"], unit=isi_unit
    )

    return feat


# ─────────────────────────────────────────────
# SECTION 3 — Macro Features
# ─────────────────────────────────────────────

def merge_macro_feature(feat: pd.DataFrame,
                         macro_series: pd.Series,
                         col_name: str = "india_repo_rate") -> pd.DataFrame:
    """
    Forward-fill a low-frequency macro series onto a higher-frequency index.

    The RBI repo rate is announced monthly and remains constant until
    the next announcement. Forward-filling is the correct treatment —
    the rate is publicly known from the announcement date onwards,
    so there is no look-ahead bias.

    Parameters
    ----------
    feat : pd.DataFrame
        Feature matrix with a DatetimeIndex (daily or hourly).
    macro_series : pd.Series
        Low-frequency series (e.g., monthly repo rate) with DatetimeIndex.
        Will be forward-filled onto feat's index.
    col_name : str, default 'india_repo_rate'
        Column name to assign to the merged macro feature.

    Returns
    -------
    pd.DataFrame
        Copy of feat with the macro feature column added.
        Rows where no prior macro value exists will be NaN.

    Examples
    --------
    >>> feat = merge_macro_feature(feat, repo_series, 'india_repo_rate')
    >>> feat['india_repo_rate'].isna().sum()
    0
    """
    feat = feat.copy()

    # Build a daily/hourly range from macro series
    macro_daily = macro_series.reindex(
        pd.date_range(
            start=macro_series.index.min(),
            end=max(macro_series.index.max(), feat.index.max()
                    .tz_localize(None) if feat.index.tz else feat.index.max()),
            freq="D"
        )
    ).ffill()
    macro_daily.index.name = feat.index.name

    # Align timezone if needed
    if feat.index.tz is not None and macro_daily.index.tz is None:
        macro_daily.index = macro_daily.index.tz_localize(feat.index.tz)

    feat[col_name] = macro_daily.reindex(feat.index).ffill().values
    return feat


# ─────────────────────────────────────────────
# SECTION 4 — Target Variable
# ─────────────────────────────────────────────

def build_target(price: pd.Series,
                  col_name: str = "target") -> pd.Series:
    """
    Build a binary next-timestep direction target.

    Target definition:
        1  if P_{t+1} > P_t  (price goes up tomorrow)
        0  if P_{t+1} <= P_t (price goes down or flat)

    The last row will always be NaN (no tomorrow exists) and should
    be dropped before model training.

    Why binary direction, not exact price?
    - Exact price prediction is unstable and overly ambitious for a thesis
    - Direction gives a clear, auditable settlement signal:
      "expect rate to improve → delay settlement by 1 day"
    - Binary classification trains more stably with surrogate gradients

    Parameters
    ----------
    price : pd.Series
        Close price series with DatetimeIndex.
    col_name : str, default 'target'
        Name for the returned Series.

    Returns
    -------
    pd.Series
        Binary (0/1) series with same index as price.
        Last row is NaN.

    Examples
    --------
    >>> target = build_target(price)
    >>> target.mean()   # should be ~0.5 for a random walk
    0.508
    """
    target = (price.shift(-1) > price).astype(float)
    target.iloc[-1] = np.nan   # no tomorrow for the last row
    target.name = col_name
    return target


# ─────────────────────────────────────────────
# SECTION 5 — Full Pipeline
# ─────────────────────────────────────────────

def build_feature_matrix(price: pd.Series,
                          macro_series: Optional[pd.Series] = None,
                          threshold: float = 0.003,
                          rolling_window: int = 7,
                          momentum_window: int = 5,
                          isi_unit: str = "days") -> pd.DataFrame:
    """
    Build the complete feature matrix from a price series.

    Orchestrates all feature building steps in correct order:
    1. Price features (return, log return, rolling stats, momentum)
    2. Spike features (signal, intensity, ISI)
    3. Macro feature (repo rate, optional)
    4. Target variable (next-day direction)

    Parameters
    ----------
    price : pd.Series
        Close price series with DatetimeIndex.
    macro_series : pd.Series, optional
        Low-frequency macro series (e.g., India repo rate).
        If None, no macro feature is added.
    threshold : float, default 0.003
        Spike encoding threshold (0.003 for daily, 0.001 for hourly).
    rolling_window : int, default 7
        Window for rolling mean/std (7 for daily, 24 for hourly).
    momentum_window : int, default 5
        Lookback for momentum (5 for daily, 8 for hourly).
    isi_unit : str, default 'days'
        Unit for ISI ('days' for daily, 'hours' for hourly).

    Returns
    -------
    pd.DataFrame
        Complete feature matrix with all features + target column.
        Column order: price features, spike features, [macro], target.

    Examples
    --------
    >>> feat = build_feature_matrix(price, repo_series, threshold=0.003)
    >>> feat.shape
    (1822, 10)
    >>> feat.columns.tolist()
    ['daily_return', 'log_return', 'rolling_mean', 'rolling_std',
     'price_momentum', 'spike_signal', 'spike_intensity',
     'inter_spike_interval', 'india_repo_rate', 'target']
    """
    # Step 1 — price features
    price_feat = build_price_features(price, rolling_window, momentum_window)

    # Step 2 — spike features (built from daily_return)
    spike_feat = build_spike_features(
        price_feat["daily_return"], threshold=threshold, isi_unit=isi_unit
    )

    # Step 3 — combine
    feat = pd.concat([price_feat, spike_feat], axis=1)

    # Step 4 — macro (optional)
    if macro_series is not None:
        feat = merge_macro_feature(feat, macro_series, "india_repo_rate")

    # Step 5 — target
    feat["target"] = build_target(price)

    return feat


# ─────────────────────────────────────────────
# SECTION 6 — Sequence Creation
# ─────────────────────────────────────────────

def create_sequences(data: pd.DataFrame,
                      feature_cols: List[str],
                      target_col: str,
                      lookback: int = 10) -> Tuple[np.ndarray,
                                                    np.ndarray,
                                                    list]:
    """
    Convert a flat feature matrix into sliding-window sequences for LSTM/SNN.

    For each row t (starting at row `lookback`):
        X[t] = features from rows (t - lookback) to (t - 1)  ← past only
        y[t] = target at row t                                ← what we predict

    This is the correct temporal ordering — X never includes row t or later,
    preventing any form of look-ahead data leakage.

    Parameters
    ----------
    data : pd.DataFrame
        Feature matrix with DatetimeIndex (should be dropna'd beforehand).
    feature_cols : list of str
        Column names to use as model input features.
    target_col : str
        Name of the target column.
    lookback : int, default 10
        Number of past timesteps in each sequence window.
        10 days for daily data, 24 hours for hourly data.

    Returns
    -------
    X : np.ndarray, shape (n_samples, lookback, n_features)
        Input sequences. dtype float32.
    y : np.ndarray, shape (n_samples,)
        Binary targets. dtype float32.
    dates : list of pd.Timestamp
        Prediction date for each sample (date of the target, not the window).

    Raises
    ------
    ValueError
        If lookback >= len(data) or feature_cols not in data.

    Examples
    --------
    >>> X, y, dates = create_sequences(train, FEATURE_COLS, 'target', 10)
    >>> X.shape
    (712, 10, 9)
    >>> y.shape
    (712,)
    """
    missing = [c for c in feature_cols + [target_col] if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")
    if lookback >= len(data):
        raise ValueError(
            f"lookback ({lookback}) must be less than data length ({len(data)})"
        )

    feature_arr = data[feature_cols].values.astype(np.float32)
    target_arr  = data[target_col].values.astype(np.float32)
    dates       = list(data.index)

    X_list, y_list, date_list = [], [], []

    for t in range(lookback, len(data)):
        X_list.append(feature_arr[t - lookback : t])
        y_list.append(target_arr[t])
        date_list.append(dates[t])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, y, date_list


# ─────────────────────────────────────────────
# SECTION 7 — Normalisation
# ─────────────────────────────────────────────

def fit_normaliser(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a MinMax normaliser on training sequences only.

    Computes per-feature min and max from X_train.
    MUST be called on training data only — never on val or test.
    Apply the returned parameters using `apply_normaliser`.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_samples, lookback, n_features)
        Training sequences.

    Returns
    -------
    feat_min : np.ndarray, shape (n_features,)
        Per-feature minimum values from training data.
    feat_max : np.ndarray, shape (n_features,)
        Per-feature maximum values from training data.

    Examples
    --------
    >>> feat_min, feat_max = fit_normaliser(X_train)
    >>> X_train_norm = apply_normaliser(X_train, feat_min, feat_max)
    >>> X_val_norm   = apply_normaliser(X_val,   feat_min, feat_max)
    """
    # Reshape to (n_samples * lookback, n_features) for per-feature stats
    n_samples, lookback, n_features = X_train.shape
    X_flat   = X_train.reshape(-1, n_features)
    feat_min = np.nanmin(X_flat, axis=0)
    feat_max = np.nanmax(X_flat, axis=0)
    return feat_min, feat_max


def apply_normaliser(X: np.ndarray,
                      feat_min: np.ndarray,
                      feat_max: np.ndarray,
                      eps: float = 1e-8) -> np.ndarray:
    """
    Apply MinMax normalisation using pre-fitted parameters.

    Scales all feature values to [0, 1] range based on training
    statistics. Values outside the training range are clipped to [0, 1].

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, lookback, n_features)
        Sequences to normalise (train, val, or test).
    feat_min : np.ndarray, shape (n_features,)
        Per-feature minimum from training data (from fit_normaliser).
    feat_max : np.ndarray, shape (n_features,)
        Per-feature maximum from training data (from fit_normaliser).
    eps : float, default 1e-8
        Small constant to prevent division by zero on constant features.

    Returns
    -------
    np.ndarray
        Normalised array with same shape as X, values in [0, 1].

    Examples
    --------
    >>> X_val_norm = apply_normaliser(X_val, feat_min, feat_max)
    >>> X_val_norm.min(), X_val_norm.max()
    (0.0, 1.0)
    """
    X_norm = (X - feat_min) / (feat_max - feat_min + eps)
    X_norm = np.clip(X_norm, 0.0, 1.0)
    return X_norm.astype(np.float32)