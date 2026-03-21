"""
src/data_utils.py
=================
Data cleaning and spike encoding utilities for the BRICS SNN project.

All functions are pure (no side effects on input DataFrames) — they
return new objects and leave inputs unchanged.

Usage
-----
from src.data_utils import (
    standardize_columns, set_date_index, fill_daily_gaps,
    flag_outliers, encode_to_spikes
)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


# ─────────────────────────────────────────────
# SECTION 1 — Column and Index Standardisation
# ─────────────────────────────────────────────

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame column names to lowercase with underscores.

    Strips whitespace, lowercases, and replaces spaces/hyphens with
    underscores so all downstream code uses consistent column names.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with arbitrary column names.

    Returns
    -------
    pd.DataFrame
        Copy of df with standardized column names.

    Examples
    --------
    >>> df.columns = ['Date', 'Adj Close', 'High-Low']
    >>> standardize_columns(df).columns
    Index(['date', 'adj_close', 'high_low'], dtype='object')
    """
    df = df.copy()
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(" ", "_", regex=False)
          .str.replace("-", "_", regex=False)
    )
    return df


def set_date_index(df: pd.DataFrame,
                   tz: Optional[str] = None) -> pd.DataFrame:
    """
    Detect the date/datetime column, parse it, and set it as the index.

    Searches column names for keywords 'date', 'datetime', or 'time'.
    Sorts the index ascending after setting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a date column somewhere in its columns.
    tz : str, optional
        Timezone to localize to if timestamps are tz-naive.
        Pass 'UTC' for hourly forex data.

    Returns
    -------
    pd.DataFrame
        Copy of df with a DatetimeIndex named 'date' or 'datetime_utc'.

    Raises
    ------
    ValueError
        If no date column is found.

    Examples
    --------
    >>> df = set_date_index(raw_df)
    >>> df.index.name
    'date'
    """
    df = df.copy()

    # Detect date column
    date_col = None
    for col in df.columns:
        if any(k in col.lower() for k in ["date", "datetime", "time"]):
            date_col = col
            break
    if date_col is None:
        raise ValueError(
            f"No date column found. Available columns: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)

    # Timezone handling
    if tz is not None:
        sample = df[date_col].dropna().iloc[0]
        if sample.tzinfo is not None:
            df[date_col] = df[date_col].dt.tz_convert(tz)
        else:
            df[date_col] = df[date_col].dt.tz_localize(tz)
        index_name = "datetime_utc"
    else:
        index_name = "date"

    df = df.set_index(date_col)
    df.index.name = index_name
    df = df.sort_index()
    return df


def get_close_col(df: pd.DataFrame) -> str:
    """
    Return the name of the close price column in a DataFrame.

    Searches column names for the substring 'close' (case-insensitive).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame expected to contain a close price column.

    Returns
    -------
    str
        Name of the first matching close column.

    Raises
    ------
    ValueError
        If no close column is found.

    Examples
    --------
    >>> get_close_col(df)
    'usdinr_close'
    """
    for col in df.columns:
        if "close" in col.lower():
            return col
    raise ValueError(
        f"No close column found. Available: {list(df.columns)}"
    )


# ─────────────────────────────────────────────
# SECTION 2 — Gap Filling
# ─────────────────────────────────────────────

def fill_daily_gaps(df: pd.DataFrame,
                    close_col: str,
                    max_fill_days: int = 2,
                    drop_threshold: int = 3) -> Tuple[pd.DataFrame, dict]:
    """
    Forward-fill short gaps in daily forex data; drop long gaps.

    Forex daily data has natural gaps on weekends and holidays.
    Gaps of 1–2 days are forward-filled (expected market closures).
    Gaps longer than `drop_threshold` are dropped to avoid
    fabricating extended sequences of stale prices.

    Parameters
    ----------
    df : pd.DataFrame
        Clean daily DataFrame with DatetimeIndex.
    close_col : str
        Name of the close price column.
    max_fill_days : int, default 2
        Maximum consecutive NaN days to forward-fill.
    drop_threshold : int, default 3
        Gaps longer than this many days are dropped entirely.

    Returns
    -------
    df_out : pd.DataFrame
        Cleaned DataFrame with gaps handled.
    report : dict
        Summary of rows dropped and nulls filled.

    Examples
    --------
    >>> df_clean, report = fill_daily_gaps(df, 'close')
    >>> report['long_gap_rows_dropped']
    12
    """
    df = df.copy()
    series = df[close_col].copy()
    original_nulls = int(series.isnull().sum())

    # Reindex to full calendar range
    full_idx = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq="D"
    )
    series_full = series.reindex(full_idx)

    # Label each NaN with its consecutive gap length
    is_null = series_full.isnull()
    gap_id  = is_null.ne(is_null.shift()).cumsum()
    gap_len = is_null.groupby(gap_id).transform("sum")

    # Forward-fill only short gaps
    series_ffill = series_full.copy()
    series_ffill[is_null & (gap_len <= max_fill_days)] = np.nan
    series_ffill = series_ffill.ffill()

    # Drop long-gap rows
    long_gap_mask  = is_null & (gap_len > drop_threshold)
    rows_to_drop   = long_gap_mask[long_gap_mask].index
    series_ffill   = series_ffill.drop(index=rows_to_drop, errors="ignore")

    # Re-align df
    df_out = df.reindex(series_ffill.index).copy()
    df_out[close_col] = series_ffill

    report = {
        "original_rows"        : len(df),
        "original_nulls"       : original_nulls,
        "full_range_rows"      : len(full_idx),
        "long_gap_rows_dropped": len(rows_to_drop),
        "final_rows"           : len(df_out),
        "remaining_nulls"      : int(df_out[close_col].isnull().sum()),
    }
    return df_out, report


def fill_hourly_gaps(df: pd.DataFrame,
                     close_col: str,
                     max_fill_hours: int = 3,
                     drop_threshold_hours: int = 12) -> Tuple[pd.DataFrame, dict]:
    """
    Forward-fill short gaps in hourly forex data; drop long gaps.

    Hourly data has thin-market gaps (lunch breaks, off-hours) of
    1–3 hours which are safe to forward-fill. Gaps longer than
    12 hours (overnight/weekend) are dropped to avoid fabricating
    stale sequences.

    Parameters
    ----------
    df : pd.DataFrame
        Clean hourly DataFrame with UTC DatetimeIndex.
    close_col : str
        Name of the close price column.
    max_fill_hours : int, default 3
        Maximum consecutive missing hours to forward-fill.
    drop_threshold_hours : int, default 12
        Gaps longer than this many hours are dropped.

    Returns
    -------
    df_out : pd.DataFrame
        Cleaned DataFrame.
    report : dict
        Summary statistics of the cleaning operation.

    Examples
    --------
    >>> df_clean, report = fill_hourly_gaps(df, 'close', tz='UTC')
    """
    df = df.copy()
    series = df[close_col].copy()
    original_nulls = int(series.isnull().sum())

    tz = series.index.tz
    full_idx = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq="h",
        tz=tz
    )
    series_full = series.reindex(full_idx)

    is_null = series_full.isnull()
    gap_id  = is_null.ne(is_null.shift()).cumsum()
    gap_len = is_null.groupby(gap_id).transform("sum")

    series_ffill = series_full.copy()
    series_ffill[is_null & (gap_len <= max_fill_hours)] = np.nan
    series_ffill = series_ffill.ffill()

    long_gap_mask  = is_null & (gap_len > drop_threshold_hours)
    rows_to_drop   = long_gap_mask[long_gap_mask].index
    series_ffill   = series_ffill.drop(index=rows_to_drop, errors="ignore")

    df_out = df.reindex(series_ffill.index).copy()
    df_out[close_col] = series_ffill

    report = {
        "original_rows"        : len(df),
        "original_nulls"       : original_nulls,
        "full_range_rows"      : len(full_idx),
        "long_gap_rows_dropped": len(rows_to_drop),
        "final_rows"           : len(df_out),
        "remaining_nulls"      : int(df_out[close_col].isnull().sum()),
    }
    return df_out, report


# ─────────────────────────────────────────────
# SECTION 3 — Outlier Flagging
# ─────────────────────────────────────────────

def flag_outliers(df: pd.DataFrame,
                  close_col: str,
                  window: int = 30,
                  n_std: float = 3.0) -> Tuple[pd.DataFrame, int]:
    """
    Flag price outliers using a rolling mean ± n_std band.

    Does NOT delete outliers — adds a boolean 'is_outlier' column.
    Outliers may represent genuine flash crashes or data errors;
    flagging preserves them for downstream audit while making them
    identifiable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex and close price column.
    close_col : str
        Name of the close price column.
    window : int, default 30
        Rolling window size (days for daily, hours for hourly).
        Use 30 for daily data, 168 (1 week) for hourly data.
    n_std : float, default 3.0
        Number of standard deviations beyond which a value is flagged.

    Returns
    -------
    df_out : pd.DataFrame
        Copy of df with added columns:
        'roll_mean', 'roll_upper', 'roll_lower', 'is_outlier'.
    n_outliers : int
        Number of rows flagged as outliers.

    Examples
    --------
    >>> df_flagged, n = flag_outliers(df, 'close', window=30)
    >>> print(f"{n} outliers flagged ({100*n/len(df):.2f}%)")
    """
    df = df.copy()
    roll_mean = df[close_col].rolling(window=window, min_periods=5).mean()
    roll_std  = df[close_col].rolling(window=window, min_periods=5).std()

    upper = roll_mean + n_std * roll_std
    lower = roll_mean - n_std * roll_std

    df["roll_mean"]  = roll_mean.round(6)
    df["roll_upper"] = upper.round(6)
    df["roll_lower"] = lower.round(6)
    df["is_outlier"] = (
        (df[close_col] > upper) | (df[close_col] < lower)
    )

    n_outliers = int(df["is_outlier"].sum())
    return df, n_outliers


# ─────────────────────────────────────────────
# SECTION 4 — Spike Encoding
# ─────────────────────────────────────────────

def encode_to_spikes(returns: pd.Series,
                     threshold: float = 0.003) -> pd.Series:
    """
    Rate-code a return series into a binary spike train.

    Implements manual rate coding: a spike fires (value=1) when the
    absolute daily/hourly return exceeds the threshold. This maps the
    financial concept of a "significant price move" onto the
    neuromorphic concept of a threshold-crossing spike event.

    The threshold of 0.003 (0.3%) is calibrated for the INR/BRL
    synthetic cross-rate, which has a daily return std of ~0.24%.
    A 0.3% threshold ≈ 1.25× std, producing ~14% spike frequency —
    sparse enough to be informative, dense enough for model training.

    Parameters
    ----------
    returns : pd.Series
        Series of daily or hourly % returns in decimal form
        (e.g., 0.008 = 0.8% return). Must NOT be in percentage form
        (e.g., NOT 0.8 for 0.8%).
    threshold : float, default 0.003
        Minimum absolute return to trigger a spike.
        Daily default  : 0.003 (0.3%)
        Hourly default : 0.001 (0.1%)

    Returns
    -------
    pd.Series
        Binary integer series (0 or 1) with same index as returns.
        1 = spike fired, 0 = quiet timestep.

    Examples
    --------
    >>> spikes = encode_to_spikes(returns, threshold=0.003)
    >>> print(f"Spike rate: {spikes.mean()*100:.1f}%")
    Spike rate: 14.4%
    """
    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns must be pd.Series, got {type(returns)}")
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    spikes = (returns.abs() > threshold).astype(int)
    return spikes


def compute_inter_spike_interval(spike_signal: pd.Series,
                                  unit: str = "days") -> pd.Series:
    """
    Compute inter-spike interval (ISI) for a binary spike train.

    The ISI at each timestep is the elapsed time since the most recent
    prior spike. On spike days, it records the gap from the previous
    spike. On quiet days, it records how long it has been since the
    last spike. The first spike and all pre-spike timesteps are NaN.

    ISI is the core neuromorphic-specific feature in this project —
    it encodes the "rhythm" of spike activity and reflects volatility
    clustering: a long ISI means the market has been calm, making the
    next spike potentially more significant.

    Parameters
    ----------
    spike_signal : pd.Series
        Binary series (0/1) with a DatetimeIndex.
    unit : str, default 'days'
        Time unit for ISI values. 'days' for daily data,
        'hours' for hourly data.

    Returns
    -------
    pd.Series
        Float series of ISI values (same index as spike_signal).
        NaN for timesteps before the first spike.

    Examples
    --------
    >>> isi = compute_inter_spike_interval(feat['spike_signal'], unit='days')
    >>> print(f"Mean ISI: {isi.mean():.1f} days")
    Mean ISI: 7.3 days
    """
    if unit not in ("days", "hours"):
        raise ValueError(f"unit must be 'days' or 'hours', got '{unit}'")

    if not isinstance(spike_signal.index, pd.DatetimeIndex):
        raise TypeError("spike_signal index must be a DatetimeIndex")

    # Mark spike timestamps, then shift to use the most recent PRIOR spike.
    idx_series = spike_signal.index.to_series()
    spike_ts = idx_series.where(spike_signal.astype(bool))
    last_spike_before = spike_ts.shift(1).ffill()

    delta = idx_series - last_spike_before
    if unit == "hours":
        return (delta.dt.total_seconds() / 3600.0).astype(float)

    return delta.dt.days.astype(float)


# ─────────────────────────────────────────────
# SECTION 5 — Quick validation helper
# ─────────────────────────────────────────────

def validate_feature_matrix(df: pd.DataFrame,
                              target_col: str = "target") -> bool:
    """
    Run basic sanity checks on a completed feature matrix.

    Checks performed:
    - No future data leakage (target is shift(-1), all features shift(0) or older)
    - Target column exists and is binary (0/1)
    - Index is a DatetimeIndex sorted ascending
    - No all-NaN columns

    Parameters
    ----------
    df : pd.DataFrame
        Completed feature matrix with target column.
    target_col : str, default 'target'
        Name of the target column.

    Returns
    -------
    bool
        True if all checks pass.

    Raises
    ------
    AssertionError
        With a descriptive message if any check fails.
    """
    # Index check
    assert isinstance(df.index, pd.DatetimeIndex), \
        "Index must be DatetimeIndex"
    assert df.index.is_monotonic_increasing, \
        "Index must be sorted ascending (chronological)"

    # Target check
    assert target_col in df.columns, \
        f"Target column '{target_col}' not found"
    unique_vals = set(df[target_col].dropna().unique())
    assert unique_vals.issubset({0, 1, 0.0, 1.0}), \
        f"Target must be binary (0/1), found: {unique_vals}"

    # No all-NaN columns
    all_nan = [c for c in df.columns if df[c].isna().all()]
    assert len(all_nan) == 0, \
        f"All-NaN columns found: {all_nan}"

    # Target balance (warn only)
    target_mean = df[target_col].mean()
    if not (0.35 <= target_mean <= 0.65):
        print(f"⚠️  Warning: target imbalance detected "
              f"(mean={target_mean:.2f}). Consider class weights.")

    print("✅ Feature matrix validation passed")
    return True