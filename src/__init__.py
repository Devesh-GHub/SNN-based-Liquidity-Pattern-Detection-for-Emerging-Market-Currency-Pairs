# src/__init__.py
# BRICS SNN Project — source package
# Import key functions for convenience

from .data_utils import (
    standardize_columns,
    set_date_index,
    get_close_col,
    fill_daily_gaps,
    fill_hourly_gaps,
    flag_outliers,
    encode_to_spikes,
    compute_inter_spike_interval,
)

from .feature_engineering import (
    build_price_features,
    build_spike_features,
    merge_macro_feature,
    build_target,
    build_feature_matrix,
    create_sequences,
)

__version__ = "0.1.0"
__author__  = "BRICS SNN Project"