# Event-Driven Synthetic Liquidity Discovery for BRICS Settlements Using Spiking Neural Networks

> **project — Month 1 of 4 complete**  
> INR/BRL direct settlement simulation powered by neuromorphic computing

---

## Problem

Indian exporters transacting with Brazilian importers currently lose over 4%
of transaction value — approximately ₹42,000 on every ₹10 lakh payment —
due to mandatory USD routing through correspondent banks with T+2 to T+3
settlement delays. No direct INR/BRL market exists; every transaction embeds
a double FX conversion cost that neither party can negotiate away.

---

## Solution

This project simulates a direct INR/BRL settlement corridor using a Spiking
Neural Network (SNN) to predict FX rate direction and identify optimal
settlement windows, targeting a reduction in per-transaction cost from ~4.2%
to ~0.1% and settlement latency from T+2 to near real-time (T+0). The SNN's
event-driven, spike-based architecture is uniquely suited to the bursty,
heavy-tailed nature of emerging-market FX return distributions — processing
information only when a meaningful price event occurs rather than at every
timestep.

---

## Data

| Item | Detail |
|---|---|
| **Sources** | Yahoo Finance (via yfinance), FRED (Federal Reserve Economic Data) |
| **Currency pairs** | USD/INR (`INR=X`), USD/BRL (`BRL=X`) |
| **Derived pair** | Synthetic INR/BRL = USDINR / USDBRL |
| **Daily data** | 5 years, ~1,250 rows per pair |
| **Hourly data** | 2 years, ~14,000–17,000 rows per pair |
| **Macro data** | India long-term interest rate (FRED: `INDIRLTLT01STM`) |

### Features engineered (daily)
| Feature | Description | Type |
|---|---|---|
| `daily_return` | % price change from previous day | Price |
| `log_return` | log(P_t / P_{t-1}) — more stable for ML | Price |
| `rolling_mean_7d` | 7-day moving average | Price |
| `rolling_std_7d` | 7-day rolling volatility | Price |
| `price_momentum_5d` | Return over past 5 days | Price |
| `spike_signal` | 1 if \|return\| > 0.5%, else 0 | Spike |
| `spike_intensity` | Magnitude of return on spike days | Spike |
| `inter_spike_interval` | Days since last spike event | Spike |
| `india_repo_rate` | Monthly rate forward-filled to daily | Macro |
| `target` | 1 if next day price is higher, else 0 | Label |

### Features engineered (hourly)
| Feature | Description |
|---|---|
| `hourly_return` | % change from previous hour |
| `log_return` | log(P_t / P_{t-1}) |
| `rolling_mean_24h` | 24-hour moving average |
| `rolling_std_24h` | 24-hour rolling volatility |
| `price_momentum_8h` | Return over past 8 hours (one session) |
| `spike_signal` | 1 if \|return\| > 0.1%, else 0 |
| `spike_intensity` | Magnitude on spike hours |
| `inter_spike_interval_h` | Hours since last spike |
| `hour_of_day` | UTC hour (0–23) — captures session patterns |

---

## Repository Structure
```
brics_snn/
├── data/
│   ├── raw/                          # Original downloads — never modified
│   │   ├── USDINR_daily_raw.csv
│   │   ├── USDBRL_daily_raw.csv
│   │   ├── INRBRL_synthetic_raw.csv
│   │   ├── USDINR_hourly_raw.csv
│   │   ├── USDBRL_hourly_raw.csv
│   │   └── india_repo_rate_raw.csv
│   └── processed/                    # Clean, feature-engineered, split
│       ├── USDINR_daily_clean.csv
│       ├── USDBRL_daily_clean.csv
│       ├── INRBRL_synthetic_clean.csv
│       ├── USDINR_hourly_clean.csv
│       ├── USDBRL_hourly_clean.csv
│       ├── feature_matrix_daily.csv
│       ├── feature_matrix_hourly.csv
│       ├── train_features.csv        # 60% — model training
│       ├── val_features.csv          # 20% — hyperparameter tuning
│       └── test_features.csv         # 20% — LOCKED until Month 3
│
├── notebooks/
│   ├── 01_data_download.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_spike_encoding.ipynb
│   ├── 05_spikingjelly_intro.ipynb
│   ├── 06_feature_engineering.ipynb
│   ├── 06b_feature_engineering_hourly.ipynb
│   └── 07_lstm_baseline_prep.ipynb
│
├── outputs/                          # All saved plots
│   ├── plot_inrbrl_history.png
│   ├── plot_returns_distribution.png
│   ├── plot_volatility.png
│   ├── plot_correlation_heatmap.png
│   ├── plot_spike_encoding_demo.png
│   ├── plot_threshold_comparison.png
│   ├── plot_lif_demo.png
│   ├── plot_lif_tau_comparison.png
│   ├── plot_lif_threshold_comparison.png
│   ├── plot_feature_matrix.png
│   ├── plot_data_split.png
│   └── plot_hourly_spike_train.png
│
└── docs/
    ├── daily_log.md                  # Entry for every working day
    ├── spike_concept_note.md         # LIF / rate coding explanation
    ├── data_quality_report.md        # Dataset audit (Week 2)
    ├── fee_research.md               # Real bank fee data + cost formula
    └── project_motivation.md         # One-page thesis motivation
    └── policy_notes.md               # Structured extraction from two policy documents 
```

---

## Environment Setup
```bash
# Step 1 — Create conda environment
conda create -n brics_snn python=3.10
conda activate brics_snn

# Step 2 — Install all dependencies
pip install \
    yfinance==1.2.0 \
    pandas==2.3.3 \
    numpy==2.2.6 \
    matplotlib==3.10.8 \
    seaborn==0.13.2 \
    scipy==1.15.3 \
    spikingjelly==0.0.0.0.14 \
    torch==2.10.0+cpu \
    tensorflow==2.21.0 \
    pandas_datareader==0.10.0 \
    scikit-learn==[X.X.X]

# Step 3 — Launch Jupyter
jupyter notebook
```

> Replace `[X.X.X]` with the versions printed by the version collector script above.
> tensorflow is optional if you chose PyTorch for the LSTM baseline.

---

## Current Progress

### Month 1 — Data & Foundations ✅
- [x] USD/INR and USD/BRL daily data downloaded (5 years)
- [x] USD/INR and USD/BRL hourly data downloaded (2 years)
- [x] Synthetic INR/BRL cross rate derived (USDINR / USDBRL)
- [x] India macro data fetched from FRED
- [x] Daily data cleaned (gaps filled, outliers flagged)
- [x] Hourly data cleaned (UTC normalized, gaps filled)
- [x] EDA complete — 4 thesis-ready plots saved
- [x] Spike encoding defined (rate coding, threshold=0.5% daily / 0.1% hourly)
- [x] SpikingJelly installed and LIF neuron verified
- [x] Daily feature matrix built (9 features + target)
- [x] Hourly feature matrix built (9 features + target)
- [x] Chronological train/val/test split (60/20/20)
- [x] LSTM baseline architecture designed
- [x] Bank fee research completed (₹42,080 current vs ₹1,180 proposed)
- [x] Policy alignment documented (RBI CBDC, BIS mBridge)


---

## Key Research Numbers

| Metric | Value | Source |
|---|---|---|
| Current transaction cost (₹10L) | ₹42,080 (4.21%) | SBI/HDFC fee research |
| Proposed transaction cost (₹10L) | ₹1,180 (0.12%) | 0.1% assumption + GST |
| Cost reduction | ~97% | Modelled |
| Current settlement time | T+2 to T+3 | SWIFT documentation |
| Proposed settlement time | T+0 | Simulation target |
| Daily spike rate | ~18–22% | threshold=0.5% |
| Hourly spike rate | ~18% | threshold=0.2% |

---

## Customer Profile

An Indian textile exporter in Surat invoicing a São Paulo importer in BRL
currently loses ₹42,080 on every ₹10,00,000 transaction — over 4% of
invoice value — due to mandatory INR→USD→BRL routing through correspondent
banks, plus 2–3 business days of settlement delay that locks up working
capital. With a direct INR/BRL settlement system powered by SNN-based
liquidity prediction, this exporter would pay approximately ₹1,180 per
₹10 lakh transaction — saving ₹40,900 per trade, or ₹2,04,500 annually
at 5 transactions per month.

---

## Academic Context

- **Policy alignment:** RBI CBDC Concept Note (Oct 2022), BIS mBridge MVP (Jun 2024)
- **Research gap:** mBridge and e₹-W provide settlement infrastructure;
  no existing system provides an ML intelligence layer for optimal
  settlement timing — this project addresses that gap
- **Methodology:** Synthetic cross-rate data, rate-coded spike encoding,
  LIF neurons, surrogate gradient training, binary direction classification
- **Honest limitation:** Synthetic data via USD intermediation; results
  are indicative, not production benchmarks

---

*Last updated: Day 19 — end of Week 3*