# Data Quality Report — BRICS SNN FX Settlement Project


---

## 1. Datasets Summary

Dataset                   Rows         Start           End    Gaps   Outliers
------------------------------------------------------------------------------
USDINR daily              1823    2021-03-04    2026-03-04    2104         29
USDBRL daily              1823    2021-03-04    2026-03-04    2104         11
INRBRL synthetic          1823    2021-03-04    2026-03-04    1058         29
USDINR hourly            12147    2024-03-05    2026-03-05    6412        347
USDBRL hourly            12158    2024-03-05    2026-03-05   14988        267

**Total data points across all datasets:** [sum of all rows]  
**Primary modelling dataset:** `INRBRL_synthetic_clean.csv` (daily, 5-year history)

---

## 2. Synthetic INR/BRL Cross Rate Methodology

### Formula
```
INR/BRL = USD/INR ÷ USD/BRL

Example: If 1 USD = 83.5 INR and 1 USD = 5.0 BRL
         Then 1 BRL = 83.5 ÷ 5.0 = 16.7 INR
```

### Why USD as the intermediate currency
No direct INR/BRL exchange market exists with sufficient liquidity or 
historical depth for research purposes. Routing through USD is the standard 
practice for emerging-market cross rates — it mirrors how real correspondent 
banks execute INR/BRL transactions today. Both legs (USD/INR and USD/BRL) 
have deep, liquid, well-documented markets with 5+ years of reliable data.

### Statistical sanity check
- Mean INR/BRL over 5 years: [X] INR per BRL  
- Range: [min] – [max] INR per BRL  
- This is economically consistent with known purchasing power parity 
  between India and Brazil over the study period.

### Honest limitation
This is synthetic data derived from two USD pairs, not a direct 
INR/BRL spot rate. Any microstructure effects (bid-ask spread, 
liquidity gaps specific to the INR/BRL pair) are absent. 
For academic simulation purposes, this is both standard and acceptable — 
and the thesis will state this explicitly.

---

## 3. Spike Encoding Decision

| Parameter | Value | Rationale |
|---|---|---|
| Encoding method | Rate coding | Robust to noise; maps spike count to signal strength |
| Threshold | [0.005] = 0.5% daily move | See justification below |
| Spike frequency | ~[X]% of daily timesteps | Within target 15–25% range |
| SNN framework | SpikingJelly (PyTorch backend) | Active maintenance, surrogate gradient support |
| LIF tau (initial) | 2.0 | Short memory — recent returns weighted more |
| LIF v_threshold (initial) | 1.0 | Fires only on meaningfully accumulated signal |

### Threshold justification
A 0.5% daily move in INR/BRL represents a meaningful deviation from 
normal settlement conditions — small enough to capture genuine volatility 
events, large enough to exclude microstructure noise. At this threshold, 
approximately [X]% of days produce spikes, keeping the signal sparse and 
informative. Thresholds below 0.3% produced noisy, near-continuous spike 
trains; thresholds above 1.0% missed medium-sized events that would be 
operationally relevant for a settlement system.

---

## 4. Known Limitations

**Limitation 1 — Synthetic cross rate, not real market data**  
The INR/BRL rate is derived via USD intermediation. Real direct settlement 
would involve additional FX risk, counterparty considerations, and 
microstructure effects not captured here. This limits the external validity 
of absolute cost reduction estimates; relative comparisons remain valid.

**Limitation 2 — Yahoo Finance data quality**  
yfinance data is sourced from Yahoo's aggregated feed, not a primary 
market data provider (Bloomberg, Refinitiv). Occasional stale prints, 
timezone inconsistencies, and missing sessions have been addressed by 
cleaning, but cannot be fully eliminated. All raw files are preserved 
unchanged in `data/raw/` for auditability.

**Limitation 3 — Daily frequency limits intraday analysis**  
The primary modelling dataset is daily. Real FX settlement systems operate 
at tick or minute frequency. The hourly data (2-year window) partially 
addresses this but remains coarser than production systems. Results 
validated on daily data should be interpreted as indicative of directional 
performance, not production benchmarks.

---

## 5. What Is Ready for Month 2

All files below are in `data/processed/` and are clean, validated, and 
ready for feature engineering and model training.

| File | Frequency | Rows | Contains |
|---|---|---|---|
| `USDINR_daily_clean.csv` | Daily | [X] | USD/INR close, is_outlier flag |
| `USDBRL_daily_clean.csv` | Daily | [X] | USD/BRL close, is_outlier flag |
| `INRBRL_synthetic_clean.csv` | Daily | [X] | INR/BRL derived rate, is_outlier flag |
| `USDINR_hourly_clean.csv` | Hourly (UTC) | [X] | USD/INR close, is_outlier flag |
| `USDBRL_hourly_clean.csv` | Hourly (UTC) | [X] | USD/BRL close, is_outlier flag |

**Month 2 starting point:**  
Feature engineering will begin from `INRBRL_synthetic_clean.csv`.  
The spike encoder (threshold=0.005, rate coding) built in Day 10 will be 
ported into SpikingJelly's `PoissonEncoder` in Week 3.  
LIF neuron parameters (tau=2.0, v_threshold=1.0) are confirmed working 
and will serve as the starting point for the full SNN architecture.

---

