# BRICS SNN Settlement API — Documentation
**Version:** 1.0.0  
**Base URL (local):** `http://localhost:8000`  
**Base URL (live):** `https://snn-based-liquidity-pattern-detection-i1t5.onrender.com/`  
**Interactive docs:** `{BASE_URL}/docs`  
**Author:** [Your name]

---

## Overview

This API serves a trained Spiking Neural Network (SNN) that predicts
optimal routing for INR/BRL foreign exchange transactions. Given the
last 10 daily closing prices of the synthetic INR/BRL rate, it returns:

- A directional prediction (rate going UP or DOWN tomorrow)
- A settlement recommendation (DIRECT or USD_FALLBACK)
- An itemised cost comparison between the current SWIFT route
  and the proposed direct SNN-powered route

**One important limitation to state upfront:** predictions are based
on a model trained on synthetic data (AUC=0.555). This is a research
prototype, not a production trading system.

---

## Quick Start

### 1. Start the server
```bash
# Clone the repo
git clone https://github.com/Devesh-GHub/SNN-based-Liquidity-Pattern-Detection-for-Emerging-Market-Currency-Pairs
cd brics-snn-api
pip install -r requirements.txt

# Start
uvicorn api.main:app --reload

# Verify
curl http://localhost:8000/health
```

### 2. Make your first prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "price_sequence": [16.2, 16.3, 16.1, 16.4, 16.5,
                       16.3, 16.6, 16.4, 16.7, 16.5],
    "transaction_amount_inr": 1000000.0,
    "monthly_tx_count": 5
  }'
```

### 3. Call from Python
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "price_sequence"        : [16.2, 16.3, 16.1, 16.4, 16.5,
                                   16.3, 16.6, 16.4, 16.7, 16.5],
        "transaction_amount_inr": 1_000_000,
        "monthly_tx_count"      : 5,
    }
)
result = response.json()
print(f"Recommendation : {result['recommendation']}")
print(f"Saving         : ₹{result['savings_inr']:,.0f}")
print(f"Spike rate     : {result['spike_rate']:.2%}")
```

---

## Endpoints

---

### `GET /`
Root info — confirms the API is running.

**Response**
```json
{
  "name"         : "BRICS SNN Liquidity API",
  "version"      : "1.0.0",
  "status"       : "running",
  "model_loaded" : true,
  "uptime_s"     : 142.3,
  "endpoints"    : { ... }
}
```

---

### `GET /health`
Model health check. Called by the dashboard every 30 seconds.

**Response — 200 OK (model loaded)**
```json
{
  "status"         : "healthy",
  "model_loaded"   : true,
  "model_version"  : "BRICSLiquiditySNN v1.0",
  "val_auc"        : 0.555,
  "uptime_seconds" : 142.3,
  "timestamp"      : "2024-03-15T10:30:00Z"
}
```

**Response — 503 Service Unavailable (model not loaded)**
```json
{
  "detail": "SNN model not loaded. Ensure outputs/snn_model_best.pth exists."
}
```

**When this fails:**
The model files (`snn_model_best.pth`, `scaler.pkl`, `snn_config.json`)
are missing from `outputs/`. Run the training notebooks first or
re-download the model files.

---

### `POST /predict`
**Core endpoint.** Runs SNN inference and returns a settlement
recommendation with full cost breakdown.

#### Request Body

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `price_sequence` | list[float] | ✅ Yes | — | Last 10 daily INR/BRL closing rates, oldest first |
| `transaction_amount_inr` | float | No | 1,000,000 | Transaction value in INR. Range: ₹1,000 – ₹10,00,00,000 |
| `monthly_tx_count` | int | No | 5 | Transactions per month for annual saving projection |

**`price_sequence` rules:**
- Must have **exactly 10 values** — not 9, not 11
- All values must be **positive**
- Values must be in range **[5, 100]** — the typical INR/BRL range
- Order: **oldest first, newest last**
- If you accidentally send USD/INR rates (~83), the validator will reject them

**Example request**
```json
{
  "price_sequence": [16.42, 16.38, 16.51, 16.45, 16.60,
                     16.53, 16.48, 16.55, 16.61, 16.57],
  "transaction_amount_inr": 1000000.0,
  "monthly_tx_count": 5
}
```

#### Response Fields

**Top-level fields (most useful for dashboard)**

| Field | Type | Description |
|---|---|---|
| `direction` | `"UP"` or `"DOWN"` | SNN's predicted direction for tomorrow's INR/BRL rate |
| `confidence` | float [0,1] | How confident the model is. 0 = uncertain, 1 = fully confident |
| `spike_rate` | float [0,1] | Mean fraction of LIF neurons that fired during inference |
| `recommendation` | `"DIRECT"` or `"USD_FALLBACK"` | Settlement routing recommendation |
| `usd_route_cost` | float | Total cost via current USD SWIFT route (INR) |
| `snn_route_cost` | float | Total cost via proposed direct SNN route (INR) |
| `savings_inr` | float | Saving per transaction (INR) |
| `savings_percentage` | float | Saving as % of USD route cost |
| `settlement_time_current` | str | e.g. `"T+2 to T+3"` |
| `settlement_time_proposed` | str | `"T+0"` |

**`usd_route` — itemised USD cost breakdown**

| Sub-field | Description |
|---|---|
| `total_cost_inr` | Total cost including all components |
| `cost_percentage` | Total cost as % of transaction |
| `settlement_days` | `"T+2 to T+3"` |
| `settlement_hours` | 48 (hours) |
| `steps` | `["INR → USD (SBI FxOut)", "USD → BRL (Brazilian correspondent)"]` |

**`snn_route` — itemised SNN cost breakdown**

| Sub-field | Description |
|---|---|
| `total_cost_inr` | Total cost (settlement fee + flat fee + GST) |
| `cost_percentage` | Total cost as % of transaction |
| `settlement_days` | `"T+0"` |
| `settlement_hours` | 0.08 (≈5 minutes) |
| `steps` | `["INR → BRL (direct, SNN-powered)"]` |

**`savings` — saving detail**

| Sub-field | Description |
|---|---|
| `amount_inr` | Saving per transaction (INR) |
| `amount_formatted` | e.g. `"₹34,564.00"` |
| `percentage` | Saving as % of USD route cost |
| `latency_saving_days` | Days saved on settlement time (≈2.0) |
| `annual_estimate_inr` | Projected annual saving at `monthly_tx_count` rate |
| `annual_formatted` | e.g. `"₹20,73,840.00"` |

**`snn_prediction` — raw model output**

| Sub-field | Description |
|---|---|
| `probability` | Raw sigmoid output [0,1] — P(rate goes UP) |
| `direction` | `"UP"` or `"DOWN"` |
| `confidence` | `abs(probability - 0.5) × 2` — how far from uncertain |
| `spike_rate` | Mean LIF spike rate across both layers [0,1] |
| `spike_rate_pct` | Spike rate as percentage |
| `recommendation` | `"DIRECT"` or `"USD_FALLBACK"` |
| `recommendation_text` | Human-readable explanation |
| `prob_threshold_used` | 0.70 — minimum probability for DIRECT |
| `rate_threshold_used` | 0.10 — minimum spike rate for DIRECT |

**`assumptions`** — list of honest caveats always included in the response:
- "0.1% direct settlement fee assumed per BIS DLT benchmark (2022)"
- "USD/INR rate of ₹84 used for correspondent fee conversion"
- "Predictions from validation-set model (AUC=0.555)"

**Full response example**
```json
{
  "direction"               : "UP",
  "confidence"              : 0.1200,
  "spike_rate"              : 0.4780,
  "recommendation"          : "USD_FALLBACK",
  "usd_route_cost"          : 36180.00,
  "snn_route_cost"          : 1236.00,
  "savings_inr"             : 34944.00,
  "savings_percentage"      : 96.58,
  "settlement_time_current" : "T+2 to T+3",
  "settlement_time_proposed": "T+0",
  "usd_route": {
    "total_cost_inr"  : 36180.00,
    "cost_percentage" : 3.6180,
    "settlement_days" : "T+2 to T+3",
    "settlement_hours": 48.0,
    "steps"           : ["INR → USD (SBI FxOut)",
                         "USD → BRL (Brazilian correspondent)"]
  },
  "snn_route": {
    "total_cost_inr"  : 1236.00,
    "cost_percentage" : 0.1236,
    "settlement_days" : "T+0",
    "settlement_hours": 0.08,
    "steps"           : ["INR → BRL (direct, SNN-powered)"]
  },
  "savings": {
    "amount_inr"          : 34944.00,
    "amount_formatted"    : "₹34,944.00",
    "percentage"          : 96.58,
    "latency_saving_days" : 2.0,
    "annual_estimate_inr" : 2096640.00,
    "annual_formatted"    : "₹20,96,640.00"
  },
  "snn_prediction": {
    "probability"         : 0.5600,
    "direction"           : "UP",
    "confidence"          : 0.1200,
    "spike_rate"          : 0.4780,
    "spike_rate_pct"      : 47.80,
    "recommendation"      : "USD_FALLBACK",
    "recommendation_text" : "Use USD route. Model probability 0.560 below threshold 0.70.",
    "prob_threshold_used" : 0.70,
    "rate_threshold_used" : 0.10
  },
  "transaction_amount_inr"  : 1000000.0,
  "model_version"           : "BRICSLiquiditySNN v1.0",
  "generated_at"            : "2024-03-15T10:30:00Z",
  "assumptions"             : [
    "0.1% direct settlement fee assumed per BIS DLT benchmark (2022)",
    "USD/INR rate of ₹84 used for correspondent fee conversion",
    "Annual saving assumes constant transaction volume",
    "Predictions from validation-set model (AUC=0.555)"
  ]
}
```

#### Error Responses

| Status | When | Example detail |
|---|---|---|
| 400 | Wrong number of prices | `"price_sequence must have exactly 10 values, got 5"` |
| 400 | Negative price | `"All prices must be positive"` |
| 400 | Out of range prices | `"Prices outside expected INR/BRL range [5, 100]"` |
| 422 | Missing required field | Pydantic validation error with field path |
| 503 | Model not loaded | `"SNN model not loaded"` |
| 500 | Internal error | `"SNN inference failed: ..."` |

---

### `GET /summary?amount=X`
Cost comparison without running inference. Uses the latest cached
prediction. Faster than `/predict` — suitable for a cost calculator
slider in the dashboard.

**Query parameter:** `amount` (float) — transaction value in INR.
Default: 1,000,000. Range: 1,000 – 100,000,000.

**Example**
```bash
curl "http://localhost:8000/summary?amount=500000"
```

**Response**
```json
{
  "transaction_amount_inr"  : 500000.0,
  "usd_route_cost"          : 18090.0,
  "snn_route_cost"          : 736.0,
  "savings_inr"             : 17354.0,
  "savings_percentage"      : 95.94,
  "annual_saving_estimate"  : 1041240.0,
  "settlement_time_current" : "T+2 to T+3",
  "settlement_time_proposed": "T+0",
  "last_recommendation"     : "USD_FALLBACK",
  "generated_at"            : "2024-03-15T10:30:00Z"
}
```

---

## Decision Logic

The `/predict` endpoint applies this routing rule:
IF  probability >= 0.70   (SNN confident in favourable rate)
AND spike_rate >= 0.10    (sufficient signal richness)
THEN  recommendation = "DIRECT"
ELSE  recommendation = "USD_FALLBACK"

**Why both conditions?**
- `probability >= 0.70` alone would route too many uncertain days
- `spike_rate >= 0.10` acts as a confidence filter: if very few
  neurons fired during the window, the model had little signal to
  work with and its probability output is unreliable
- Together they produced 51.4% DIRECT routing accuracy in the
  354-day backtest (vs 50% random baseline)

**Conservative by design:** Only 9.9% of days are routed DIRECT.
The majority (90.1%) use the established SWIFT route, limiting
exposure to model error.

---

## Fee Structure Used

All cost calculations use real bank fee data sourced from SBI FxOut
FAQ and HDFC remittance fee pages (accessed 2024).

**USD route components (per ₹10,00,000 transaction)**

| Component | Value | Source |
|---|---|---|
| INR→USD TT spread | 2.0% = ₹20,000 | SBI TT selling rate markup |
| SWIFT flat fee | ₹1,000 | SBI/HDFC outward remittance fee |
| GST on flat fee | ₹180 (18%) | Government of India GST |
| Correspondent bank | $25 = ₹2,100 | Standard SWIFT correspondent |
| USD→BRL spread | 1.5% = ₹15,000 | Brazilian correspondent markup |
| Brazilian IOF tax | 0.38% = ₹3,800 | Brazil financial operations tax |
| **Total** | **₹42,080 (4.21%)** | |

**SNN direct route components (per ₹10,00,000 transaction)**

| Component | Value | Note |
|---|---|---|
| Settlement fee | 0.1% = ₹1,000 | ⚠️ Assumed (BIS DLT benchmark) |
| Flat fee | ₹200 | Infrastructure fee |
| GST on flat fee | ₹36 | 18% on ₹200 |
| **Total** | **₹1,236 (0.12%)** | |

> ⚠️ The 0.1% direct settlement fee is an **assumption** based on
> BIS DLT settlement cost benchmarks (2022). It has not been validated
> against any operational system. All savings figures are indicative
> under this assumption.

---

## Running the Test Suite

```bash
# Start API first (Terminal 1)
uvicorn api.main:app --reload

# Run tests (Terminal 2)
python tests/test_api.py

# Run against live Render URL
API_URL=https://brics-snn-api.onrender.com python tests/test_api.py
```

Expected output:
✅ All 8 tests passed.
This output is your thesis proof-of-concept evidence.


## For New Developers — Getting Started Checklist
□ Clone the repo
□ Create conda env: conda create -n brics_snn python=3.10
□ Install: pip install -r requirements.txt
□ Verify model files exist in outputs/:
   outputs/snn_model_best.pth
   outputs/snn_config.json
   outputs/scaler.pkl
□ Start API: uvicorn api.main:app --reload
□ Open: http://localhost:8000/docs
□ Click "POST /predict" → "Try it out" → "Execute"
□ Run tests: python tests/test_api.py
□ All 8 tests pass → you are set up correctly

If any model file is missing, re-run the training notebooks in order:
`09_snn_model.ipynb` → `10_snn_experiments.ipynb`

---


## Glossary

| Term | Meaning |
|---|---|
| LIF | Leaky Integrate-and-Fire — the neuron type used in the SNN |
| Spike rate | Fraction of LIF neurons that fired during a forward pass |
| Spike signal | 1 if \|daily_return\| > 0.3%, else 0 |
| ISI | Inter-spike interval — days since last spike event |
| AUC | Area Under the ROC Curve — model quality metric (0.5 = random) |
| Youden J | Optimal threshold = argmax(TPR − FPR) on ROC curve |
| SynOps | Synaptic operations — computational unit for SNN energy |
| DIRECT | Route transaction through proposed INR/BRL channel |
| USD_FALLBACK | Route transaction through conventional SWIFT USD route |
| T+0 / T+2 | Settlement time: T+0 = same day, T+2 = 2 business days |
| pos_weight | Loss function weight to correct class imbalance (1.78) |
| surrogate gradient | Differentiable approximation for spike backprop |