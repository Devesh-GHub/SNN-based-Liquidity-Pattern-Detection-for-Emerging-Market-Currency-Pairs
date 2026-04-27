"""
dashboard/config.py
===================
All configuration constants for the BRICS SNN dashboard.
Change API_URL here to switch between local and Render deployment.
"""

import os

# ── API connection ────────────────────────────────────────────────────
API_URL = os.environ.get(
    "BRICS_API_URL",
    "https://snn-based-liquidity-pattern-detection-i1t5.onrender.com"   # override with Render URL for production
)
API_TIMEOUT_S   = 60     # seconds — allow for Render cold start
HEALTH_POLL_S   = 30     # how often dashboard checks /health

# ── SNN decision thresholds (must match api/predictor.py) ────────────
PROB_THRESHOLD  = 0.70
RATE_THRESHOLD  = 0.10

# ── Default transaction parameters ────────────────────────────────────
DEFAULT_AMOUNT     = 1_000_000    # ₹10 lakh
MIN_AMOUNT         = 10_000       # ₹10 thousand
MAX_AMOUNT         = 50_000_000   # ₹5 crore
DEFAULT_MONTHLY_TX = 5

# ── Demo price sequence (last 10 INR/BRL closes — update periodically)
DEMO_PRICE_SEQUENCE = [
    16.42, 16.38, 16.51, 16.45, 16.60,
    16.53, 16.48, 16.55, 16.61, 16.57,
]

# ── Colour palette ────────────────────────────────────────────────────
COLOR_RED    = "#e74c3c"    # USD route (expensive)
COLOR_GREEN  = "#2ecc71"    # SNN route (cheap)
COLOR_BLUE   = "#3498db"    # neutral / info
COLOR_ORANGE = "#f39c12"    # warning / caution
COLOR_GREY   = "#95a5a6"    # muted / secondary

# ── Static placeholder data (used on Day 1 before API wiring) ─────────
PLACEHOLDER_DATA = {
    "direction"               : "UP",
    "confidence"              : 0.24,
    "spike_rate"              : 0.478,
    "recommendation"          : "USD_FALLBACK",
    "usd_route_cost"          : 36180.0,
    "snn_route_cost"          : 1236.0,
    "savings_inr"             : 34944.0,
    "savings_percentage"      : 96.58,
    "settlement_time_current" : "T+2 to T+3",
    "settlement_time_proposed": "T+0",
    "usd_route": {
        "total_cost_inr"  : 36180.0,
        "cost_percentage" : 3.618,
        "settlement_days" : "T+2 to T+3",
        "settlement_hours": 48.0,
        "steps"           : ["INR → USD (SBI FxOut)",
                             "USD → BRL (Brazilian correspondent)"],
        "itemised": {
            "inr_usd_spread_inr"   : 20000.0,
            "swift_flat_fee_inr"   : 1000.0,
            "swift_gst_inr"        : 180.0,
            "correspondent_fee_inr": 2100.0,
            "brl_spread_inr"       : 15000.0,
            "iof_tax_inr"          : 3800.0,
        },
    },
    "snn_route": {
        "total_cost_inr"  : 1236.0,
        "cost_percentage" : 0.124,
        "settlement_days" : "T+0",
        "settlement_hours": 0.08,
        "steps"           : ["INR → BRL (direct, SNN-powered)"],
        "itemised": {
            "settlement_fee_inr": 1000.0,
            "flat_fee_inr"      : 200.0,
            "flat_gst_inr"      : 36.0,
        },
    },
    "savings": {
        "amount_inr"          : 34944.0,
        "amount_formatted"    : "₹34,944.00",
        "percentage"          : 96.58,
        "latency_saving_hours": 47.92,
        "latency_saving_days" : 2.0,
        "annual_estimate_inr" : 2096640.0,
        "annual_formatted"    : "₹20,96,640.00",
    },
    "snn_prediction": {
        "probability"         : 0.62,
        "direction"           : "UP",
        "confidence"          : 0.24,
        "spike_rate"          : 0.478,
        "spike_rate_pct"      : 47.8,
        "recommendation"      : "USD_FALLBACK",
        "recommendation_text" : "Use USD route — model confidence below threshold",
        "prob_threshold_used" : 0.70,
        "rate_threshold_used" : 0.10,
    },
    "assumptions": [
        "0.1% direct settlement fee assumed per BIS DLT benchmark (2022)",
        "USD/INR rate of ₹84 used for correspondent fee conversion",
        "Annual saving assumes 5 transactions per month",
        "Predictions from validation-set model (AUC=0.555)",
    ],
}