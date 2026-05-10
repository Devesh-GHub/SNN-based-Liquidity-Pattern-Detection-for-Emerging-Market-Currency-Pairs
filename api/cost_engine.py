"""
api/cost_engine.py
==================
Cost calculation engine for the BRICS SNN Settlement API.

Ported from notebooks/13_business_logic.ipynb Section 6.
This module is imported by api/predictor.py and api/main.py.
All monetary values are in INR.
"""

import os
import json
from datetime import datetime

import pandas as pd

_OUTPUTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
)

# ── Fee constants (from Month 1 Day 13 research) ──────────────────────
FEE_CONFIG = {
    "usd_route": {
        "inr_usd_spread_pct" : 0.020,
        "swift_flat_fee_inr" : 1_000,
        "gst_rate"           : 0.18,
        "correspondent_usd"  : 25,
        "usd_to_inr_rate"    : 84.0,
        "brl_spread_pct"     : 0.015,
        "iof_tax_pct"        : 0.0038,
        "settlement_days"    : "T+2 to T+3",
        "settlement_hours"   : 48,
        "steps"              : ["INR → USD (SBI FxOut)",
                                "USD → BRL (Brazilian correspondent)"],
        "source"             : "SBI FxOut FAQ; HDFC remittance page (2024)",
    },
    "snn_route": {
        "settlement_fee_pct" : 0.001,
        "flat_fee_inr"       : 200,
        "gst_rate"           : 0.18,
        "settlement_days"    : "T+0",
        "settlement_hours"   : 0.08,
        "steps"              : ["INR → BRL (direct, SNN-powered)"],
        "assumption_note"    : "0.1% fee assumed per BIS DLT benchmark (2022)",
    },
}


def calculate_usd_route(amount_inr: float) -> dict:
    r = FEE_CONFIG["usd_route"]
    spread        = amount_inr * r["inr_usd_spread_pct"]
    flat          = r["swift_flat_fee_inr"]
    gst           = flat * r["gst_rate"]
    correspondent = r["correspondent_usd"] * r["usd_to_inr_rate"]
    brl_spread    = amount_inr * r["brl_spread_pct"]
    iof           = amount_inr * r["iof_tax_pct"]
    total         = spread + flat + gst + correspondent + brl_spread + iof
    return {
        "total_cost_inr"  : round(total,          2),
        "cost_percentage" : round(total/amount_inr*100, 4),
        "settlement_days" : r["settlement_days"],
        "settlement_hours": r["settlement_hours"],
        "steps"           : r["steps"],
        "itemised"        : {
            "inr_usd_spread_inr"   : round(spread,        2),
            "swift_flat_fee_inr"   : round(flat,           2),
            "swift_gst_inr"        : round(gst,            2),
            "correspondent_fee_inr": round(correspondent,  2),
            "brl_spread_inr"       : round(brl_spread,     2),
            "iof_tax_inr"          : round(iof,            2),
        },
    }


def calculate_snn_route(amount_inr: float) -> dict:
    r = FEE_CONFIG["snn_route"]
    fee   = amount_inr * r["settlement_fee_pct"]
    flat  = r["flat_fee_inr"]
    gst   = flat * r["gst_rate"]
    total = fee + flat + gst
    return {
        "total_cost_inr"  : round(total,          2),
        "cost_percentage" : round(total/amount_inr*100, 4),
        "settlement_days" : r["settlement_days"],
        "settlement_hours": r["settlement_hours"],
        "steps"           : r["steps"],
        "itemised"        : {
            "settlement_fee_inr": round(fee,  2),
            "flat_fee_inr"      : round(flat, 2),
            "flat_gst_inr"      : round(gst,  2),
        },
        "assumption_note" : r["assumption_note"],
    }


def get_dashboard_summary(transaction_amount : float = 1_000_000,
                           prediction         : dict  = None,
                           monthly_tx_count   : int   = 5) -> dict:
    """
    Build complete dashboard data payload. See notebook 13 for full docs.
    """
    if prediction is None:
        prediction = {
            "prob": 0.52, "spike_rate": 0.15,
            "decision": "DIRECT", "date": "latest"
        }

    usd   = calculate_usd_route(transaction_amount)
    snn   = calculate_snn_route(transaction_amount)

    saving_inr = usd["total_cost_inr"] - snn["total_cost_inr"]
    saving_pct = saving_inr / usd["total_cost_inr"] * 100
    annual_est = saving_inr * monthly_tx_count * 12

    prob       = prediction.get("prob", 0.52)
    spike_rate = prediction.get("spike_rate", 0.15)
    decision   = prediction.get("decision", "USD_FALLBACK")
    confidence = abs(prob - 0.5) * 2
    direction  = "UP" if prob >= 0.5 else "DOWN"

    return {
        "meta"            : {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "model"       : "BRICSLiquiditySNN v1.0",
        },
        "transaction"     : {
            "amount_inr"      : transaction_amount,
            "amount_formatted": f"₹{transaction_amount:,.0f}",
        },
        "usd_route"       : usd,
        "snn_route"       : snn,
        "savings"         : {
            "amount_inr"        : round(saving_inr, 2),
            "amount_formatted"  : f"₹{saving_inr:,.2f}",
            "percentage"        : round(saving_pct, 2),
            "latency_saving_hours": 48 - 0.08,
            "latency_saving_days" : round((48 - 0.08) / 24, 1),
            "annual_estimate_inr" : round(annual_est, 2),
            "annual_formatted"    : f"₹{annual_est:,.2f}",
            "monthly_tx_assumption": monthly_tx_count,
        },
        "snn_prediction"  : {
            "probability"        : round(prob,        4),
            "direction"          : direction,
            "confidence"         : round(confidence,  4),
            "spike_rate"         : round(spike_rate,  4),
            "spike_rate_pct"     : round(spike_rate*100, 2),
            "recommendation"     : decision,
            "recommendation_text": (
                f"Route DIRECT — confidence {confidence*100:.0f}%"
                if decision == "DIRECT"
                else "Use USD route — model confidence below threshold"
            ),
            "prob_threshold_used": 0.08,  # matches optimal_threshold in snn_config.json
            "rate_threshold_used": 0.10,
        },
        "assumptions"     : [
            "0.1% direct settlement fee assumed per BIS DLT benchmark (2022)",
            "USD/INR rate of ₹84 used for correspondent fee conversion",
            "Annual saving assumes constant transaction volume",
            "Predictions from validation-set model (AUC=0.555)",
        ],
    }


def get_liquidity_analysis() -> dict:
    """
    Load and return the full backtest-based liquidity analysis.

    Reads outputs/backtest_results.csv and outputs/backtest_summary.json
    (produced by notebooks/13_business_logic.ipynb) and returns a
    structured payload consumed by GET /liquidity and the dashboard panel.

    Returns
    -------
    dict with keys:
        summary          - aggregate backtest stats from JSON
        timeline         - per-day series (date, decision, savings, cumulative)
        risk_analysis    - right/wrong DIRECT breakdown
        savings_scaling  - per-transaction costs at 3 transaction sizes
    """
    backtest_csv  = os.path.join(_OUTPUTS_DIR, "backtest_results.csv")
    summary_json  = os.path.join(_OUTPUTS_DIR, "backtest_summary.json")

    with open(summary_json) as f:
        summary = json.load(f)

    df = pd.read_csv(backtest_csv, index_col="date", parse_dates=True)
    df = df.sort_index()

    # ── Timeline series ───────────────────────────────────────────────
    timeline = [
        {
            "date"               : str(idx.date()),
            "decision"           : str(row["decision"]),
            "snn_prob"           : round(float(row["snn_prob"]),        4),
            "snn_spike_rate"     : round(float(row["snn_spike_rate"]),  4),
            "saving_day"         : round(float(row["saving_day"]),      2),
            "cumulative_saving"  : round(float(row["cumulative_saving"]),  2),
            "cumulative_baseline": round(float(row["cumulative_baseline"]), 2),
            "cumulative_direct"  : round(float(row["cumulative_direct"]),  2),
            "snn_correct"        : int(row["snn_correct"]),
            "confidence"         : round(float(row["confidence"]),      4),
        }
        for idx, row in df.iterrows()
    ]

    # ── Risk analysis ─────────────────────────────────────────────────
    direct_df    = df[df["decision"] == "DIRECT"]
    right_direct = direct_df[direct_df["snn_correct"] == 1]
    wrong_direct = direct_df[direct_df["snn_correct"] == 0]

    wrong_direct_cost = float(wrong_direct["cost_taken"].sum())   if len(wrong_direct) > 0 else 0.0
    wrong_swift_cost  = float(wrong_direct["cost_baseline"].sum()) if len(wrong_direct) > 0 else 0.0

    risk_analysis = {
        "n_right_direct"       : int(len(right_direct)),
        "n_wrong_direct"       : int(len(wrong_direct)),
        "avg_conf_correct"     : round(float(right_direct["confidence"].mean()), 4) if len(right_direct) > 0 else 0.0,
        "avg_conf_wrong"       : round(float(wrong_direct["confidence"].mean()),  4) if len(wrong_direct) > 0 else 0.0,
        "wrong_direct_cost"    : round(wrong_direct_cost,  2),
        "wrong_swift_cost"     : round(wrong_swift_cost,   2),
        "even_wrong_saved_money": wrong_direct_cost < wrong_swift_cost,
    }

    # ── Savings scaling across three transaction sizes ────────────────
    savings_scaling = []
    for amount, label in [
        (100_000,   "₹1 Lakh"),
        (1_000_000, "₹10 Lakh"),
        (5_000_000, "₹50 Lakh"),
    ]:
        usd    = calculate_usd_route(amount)
        snn    = calculate_snn_route(amount)
        saving = usd["total_cost_inr"] - snn["total_cost_inr"]
        savings_scaling.append({
            "label"      : label,
            "amount"     : float(amount),
            "usd_cost"   : float(usd["total_cost_inr"]),
            "snn_cost"   : float(snn["total_cost_inr"]),
            "saving"     : round(saving, 2),
            "saving_pct" : round(saving / usd["total_cost_inr"] * 100, 1),
        })

    return {
        "summary"        : summary,
        "timeline"       : timeline,
        "risk_analysis"  : risk_analysis,
        "savings_scaling": savings_scaling,
    }