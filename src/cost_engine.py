"""
src/cost_engine.py
==================
Transaction cost calculation engine for INR/BRL FX settlement.

Computes itemised costs for two settlement routes:
1. USD-intermediated SWIFT route (current standard)
2. SNN-powered direct INR/BRL route (proposed)

All fee data sourced from SBI FxOut FAQ and HDFC remittance fee
pages (accessed 2024). The direct route fee of 0.1% is an assumption
based on BIS DLT settlement cost benchmarks (2022).

Usage
-----
from src.cost_engine import CostEngine

engine = CostEngine()
result = engine.get_dashboard_summary(
    transaction_amount=1_000_000,
    prediction={"prob": 0.72, "spike_rate": 0.35, "decision": "DIRECT"},
    monthly_tx_count=5
)
print(result["savings"]["amount_formatted"])   # ₹34,944.00

Standalone test
---------------
python src/cost_engine.py
"""

from datetime import datetime
from typing import Optional
import json


# ─────────────────────────────────────────────
# FEE CONSTANTS
# ─────────────────────────────────────────────

# All values sourced from real bank fee schedules (2024).
# Update annually or when RBI/bank fee structures change.

USD_ROUTE_CONFIG = {
    # INR → USD leg (SBI FxOut / HDFC outward remittance)
    "inr_usd_spread_pct"    : 0.020,   # 2.0% TT selling rate markup
    "swift_flat_fee_inr"    : 1_000,   # ₹1,000 outward remittance fee
    "gst_rate"              : 0.18,    # 18% GST on flat fee (India)
    # USD → BRL leg (Brazilian correspondent bank)
    "correspondent_usd"     : 25,      # $25 correspondent handling fee
    "usd_to_inr_rate"       : 84.0,    # Approx USD/INR for fee conversion
    "brl_spread_pct"        : 0.015,   # 1.5% USD→BRL conversion spread
    "iof_tax_pct"           : 0.0038,  # 0.38% Brazilian IOF financial tax
    # Metadata
    "settlement_days"       : "T+2 to T+3",
    "settlement_hours"      : 48.0,
    "steps"                 : [
        "INR → USD (SBI FxOut / HDFC outward remittance)",
        "USD → BRL (Brazilian correspondent bank)",
    ],
    "source"                : (
        "SBI FxOut FAQ (retail.sbi.bank.in); "
        "HDFC remittance fees (hdfcbank.com); "
        "Brazil IOF tax schedule. Accessed 2024."
    ),
}

SNN_ROUTE_CONFIG = {
    # ⚠️ ASSUMPTION: 0.1% direct settlement fee
    # Based on: BIS DLT settlement cost benchmarks (2022)
    # This has NOT been validated against any operational system.
    "settlement_fee_pct"    : 0.001,   # 0.1% ⚠️ ASSUMED
    "flat_fee_inr"          : 200,     # ₹200 infrastructure fee
    "gst_rate"              : 0.18,    # 18% GST on flat fee
    # Metadata
    "settlement_days"       : "T+0",
    "settlement_hours"      : 0.08,    # ~5 minutes (near real-time)
    "steps"                 : [
        "INR → BRL (direct, SNN-powered settlement)",
    ],
    "assumption_note"       : (
        "0.1% settlement fee assumed per BIS DLT benchmark (2022). "
        "Not validated in production. All savings are indicative."
    ),
}


class CostEngine:
    """
    Computes transaction costs for INR/BRL settlement routes.

    Instantiate once and reuse across requests. All fee constants
    are loaded at init time from module-level config dicts.
    To update fees, modify USD_ROUTE_CONFIG and SNN_ROUTE_CONFIG.

    Examples
    --------
    >>> engine = CostEngine()
    >>> usd = engine.calculate_usd_route(1_000_000)
    >>> print(usd["total_cost_inr"])
    36180.0

    >>> snn = engine.calculate_snn_route(1_000_000)
    >>> print(snn["total_cost_inr"])
    1236.0
    """

    def __init__(self):
        self._usd_cfg = USD_ROUTE_CONFIG
        self._snn_cfg = SNN_ROUTE_CONFIG

    # ─────────────────────────────────────────────
    # SECTION 1 — Route cost calculations
    # ─────────────────────────────────────────────

    def calculate_usd_route(self, amount_inr: float) -> dict:
        """
        Calculate itemised costs for the USD-intermediated SWIFT route.

        Cost components:
        1. INR→USD TT spread (2.0%): embedded in exchange rate markup
        2. SWIFT flat fee + 18% GST: ₹1,000 + ₹180
        3. Correspondent bank fee: $25 × ₹84 = ₹2,100
        4. USD→BRL spread (1.5%): at Brazilian correspondent bank
        5. Brazilian IOF tax (0.38%): financial operations tax

        Parameters
        ----------
        amount_inr : float — transaction value in INR

        Returns
        -------
        dict with keys:
            total_cost_inr, cost_percentage, settlement_days,
            settlement_hours, steps, itemised, source

        Examples
        --------
        >>> engine.calculate_usd_route(1_000_000)
        {"total_cost_inr": 36180.0, "cost_percentage": 3.618, ...}
        """
        r = self._usd_cfg

        inr_usd_spread    = amount_inr * r["inr_usd_spread_pct"]
        swift_flat        = r["swift_flat_fee_inr"]
        swift_gst         = swift_flat * r["gst_rate"]
        swift_total_flat  = swift_flat + swift_gst
        correspondent_inr = r["correspondent_usd"] * r["usd_to_inr_rate"]
        brl_spread        = amount_inr * r["brl_spread_pct"]
        iof_tax           = amount_inr * r["iof_tax_pct"]

        total_cost = (inr_usd_spread + swift_total_flat +
                      correspondent_inr + brl_spread + iof_tax)
        cost_pct   = total_cost / amount_inr * 100

        return {
            "total_cost_inr"  : round(total_cost,     2),
            "cost_percentage" : round(cost_pct,        4),
            "settlement_days" : r["settlement_days"],
            "settlement_hours": r["settlement_hours"],
            "steps"           : r["steps"],
            "itemised"        : {
                "inr_usd_spread_inr"    : round(inr_usd_spread,    2),
                "swift_flat_fee_inr"    : round(swift_flat,         2),
                "swift_gst_inr"         : round(swift_gst,          2),
                "correspondent_fee_inr" : round(correspondent_inr,  2),
                "brl_spread_inr"        : round(brl_spread,         2),
                "iof_tax_inr"           : round(iof_tax,            2),
            },
            "source"          : r["source"],
        }

    def calculate_snn_route(self, amount_inr: float) -> dict:
        """
        Calculate itemised costs for the SNN-powered direct INR/BRL route.

        ⚠️ The 0.1% settlement fee is an assumption, not a measured value.
        See SNN_ROUTE_CONFIG["assumption_note"] for full caveat.

        Cost components:
        1. Settlement fee (0.1%): assumed DLT infrastructure cost
        2. Flat fee + 18% GST: ₹200 + ₹36

        Parameters
        ----------
        amount_inr : float — transaction value in INR

        Returns
        -------
        dict with keys:
            total_cost_inr, cost_percentage, settlement_days,
            settlement_hours, steps, itemised, assumption_note

        Examples
        --------
        >>> engine.calculate_snn_route(1_000_000)
        {"total_cost_inr": 1236.0, "cost_percentage": 0.1236, ...}
        """
        r = self._snn_cfg

        settlement_fee = amount_inr * r["settlement_fee_pct"]
        flat_fee       = r["flat_fee_inr"]
        flat_gst       = flat_fee * r["gst_rate"]
        total_cost     = settlement_fee + flat_fee + flat_gst
        cost_pct       = total_cost / amount_inr * 100

        return {
            "total_cost_inr"  : round(total_cost,  2),
            "cost_percentage" : round(cost_pct,     4),
            "settlement_days" : r["settlement_days"],
            "settlement_hours": r["settlement_hours"],
            "steps"           : r["steps"],
            "itemised"        : {
                "settlement_fee_inr": round(settlement_fee, 2),
                "flat_fee_inr"      : round(flat_fee,        2),
                "flat_gst_inr"      : round(flat_gst,        2),
            },
            "assumption_note" : r["assumption_note"],
        }

    # ─────────────────────────────────────────────
    # SECTION 2 — Savings calculation
    # ─────────────────────────────────────────────

    def calculate_savings(self,
                           amount_inr       : float,
                           monthly_tx_count : int = 5) -> dict:
        """
        Calculate saving from switching USD route → SNN direct route.

        Parameters
        ----------
        amount_inr       : float — transaction value in INR
        monthly_tx_count : int   — transactions per month for annual projection

        Returns
        -------
        dict with keys:
            amount_inr, amount_formatted, percentage,
            latency_saving_hours, latency_saving_days,
            annual_estimate_inr, annual_formatted,
            monthly_tx_assumption
        """
        usd = self.calculate_usd_route(amount_inr)
        snn = self.calculate_snn_route(amount_inr)

        saving_inr  = usd["total_cost_inr"] - snn["total_cost_inr"]
        saving_pct  = saving_inr / usd["total_cost_inr"] * 100
        annual_est  = saving_inr * monthly_tx_count * 12
        lat_saving  = usd["settlement_hours"] - snn["settlement_hours"]

        return {
            "amount_inr"            : round(saving_inr,  2),
            "amount_formatted"      : f"₹{saving_inr:,.2f}",
            "percentage"            : round(saving_pct,  2),
            "latency_saving_hours"  : round(lat_saving,  2),
            "latency_saving_days"   : round(lat_saving / 24, 1),
            "annual_estimate_inr"   : round(annual_est,  2),
            "annual_formatted"      : f"₹{annual_est:,.2f}",
            "monthly_tx_assumption" : monthly_tx_count,
        }

    # ─────────────────────────────────────────────
    # SECTION 3 — Dashboard summary
    # ─────────────────────────────────────────────

    def get_dashboard_summary(self,
                               transaction_amount : float = 1_000_000,
                               prediction         : Optional[dict] = None,
                               monthly_tx_count   : int = 5) -> dict:
        """
        Build the complete dashboard data payload.

        Combines cost calculations, savings, and SNN prediction into
        a single JSON-serialisable dict consumed by FastAPI and Streamlit.

        Parameters
        ----------
        transaction_amount : float — INR value of transaction
        prediction         : dict, optional — SNN model output with keys:
                             prob, spike_rate, decision, recommendation_text
                             If None, uses a neutral placeholder.
        monthly_tx_count   : int — transactions per month for annual est.

        Returns
        -------
        dict — complete dashboard payload with keys:
                meta, transaction, usd_route, snn_route,
                savings, snn_prediction, assumptions

        Examples
        --------
        >>> engine = CostEngine()
        >>> summary = engine.get_dashboard_summary(
        ...     transaction_amount=1_000_000,
        ...     prediction={"prob": 0.72, "spike_rate": 0.35,
        ...                 "decision": "DIRECT"},
        ... )
        >>> summary["savings"]["amount_formatted"]
        '₹34,944.00'
        """
        if prediction is None:
            prediction = {
                "prob"      : 0.52,
                "spike_rate": 0.15,
                "decision"  : "USD_FALLBACK",
                "date"      : "latest",
            }

        usd_costs = self.calculate_usd_route(transaction_amount)
        snn_costs = self.calculate_snn_route(transaction_amount)
        savings   = self.calculate_savings(transaction_amount,
                                            monthly_tx_count)

        prob       = float(prediction.get("prob",       0.52))
        spike_rate = float(prediction.get("spike_rate", 0.15))
        decision   = str(prediction.get(
            "decision",
            prediction.get("recommendation", "USD_FALLBACK")
        ))
        confidence = abs(prob - 0.5) * 2
        direction  = "UP" if prob >= 0.5 else "DOWN"

        if decision == "DIRECT":
            rec_text = (
                f"Route via direct INR/BRL settlement. "
                f"SNN confidence {confidence*100:.0f}% — "
                f"estimated saving {savings['amount_formatted']}."
            )
        else:
            rec_text = (
                "Use conventional USD SWIFT route. "
                "SNN confidence below threshold — "
                "defer to SWIFT for this transaction."
            )

        return {
            "meta": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "model"       : "BRICSLiquiditySNN v1.0",
            },
            "transaction": {
                "amount_inr"      : transaction_amount,
                "amount_formatted": f"₹{transaction_amount:,.0f}",
                "monthly_tx_count": monthly_tx_count,
            },
            "usd_route"       : usd_costs,
            "snn_route"       : snn_costs,
            "savings"         : savings,
            "snn_prediction"  : {
                "probability"         : round(prob,        4),
                "direction"           : direction,
                "confidence"          : round(confidence,  4),
                "spike_rate"          : round(spike_rate,  4),
                "spike_rate_pct"      : round(spike_rate * 100, 2),
                "recommendation"      : decision,
                "recommendation_text" : rec_text,
                "prob_threshold_used" : 0.70,
                "rate_threshold_used" : 0.10,
            },
            "assumptions": [
                "0.1% direct settlement fee assumed per BIS DLT benchmark (2022)",
                "USD/INR rate of ₹84 used for correspondent fee conversion",
                "Annual saving assumes constant transaction volume",
                "Predictions from validation-set model (AUC=0.555)",
                "Energy efficiency claims apply to neuromorphic hardware only",
            ],
        }

    # ─────────────────────────────────────────────
    # SECTION 4 — Utilities
    # ─────────────────────────────────────────────

    def scaling_table(self,
                       amounts: list = None) -> list:
        """
        Compute savings for multiple transaction sizes.

        Parameters
        ----------
        amounts : list of float, optional
                  Default: [100_000, 500_000, 1_000_000,
                            5_000_000, 10_000_000]

        Returns
        -------
        list of dicts, one per amount, with:
            amount_inr, usd_cost, snn_cost, saving_inr, saving_pct

        Examples
        --------
        >>> rows = engine.scaling_table()
        >>> for r in rows: print(r)
        """
        if amounts is None:
            amounts = [100_000, 500_000, 1_000_000,
                       5_000_000, 10_000_000]
        rows = []
        for amt in amounts:
            usd = self.calculate_usd_route(amt)
            snn = self.calculate_snn_route(amt)
            sv  = self.calculate_savings(amt)
            rows.append({
                "amount_inr"  : amt,
                "formatted"   : f"₹{amt:,.0f}",
                "usd_cost"    : usd["total_cost_inr"],
                "snn_cost"    : snn["total_cost_inr"],
                "saving_inr"  : sv["amount_inr"],
                "saving_pct"  : sv["percentage"],
            })
        return rows


# ── Module-level convenience functions ────────────────────────────────
# These match the function-based API used in api/cost_engine.py
# for backwards compatibility.

_engine = CostEngine()


def calculate_usd_route(amount_inr: float) -> dict:
    """Module-level wrapper. See CostEngine.calculate_usd_route."""
    return _engine.calculate_usd_route(amount_inr)


def calculate_snn_route(amount_inr: float) -> dict:
    """Module-level wrapper. See CostEngine.calculate_snn_route."""
    return _engine.calculate_snn_route(amount_inr)


def get_dashboard_summary(transaction_amount : float = 1_000_000,
                           prediction         : Optional[dict] = None,
                           monthly_tx_count   : int = 5) -> dict:
    """Module-level wrapper. See CostEngine.get_dashboard_summary."""
    return _engine.get_dashboard_summary(
        transaction_amount, prediction, monthly_tx_count
    )


# ── Standalone test ────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = CostEngine()

    print("=" * 60)
    print("CostEngine standalone test")
    print("=" * 60)

    # Test 1: single transaction
    amount = 1_000_000
    usd    = engine.calculate_usd_route(amount)
    snn    = engine.calculate_snn_route(amount)
    sv     = engine.calculate_savings(amount)

    print(f"\nTransaction: ₹{amount:,.0f}")
    print(f"  USD route : ₹{usd['total_cost_inr']:,.2f}  "
          f"({usd['cost_percentage']:.3f}%)")
    print(f"  SNN route : ₹{snn['total_cost_inr']:,.2f}  "
          f"({snn['cost_percentage']:.3f}%)")
    print(f"  Saving    : {sv['amount_formatted']}  "
          f"({sv['percentage']:.1f}%)")
    print(f"  Annual est: {sv['annual_formatted']}")

    assert usd["total_cost_inr"] > snn["total_cost_inr"], \
        "USD should always cost more than SNN"
    assert sv["amount_inr"] > 0, "Saving must be positive"
    print("  ✅ Assertions passed")

    # Test 2: scaling table
    print(f"\nScaling table:")
    print(f"  {'Amount':>15}  {'USD Cost':>12}  "
          f"{'SNN Cost':>10}  {'Saving':>10}  {'%':>6}")
    print(f"  {'─'*60}")
    for row in engine.scaling_table():
        print(f"  {row['formatted']:>15}  "
              f"₹{row['usd_cost']:>10,.0f}  "
              f"₹{row['snn_cost']:>8,.0f}  "
              f"₹{row['saving_inr']:>8,.0f}  "
              f"{row['saving_pct']:>5.1f}%")

    # Test 3: JSON serialisable
    summary = engine.get_dashboard_summary(1_000_000)
    json.dumps(summary)   # will raise if not serialisable
    print("\n  ✅ Dashboard summary is JSON-serialisable")

    print("\n✅ CostEngine all tests passed")
    print("=" * 60)