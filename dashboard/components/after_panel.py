"""
dashboard/components/after_panel.py
=====================================
The 'After' panel — proposed SNN-powered direct INR/BRL route.
Three fee components only (vs six for USD route).
Assumption note is mandatory — never remove it.
"""

import streamlit as st
import pandas as pd
from dashboard.utils.formatters import fmt_inr


def render_after_panel(transaction_amount: float, snn_route: dict):
    """
    Render the 'After' panel — proposed SNN direct INR/BRL route.

    Parameters
    ----------
    transaction_amount : float — INR value of the transaction
    snn_route          : dict  — snn_route section of API response
    """
    st.markdown(
        '<div class="panel-title panel-title-success">SPANDA Route &nbsp;·&nbsp; INR → BRL Direct</div>',
        unsafe_allow_html=True,
    )

    # ── Top metrics ───────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    col1.metric(
        label       = "Settlement Time",
        value       = "T+0 (same day)",
        delta       = "-48 hours vs current",
        delta_color = "normal",
        help        = (
            "Direct INR/BRL settlement on SNN-powered infrastructure "
            "targets same-day (T+0) finality."
        ),
    )
    col2.metric(
        label       = "Total Cost",
        value       = fmt_inr(snn_route["total_cost_inr"]),
        delta       = f"{snn_route['cost_percentage']:.3f}% of transaction",
        delta_color = "off",
        help        = (
            "Includes 0.1% settlement fee (assumed), "
            "₹200 flat infrastructure fee, and 18% GST on flat fee."
        ),
    )

    # ── Route steps ───────────────────────────────────────────────────
    st.markdown("**Payment route:**")
    steps = snn_route.get("steps", ["INR → BRL (direct, SNN-powered)"])
    st.markdown(
        f"""
        <div class="route-card route-card-success">
            {"  →  ".join(steps)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Fee breakdown table ───────────────────────────────────────────
    st.markdown("**Fee Breakdown:**")
    items = snn_route.get("itemised", {})

    fee_rows = [
        (
            "Settlement Fee (0.1% ⚠️ assumed)",
            items.get("settlement_fee_inr", 0.0),
            "Based on BIS DLT benchmark (2022) — not production-validated",
        ),
        (
            "Infrastructure Flat Fee",
            items.get("flat_fee_inr", 0.0),
            "Fixed platform fee per transaction",
        ),
        (
            "GST on Flat Fee (18%)",
            items.get("flat_gst_inr", 0.0),
            "Indian Government GST on the flat fee component",
        ),
    ]

    total_inr = snn_route["total_cost_inr"]

    # Build custom HTML table
    table_html = '<div class="premium-table"><div class="premium-table-row premium-table-header"><div>Fee Component</div><div style="text-align: right;">Amount (₹)</div><div>Note</div></div>'
    for name, amt, note in fee_rows:
        table_html += f'''
        <div class="premium-table-row">
            <div class="fee-name">{name}</div>
            <div class="fee-amt fee-amt-success">₹{amt:,.2f}</div>
            <div class="fee-note">{note}</div>
        </div>'''
    
    # Total row
    note_total = f"= {snn_route['cost_percentage']:.3f}% of ₹{transaction_amount:,.0f}"
    table_html += f'''
        <div class="premium-table-row premium-table-total">
            <div class="fee-name">TOTAL</div>
            <div class="fee-amt fee-amt-success">₹{total_inr:,.2f}</div>
            <div class="fee-note" style="color: var(--brand-dark); font-weight: 700;">{note_total}</div>
        </div>
    </div>'''

    st.markdown(table_html, unsafe_allow_html=True)

    # ── Comparison highlight ──────────────────────────────────────────
    usd_equivalent = transaction_amount * 0.0362   # ~3.62% total USD cost
    ratio = usd_equivalent / total_inr if total_inr > 0 else 0

    st.markdown(
        f"""
        <div class="comparison-chip">
            ≈ {ratio:.0f}× cheaper than USD route
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Assumption note — MANDATORY ───────────────────────────────────
    st.caption(
        "⚠️ Assumption: 0.1% settlement fee per BIS DLT benchmark (2022). "
        "Not yet validated in production. "
        "All savings figures are indicative under this assumption."
    )