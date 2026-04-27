"""
dashboard/components/before_panel.py
=====================================
The 'Before' panel — current USD-intermediated SWIFT route.
Shows all 6 real fee components sourced from SBI/HDFC (2024).
Source citations are mandatory for government credibility.
"""

import streamlit as st
import pandas as pd
from dashboard.utils.formatters import fmt_inr


def render_before_panel(transaction_amount: float, usd_route: dict):
    """
    Render the 'Before' panel — current USD SWIFT route costs.

    All 6 fee components are displayed with real labels matching
    SBI FxOut and HDFC remittance fee schedules. No component is
    rounded to zero — IOF tax and SWIFT GST are always shown.

    Parameters
    ----------
    transaction_amount : float — INR value of the transaction
    usd_route          : dict  — usd_route section of API response

    Data keys used from usd_route
    --------------------------------
    total_cost_inr, cost_percentage, settlement_days,
    settlement_hours, steps, itemised (6 sub-keys)
    """
    st.markdown(
        '<div class="panel-title panel-title-danger">Current Route &nbsp;·&nbsp; INR → USD → BRL</div>',
        unsafe_allow_html=True,
    )

    # ── Top metrics ───────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    col1.metric(
        label       = "Settlement Time",
        value       = usd_route["settlement_days"],
        delta       = f"≈{usd_route['settlement_hours']:.0f} hours",
        delta_color = "inverse",
        help        = (
            "SWIFT payments require 2–3 business days due to "
            "correspondent bank processing and USD→BRL conversion."
        ),
    )
    col2.metric(
        label       = "Total Cost",
        value       = fmt_inr(usd_route["total_cost_inr"]),
        delta       = f"{usd_route['cost_percentage']:.2f}% of transaction",
        delta_color = "inverse",
        help        = (
            "Includes INR→USD spread, SWIFT fees, GST, "
            "correspondent bank, BRL spread, and Brazilian IOF tax."
        ),
    )

    # ── Route steps ───────────────────────────────────────────────────
    st.markdown("**Payment route:**")
    steps = usd_route.get("steps", [])
    route_str = " → ".join(steps)
    st.markdown(
        f"""
        <div class="route-card route-card-danger">
            {route_str}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Fee breakdown table ───────────────────────────────────────────
    st.markdown("**Fee Breakdown:**")
    items = usd_route.get("itemised", {})

    # Every fee component must appear — even small ones (IOF, GST)
    fee_rows = [
        (
            "INR → USD Spread (SBI TT rate, 2.0%)",
            items.get("inr_usd_spread_inr",    0.0),
            "Hidden in exchange rate — not shown as a line item by banks",
        ),
        (
            "SWIFT Flat Remittance Fee",
            items.get("swift_flat_fee_inr",    0.0),
            "SBI FxOut / HDFC outward remittance flat fee",
        ),
        (
            "GST on SWIFT Fee (18%)",
            items.get("swift_gst_inr",         0.0),
            "Indian Government GST on the flat remittance fee",
        ),
        (
            "Correspondent Bank Fee ($25 × ₹84)",
            items.get("correspondent_fee_inr", 0.0),
            "Charged by the US intermediary bank; deducted in transit",
        ),
        (
            "USD → BRL Conversion Spread (1.5%)",
            items.get("brl_spread_inr",        0.0),
            "Charged by the Brazilian correspondent bank",
        ),
        (
            "Brazilian IOF Tax (0.38%)",
            items.get("iof_tax_inr",           0.0),
            "Imposto sobre Operações Financeiras — mandatory Brazilian tax",
        ),
    ]

    total_inr = usd_route["total_cost_inr"]

    # Build custom HTML table
    table_html = '<div class="premium-table"><div class="premium-table-row premium-table-header"><div>Fee Component</div><div style="text-align: right;">Amount (₹)</div><div>Note</div></div>'
    for name, amt, note in fee_rows:
        table_html += f'''
        <div class="premium-table-row">
            <div class="fee-name">{name}</div>
            <div class="fee-amt fee-amt-danger">₹{amt:,.2f}</div>
            <div class="fee-note">{note}</div>
        </div>'''
    
    # Total row
    note_total = f"= {usd_route['cost_percentage']:.2f}% of ₹{transaction_amount:,.0f}"
    table_html += f'''
        <div class="premium-table-row premium-table-total">
            <div class="fee-name">TOTAL</div>
            <div class="fee-amt fee-amt-danger">₹{total_inr:,.2f}</div>
            <div class="fee-note" style="color: var(--brand-dark); font-weight: 700;">{note_total}</div>
        </div>
    </div>'''

    st.markdown(table_html, unsafe_allow_html=True)

    # ── Visual cost bar ───────────────────────────────────────────────
    st.markdown("**Cost distribution:**")
    _render_cost_bar(items, total_inr)

    # ── Source citation — mandatory ───────────────────────────────────
    st.caption(
        "📎 Sources: SBI FxOut FAQ (retail.sbi.bank.in, accessed 2024); "
        "HDFC Remittance Fees (hdfcbank.com, accessed 2024); "
        "Brazil IOF tax schedule (Receita Federal do Brasil)."
    )


def _render_cost_bar(items: dict, total: float):
    """
    Render a proportional horizontal bar showing fee breakdown.

    Each fee component gets a segment sized by its fraction of total cost.
    Helps viewers see visually that the FX spread dominates — not the flat fee.
    """
    if total <= 0:
        return

    segments = [
        ("INR/USD Spread",  items.get("inr_usd_spread_inr",    0), "#e74c3c"),
        ("BRL Spread",      items.get("brl_spread_inr",        0), "#c0392b"),
        ("IOF Tax",         items.get("iof_tax_inr",           0), "#e67e22"),
        ("Correspondent",   items.get("correspondent_fee_inr", 0), "#f39c12"),
        ("SWIFT + GST",     items.get("swift_flat_fee_inr", 0) +
                            items.get("swift_gst_inr",         0), "#f1c40f"),
    ]

    bar_html = '<div class="cost-bar">'
    legend   = '<div class="cost-legend">'

    for label, amount, color in segments:
        pct = amount / total * 100
        if pct < 0.5:
            continue
        bar_html += (
            f'<div class="cost-segment" style="width:{pct:.1f}%; background:{color}; '
            f'title="{label}: ₹{amount:,.0f}"></div>'
        )
        legend += (
            f'<span class="cost-legend-item">'
            f'<span class="swatch" style="color:{color};">■</span> '
            f'{label} ({pct:.0f}%)</span>'
        )

    bar_html += "</div>"
    legend   += "</div>"

    st.markdown(bar_html + legend, unsafe_allow_html=True)