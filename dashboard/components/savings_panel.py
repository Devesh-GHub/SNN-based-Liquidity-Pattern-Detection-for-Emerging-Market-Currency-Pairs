"""
dashboard/components/savings_panel.py
======================================
Central savings summary panel — the headline of the dashboard.
"""

import streamlit as st
from dashboard.utils.formatters import fmt_inr


def render_savings_panel(savings: dict, recommendation: str, snn_pred: dict = None):
    """
    Render the central savings arrow panel.

    Parameters
    ----------
    savings        : dict — savings section of API response
    recommendation : str  — "DIRECT" or "USD_FALLBACK"
    snn_pred       : dict — snn_prediction section (for surfacing direction)
    """
    st.markdown(
        '<div class="panel-title panel-title-accent">Savings Intelligence</div>',
        unsafe_allow_html=True,
    )

    amount_inr    = savings["amount_inr"]
    pct           = savings["percentage"]
    tx_assumption = savings.get("monthly_tx_assumption", 5)
    monthly_saving = amount_inr * tx_assumption

    latency_days  = savings.get("latency_saving_days")
    latency_hours = savings.get("latency_saving_hours")
    if latency_hours is None and latency_days is not None:
        latency_hours = float(latency_days) * 24.0
    if latency_days is None and latency_hours is not None:
        latency_days = float(latency_hours) / 24.0

    # ── Big savings spotlight ─────────────────────────────────────────
    st.markdown(
        f"""
        <div class="savings-spotlight">
            <div class="savings-amount">{fmt_inr(amount_inr, decimals=2)}</div>
            <div class="savings-copy">Estimated net saving per transaction</div>
            <div class="savings-band">4.21% &#8594; 0.12% fee compression</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── SNN signal card — direction prediction ────────────────────────
    if snn_pred:
        direction  = snn_pred.get("direction", "UP")
        prob       = snn_pred.get("probability", 0.62)
        confidence = snn_pred.get("confidence", 0.24)
        rec        = snn_pred.get("recommendation", "USD_FALLBACK")

        if rec == "DIRECT":
            sig_cls    = "signal-card-direct"
            sig_icon   = "&#9650;"   # ▲
            sig_label  = "DIRECT ROUTE"
            dir_color  = "#10b981"
        else:
            sig_cls    = "signal-card-fallback"
            sig_icon   = "&#9660;"   # ▼
            sig_label  = "USD FALLBACK"
            dir_color  = "#f59e0b"

        dir_arrow  = "&#8599;" if direction == "UP" else "&#8600;"
        dir_color2 = "#10b981" if direction == "UP" else "#ef4444"

        st.markdown(
            f"""
            <div class="signal-card {sig_cls}">
                <div class="signal-top">
                    <div class="signal-label">{sig_label}</div>
                    <div class="signal-route-icon">{sig_icon}</div>
                </div>
                <div class="signal-row">
                    <div class="signal-item">
                        <div class="signal-val" style="color:{dir_color2};">{dir_arrow} {direction}</div>
                        <div class="signal-lbl">INR/BRL Direction</div>
                    </div>
                    <div class="signal-vr"></div>
                    <div class="signal-item">
                        <div class="signal-val">{prob:.3f}</div>
                        <div class="signal-lbl">Model Probability</div>
                    </div>
                    <div class="signal-vr"></div>
                    <div class="signal-item">
                        <div class="signal-val">{confidence:.2f}</div>
                        <div class="signal-lbl">Confidence</div>
                    </div>
                </div>
                <div class="signal-note">SNN 10-day price signal prediction</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Compact stats grid (replaces broken 3-column st.metric) ───────
    st.markdown(
        f"""
        <div class="smc-grid">
            <div class="smc-card">
                <div class="smc-val">{(latency_days or 0):.0f} days</div>
                <div class="smc-lbl">Settlement Time Saved</div>
                <div class="smc-sub">&#8776;{(latency_hours or 0):.0f} hours faster</div>
            </div>
            <div class="smc-card">
                <div class="smc-val">{fmt_inr(monthly_saving)}</div>
                <div class="smc-lbl">Monthly Saving ({tx_assumption} tx)</div>
            </div>
            <div class="smc-card smc-card-accent">
                <div class="smc-val">{savings["annual_formatted"]}</div>
                <div class="smc-lbl">Annual Saving Estimate</div>
                <div class="smc-sub">{tx_assumption} tx/month assumed</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Routing recommendation badge ──────────────────────────────────
    if recommendation == "DIRECT":
        trajectory_tone  = "trajectory-direct"
        trajectory_icon  = "&#8595;"
        trajectory_label = "Cost goes DOWN"
    else:
        trajectory_tone  = "trajectory-fallback"
        trajectory_icon  = "&#9888;"
        trajectory_label = "Using USD route"

    st.markdown(
        f"""
        <div class="trajectory-card {trajectory_tone}">
            <div class="trajectory-icon">{trajectory_icon}</div>
            <div class="trajectory-label">{trajectory_label}</div>
            <div class="trajectory-copy">
                from <b>4.21%</b> &#8594; <b>0.12%</b> of transaction
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Savings realised only when SNN confidence threshold is met "
        "(prob >= 0.70 AND spike_rate >= 0.10). "
        "System defaults to USD route when confidence is low — "
        "only 9.9% of days triggered direct settlement in backtesting."
    )


def render_metrics_row(data: dict):
    """
    Render the four top-level summary metric cards.

    Parameters
    ----------
    data : dict — full API response dict
    """
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        label       = "USD Route Cost",
        value       = f"₹{data['usd_route_cost']:,.0f}",
        delta       = f"{data['usd_route']['cost_percentage']:.2f}% of tx",
        delta_color = "inverse",
        help        = "Total cost via current SWIFT USD route",
    )
    c2.metric(
        label       = "SNN Route Cost",
        value       = f"₹{data['snn_route_cost']:,.0f}",
        delta       = f"{data['snn_route']['cost_percentage']:.3f}% of tx",
        delta_color = "off",
        help        = "Total cost via proposed direct SNN route (assumed fees)",
    )
    c3.metric(
        label       = "Saving per Transaction",
        value       = f"₹{data['savings_inr']:,.0f}",
        delta       = f"{data['savings_percentage']:.1f}%",
        delta_color = "normal",
        help        = "USD cost − SNN cost",
    )
    c4.metric(
        label       = "Settlement Time",
        value       = data["settlement_time_proposed"],
        delta       = f"was {data['settlement_time_current']}",
        delta_color = "normal",
        help        = "SNN route enables same-day settlement",
    )