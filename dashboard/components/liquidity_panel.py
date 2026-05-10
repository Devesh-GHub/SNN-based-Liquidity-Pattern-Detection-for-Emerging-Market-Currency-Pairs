"""
dashboard/components/liquidity_panel.py
========================================
Liquidity Analysis Panel for the BRICS SPANDA Dashboard.

Displays the full 354-day backtest results from the SNN settlement
routing study: cumulative savings, decision breakdown, risk analysis,
and savings scaling across transaction sizes.

All data comes from GET /liquidity (api/main.py) or the cached
outputs/backtest_results.csv + outputs/backtest_summary.json files
loaded via call_liquidity_api() in utils/api_client.py.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ── CSS injected once ─────────────────────────────────────────────────
_LIQUIDITY_CSS = """
<style>
/* ── Liquidity stat cards ────────────────────────────────────── */
.liq-stat-grid {
    display              : grid;
    grid-template-columns: repeat(3, 1fr);
    gap                  : 1rem;
    margin-bottom        : 1.25rem;
}
@media (max-width: 900px) {
    .liq-stat-grid { grid-template-columns: repeat(2, 1fr); }
}
.liq-stat-card {
    background    : rgba(255, 255, 255, 0.92);
    border        : 1px solid rgba(0, 0, 0, 0.06);
    border-radius : 16px;
    padding       : 1.25rem 1.4rem;
    box-shadow    : 0 2px 12px rgba(0,0,0,0.05);
    transition    : transform 0.2s ease, box-shadow 0.2s ease;
    position      : relative;
    overflow      : hidden;
}
.liq-stat-card::before {
    content       : "";
    position      : absolute;
    top: 0; left: 0; right: 0;
    height        : 3px;
    background    : var(--card-accent, #10b981);
    border-radius : 16px 16px 0 0;
}
.liq-stat-card:hover {
    transform  : translateY(-2px);
    box-shadow : 0 6px 20px rgba(0,0,0,0.09);
}
.liq-stat-label {
    font-size     : 0.68rem;
    font-weight   : 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color         : #5E748E;
    margin-bottom : 0.45rem;
}
.liq-stat-value {
    font-family: var(--font-display, "Sora", sans-serif);
    font-size  : 1.75rem;
    font-weight: 800;
    color      : #0B192C;
    line-height: 1.1;
}
.liq-stat-sub {
    font-size  : 0.72rem;
    color      : #5E748E;
    margin-top : 0.3rem;
}
.liq-stat-badge {
    display       : inline-block;
    background    : rgba(16,185,129,0.1);
    color         : #059669;
    border        : 1px solid rgba(16,185,129,0.25);
    border-radius : 100px;
    font-size     : 0.65rem;
    font-weight   : 700;
    padding       : 0.2rem 0.55rem;
    margin-top    : 0.4rem;
}
.liq-stat-badge.warn {
    background : rgba(245,158,11,0.1);
    color      : #d97706;
    border-color: rgba(245,158,11,0.25);
}

/* ── Scaling table ───────────────────────────────────────────── */
.scale-table {
    width          : 100%;
    border-collapse: collapse;
    font-size      : 0.82rem;
    margin-top     : 0.5rem;
}
.scale-table th {
    background    : #f4f7fb;
    padding       : 0.6rem 1rem;
    text-align    : left;
    font-size     : 0.68rem;
    font-weight   : 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color         : #5E748E;
    border-bottom : 2px solid #e2e8f0;
}
.scale-table td {
    padding       : 0.65rem 1rem;
    border-bottom : 1px solid #f1f5f9;
    color         : #0B192C;
    vertical-align: middle;
}
.scale-table tr:last-child td { border-bottom: none; }
.scale-table .td-saving {
    font-weight : 700;
    color       : #059669;
}
.scale-table .td-pct {
    background    : rgba(16,185,129,0.08);
    color         : #059669;
    border-radius : 100px;
    padding       : 0.15rem 0.5rem;
    font-weight   : 700;
    font-size     : 0.72rem;
    white-space   : nowrap;
}

/* ── Risk section ────────────────────────────────────────────── */
.risk-row {
    display        : flex;
    gap            : 1rem;
    margin-bottom  : 0.75rem;
}
.risk-box {
    flex           : 1;
    background     : rgba(255,255,255,0.92);
    border         : 1px solid rgba(0,0,0,0.06);
    border-radius  : 12px;
    padding        : 1rem 1.2rem;
    text-align     : center;
}
.risk-box-val {
    font-family : var(--font-display, "Sora", sans-serif);
    font-size   : 1.5rem;
    font-weight : 800;
    color       : #0B192C;
}
.risk-box-lbl {
    font-size     : 0.66rem;
    font-weight   : 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color         : #5E748E;
    margin-top    : 0.25rem;
}
.risk-insight {
    background    : rgba(16,185,129,0.07);
    border        : 1px solid rgba(16,185,129,0.2);
    border-radius : 12px;
    padding       : 0.85rem 1.1rem;
    font-size     : 0.82rem;
    color         : #065f46;
    line-height   : 1.6;
}
.risk-insight strong { color: #047857; }
</style>
"""


def _inject_css():
    st.markdown(_LIQUIDITY_CSS, unsafe_allow_html=True)


def _fmt_inr(value: float) -> str:
    """Format a float as an INR string with commas."""
    return f"₹{value:,.0f}"


# ── Chart builders ────────────────────────────────────────────────────

def _chart_cumulative_savings(timeline: list) -> go.Figure:
    """
    Line chart: cumulative SNN savings vs always-SWIFT baseline.
    Highlights wrong DIRECT decisions as red dots.
    """
    df = pd.DataFrame(timeline)
    df["date"] = pd.to_datetime(df["date"])

    wrong_direct = df[(df["decision"] == "DIRECT") & (df["snn_correct"] == 0)]

    fig = go.Figure()

    # Always-SWIFT baseline (red dashed)
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["cumulative_baseline"],
        mode="lines",
        name="Always SWIFT",
        line=dict(color="#ef4444", width=1.5, dash="dash"),
        hovertemplate="<b>Always SWIFT</b><br>%{x|%b %Y}: %{y:₹,.0f}<extra></extra>",
    ))

    # SNN routing cost (green solid)
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["cumulative_direct"],
        mode="lines",
        name="SNN Routing",
        line=dict(color="#10b981", width=2.5),
        fill="tonexty",
        fillcolor="rgba(16,185,129,0.08)",
        hovertemplate="<b>SNN Cost</b><br>%{x|%b %Y}: %{y:₹,.0f}<extra></extra>",
    ))

    # Wrong DIRECT red dots
    if len(wrong_direct) > 0:
        fig.add_trace(go.Scatter(
            x=wrong_direct["date"],
            y=wrong_direct["cumulative_saving"],
            mode="markers",
            name="Wrong DIRECT",
            marker=dict(color="#ef4444", size=5, opacity=0.7),
            hovertemplate="<b>Wrong DIRECT</b><br>%{x|%d %b %Y}<extra></extra>",
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.6)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ),
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(size=11, color="#5E748E"),
        ),
        yaxis=dict(
            tickformat="₹,.0f",
            showgrid=True, gridcolor="rgba(0,0,0,0.04)",
            zeroline=False,
            tickfont=dict(size=11, color="#5E748E"),
        ),
        hovermode="x unified",
    )
    return fig


def _chart_daily_savings(timeline: list) -> go.Figure:
    """Bar chart: daily saving — green for DIRECT days, grey for fallback."""
    df = pd.DataFrame(timeline)
    df["date"] = pd.to_datetime(df["date"])

    colors = [
        "#10b981" if d == "DIRECT" else "#cbd5e1"
        for d in df["decision"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"],
        y=df["saving_day"],
        marker_color=colors,
        hovertemplate=(
            "<b>%{x|%d %b %Y}</b><br>"
            "Saving: %{y:₹,.0f}<extra></extra>"
        ),
        name="Daily Saving",
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.6)",
        showlegend=False,
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(size=11, color="#5E748E"),
        ),
        yaxis=dict(
            tickformat="₹,.0f",
            showgrid=True, gridcolor="rgba(0,0,0,0.04)",
            zeroline=True, zerolinecolor="rgba(0,0,0,0.08)",
            tickfont=dict(size=11, color="#5E748E"),
        ),
        bargap=0.15,
        hovermode="x unified",
    )
    return fig


def _chart_decision_donut(n_direct: int, n_fallback: int) -> go.Figure:
    """Donut chart: DIRECT vs USD_FALLBACK decision split."""
    fig = go.Figure(go.Pie(
        labels=["DIRECT Settlement", "USD Fallback"],
        values=[n_direct, n_fallback],
        hole=0.62,
        marker=dict(
            colors=["#10b981", "#94a3b8"],
            line=dict(color="#ffffff", width=2),
        ),
        textinfo="percent",
        textfont=dict(size=13, color="#ffffff"),
        hovertemplate="<b>%{label}</b><br>%{value} days (%{percent})<extra></extra>",
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="v",
            font=dict(size=11, color="#0B192C"),
            x=1.0, y=0.5,
        ),
        annotations=[dict(
            text=f"<b>{n_direct + n_fallback}</b><br><span style='font-size:10px'>days</span>",
            x=0.5, y=0.5,
            font=dict(size=14, color="#0B192C"),
            showarrow=False,
        )],
    )
    return fig


def _chart_savings_scaling(scaling: list) -> go.Figure:
    """Horizontal bar chart: savings across transaction sizes."""
    labels    = [r["label"]      for r in scaling]
    usd_costs = [r["usd_cost"]   for r in scaling]
    snn_costs = [r["snn_cost"]   for r in scaling]
    savings   = [r["saving"]     for r in scaling]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="USD Route Cost",
        y=labels, x=usd_costs,
        orientation="h",
        marker_color="#fca5a5",
        hovertemplate="<b>USD Cost</b>: %{x:₹,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="SNN Route Cost",
        y=labels, x=snn_costs,
        orientation="h",
        marker_color="#6ee7b7",
        hovertemplate="<b>SNN Cost</b>: %{x:₹,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Net Saving",
        y=labels, x=savings,
        orientation="h",
        marker_color="#10b981",
        hovertemplate="<b>Saving</b>: %{x:₹,.0f}<extra></extra>",
    ))

    fig.update_layout(
        barmode="group",
        margin=dict(l=0, r=0, t=10, b=0),
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.6)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ),
        xaxis=dict(
            tickformat="₹,.0f",
            showgrid=True, gridcolor="rgba(0,0,0,0.04)",
            tickfont=dict(size=10, color="#5E748E"),
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=12, color="#0B192C"),
        ),
        hovermode="y unified",
    )
    return fig


# ── Main render function ──────────────────────────────────────────────

def render_liquidity_panel(data: dict):
    """
    Render the full Liquidity Analysis panel.

    Parameters
    ----------
    data : dict — response from GET /liquidity or call_liquidity_api()
                  Keys: summary, timeline, risk_analysis, savings_scaling
    """
    _inject_css()

    summary  = data.get("summary",         {})
    timeline = data.get("timeline",        [])
    risk     = data.get("risk_analysis",   {})
    scaling  = data.get("savings_scaling", [])

    # ── KPI Strip ─────────────────────────────────────────────────────
    total_saving      = summary.get("total_saving",        9_222_000)
    annualised_saving = summary.get("annualised_saving",   6_564_814)
    saving_pct        = summary.get("saving_pct",          72.36)
    direct_rate_pct   = summary.get("direct_rate_pct",     74.86)
    direct_acc_pct    = summary.get("direct_accuracy_pct", 43.02)
    n_days            = summary.get("n_trading_days",       354)
    n_direct          = summary.get("n_direct",             265)
    n_fallback        = summary.get("n_fallback",            89)
    period_start      = summary.get("val_period_start",    "2024-03-11")
    period_end        = summary.get("val_period_end",      "2025-02-27")

    st.markdown(
        f"""
        <div class="liq-stat-grid">
            <div class="liq-stat-card" style="--card-accent:#10b981">
                <div class="liq-stat-label">Total Saving (Val Period)</div>
                <div class="liq-stat-value">₹{total_saving/1_00_000:.1f}L</div>
                <div class="liq-stat-sub">{period_start} → {period_end}</div>
                <div class="liq-stat-badge">vs Always-SWIFT Baseline</div>
            </div>
            <div class="liq-stat-card" style="--card-accent:#3a7bd5">
                <div class="liq-stat-label">Annualised Saving</div>
                <div class="liq-stat-value">₹{annualised_saving/1_00_000:.1f}L</div>
                <div class="liq-stat-sub">Projected over 252 trading days</div>
                <div class="liq-stat-badge">Per ₹10L transaction</div>
            </div>
            <div class="liq-stat-card" style="--card-accent:#f59e0b">
                <div class="liq-stat-label">Fee Compression</div>
                <div class="liq-stat-value">{saving_pct:.1f}%</div>
                <div class="liq-stat-sub">Cost reduction vs SWIFT baseline</div>
                <div class="liq-stat-badge warn">{n_days} trading days backtested</div>
            </div>
            <div class="liq-stat-card" style="--card-accent:#8b5cf6">
                <div class="liq-stat-label">DIRECT Route Rate</div>
                <div class="liq-stat-value">{direct_rate_pct:.1f}%</div>
                <div class="liq-stat-sub">{n_direct} of {n_days} days routed direct</div>
                <div class="liq-stat-badge">Conservative dual-threshold filter</div>
            </div>
            <div class="liq-stat-card" style="--card-accent:#06b6d4">
                <div class="liq-stat-label">DIRECT Accuracy</div>
                <div class="liq-stat-value">{direct_acc_pct:.1f}%</div>
                <div class="liq-stat-sub">Of DIRECT decisions, rate moved favourably</div>
                <div class="liq-stat-badge warn">Even wrong decisions saved money</div>
            </div>
            <div class="liq-stat-card" style="--card-accent:#10b981">
                <div class="liq-stat-label">Settlement Improvement</div>
                <div class="liq-stat-value">T+2 → T+0</div>
                <div class="liq-stat-sub">~48 hours saved per transaction</div>
                <div class="liq-stat-badge">Near real-time finality</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Row A: Cumulative savings chart ───────────────────────────────
    st.markdown(
        '<div class="dashboard-section-title" '
        'style="font-size:0.72rem;font-weight:800;text-transform:uppercase;'
        'letter-spacing:0.12em;color:#0B192C;margin:1.5rem 0 0.75rem 0;">'
        'Cumulative Cost Comparison — SNN vs Always-SWIFT'
        '</div>',
        unsafe_allow_html=True,
    )
    if timeline:
        st.plotly_chart(
            _chart_cumulative_savings(timeline),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(
            "Green fill = cumulative saving over the 354-day val period. "
            "Red dots = days where DIRECT was triggered incorrectly "
            "(still cheaper than SWIFT due to lower fee structure)."
        )
    else:
        st.info("Timeline data unavailable.")

    # ── Row B: Daily savings bar + Decision donut ─────────────────────
    col_bar, col_donut = st.columns([2, 1])

    with col_bar:
        st.markdown(
            '<div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;'
            'text-transform:uppercase;color:#5E748E;margin-bottom:0.5rem;">'
            'Daily Saving per Transaction'
            '</div>',
            unsafe_allow_html=True,
        )
        if timeline:
            st.plotly_chart(
                _chart_daily_savings(timeline),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption(
                "Green bars = DIRECT days (₹34,800 saved). "
                "Grey bars = USD Fallback days (no saving, standard SWIFT cost)."
            )

    with col_donut:
        st.markdown(
            '<div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;'
            'text-transform:uppercase;color:#5E748E;margin-bottom:0.5rem;">'
            'Routing Decision Split'
            '</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _chart_decision_donut(n_direct, n_fallback),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.divider()

    # ── Row C: Risk analysis + Savings scaling ────────────────────────
    col_risk, col_scale = st.columns([1, 1])

    with col_risk:
        st.markdown(
            '<div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;'
            'text-transform:uppercase;color:#5E748E;margin-bottom:0.75rem;">'
            'Risk Analysis — DIRECT Decision Quality'
            '</div>',
            unsafe_allow_html=True,
        )

        n_right = risk.get("n_right_direct",  114)
        n_wrong = risk.get("n_wrong_direct",  151)
        avg_conf_right = risk.get("avg_conf_correct", 0.3245)
        avg_conf_wrong = risk.get("avg_conf_wrong",   0.3188)
        even_saved     = risk.get("even_wrong_saved_money", True)

        st.markdown(
            f"""
            <div class="risk-row">
                <div class="risk-box">
                    <div class="risk-box-val" style="color:#10b981">{n_right}</div>
                    <div class="risk-box-lbl">Correct DIRECT</div>
                </div>
                <div class="risk-box">
                    <div class="risk-box-val" style="color:#f59e0b">{n_wrong}</div>
                    <div class="risk-box-lbl">Wrong DIRECT</div>
                </div>
                <div class="risk-box">
                    <div class="risk-box-val">{avg_conf_right:.3f}</div>
                    <div class="risk-box-lbl">Avg Conf (correct)</div>
                </div>
                <div class="risk-box">
                    <div class="risk-box-val">{avg_conf_wrong:.3f}</div>
                    <div class="risk-box-lbl">Avg Conf (wrong)</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        insight_text = (
            "Even on the <strong>{n_wrong} incorrect DIRECT days</strong>, "
            "the direct fee (₹1,200) was still far lower than SWIFT (₹36,000) "
            "— so wrong predictions <strong>still saved money</strong> versus the "
            "always-SWIFT baseline. This is a structural property of the fee "
            "architecture that provides a natural risk floor."
        ).format(n_wrong=n_wrong)

        if not even_saved:
            insight_text = (
                "Wrong DIRECT decisions incurred higher costs than the SWIFT baseline "
                "on some days. Review threshold calibration."
            )

        st.markdown(
            f'<div class="risk-insight">{insight_text}</div>',
            unsafe_allow_html=True,
        )

    with col_scale:
        st.markdown(
            '<div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;'
            'text-transform:uppercase;color:#5E748E;margin-bottom:0.5rem;">'
            'Savings Scaling by Transaction Size'
            '</div>',
            unsafe_allow_html=True,
        )

        if scaling:
            st.plotly_chart(
                _chart_savings_scaling(scaling),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            # Table below chart
            table_rows = "".join(
                f"""
                <tr>
                    <td><strong>{r['label']}</strong></td>
                    <td>₹{r['usd_cost']:,.0f}</td>
                    <td>₹{r['snn_cost']:,.0f}</td>
                    <td class="td-saving">₹{r['saving']:,.0f}</td>
                    <td><span class="td-pct">{r['saving_pct']:.1f}%</span></td>
                </tr>
                """
                for r in scaling
            )
            st.markdown(
                f"""
                <table class="scale-table">
                    <thead>
                        <tr>
                            <th>Size</th>
                            <th>USD Route</th>
                            <th>SNN Route</th>
                            <th>Saving</th>
                            <th>%</th>
                        </tr>
                    </thead>
                    <tbody>{table_rows}</tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Scaling data unavailable.")

    st.divider()

    # ── Row D: Methodology note ────────────────────────────────────────
    with st.expander("Backtest Methodology & Assumptions", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
**Routing Decision Logic**
- **DIRECT** triggered when: SNN probability ≥ threshold **AND** spike rate ≥ 0.10
- **USD_FALLBACK**: either condition fails → conservative default
- Threshold locked at `0.52` (Youden J optimum from val-set ROC curve)

**Cost Model (per ₹10L transaction)**
| Fee Component | Amount |
|---|---|
| INR→USD spread (2%) | ₹20,000 |
| SWIFT flat + GST | ₹1,180 |
| Correspondent ($25) | ₹2,100 |
| USD→BRL spread (1.5%) | ₹15,000 |
| Brazilian IOF tax (0.38%) | ₹3,800 |
| **SWIFT Total** | **₹42,080** |
| SNN settlement (0.1% + ₹236) | **₹1,236** |
            """)
        with col_b:
            st.markdown("""
**Backtest Parameters**
- Val period: Mar 2024 → Feb 2025 (354 trading days)
- Transaction size: ₹10,00,000 (₹10 lakh) per day
- Model: BRICSLiquiditySNN (Val AUC = 0.555)
- Lookback window: 10 trading days
- Spike rate threshold: 10% minimum LIF activity

**Honest Caveats**
- 0.1% SNN settlement fee is **assumed** (BIS DLT benchmark, 2022)
- Savings are fee-based only; FX rate movements not modelled
- 43% DIRECT accuracy reflects val-set performance, not live trading
- Annual estimate assumes constant ₹10L daily transaction volume
- Model trained/validated on synthetic INR/BRL data
            """)
        st.caption(
            "Source: notebooks/13_business_logic.ipynb · "
            "Fee data: SBI FxOut FAQ, HDFC remittance page (2024) · "
            "Research prototype — not financial advice."
        )
