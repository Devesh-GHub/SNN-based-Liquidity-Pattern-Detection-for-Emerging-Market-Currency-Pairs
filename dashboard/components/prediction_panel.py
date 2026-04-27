"""
dashboard/components/prediction_panel.py
==========================================
SNN prediction engine panel — the most technically distinct part
of the dashboard. Shows direction, spike rate, confidence, and
the decision logic that makes this an SNN (not just any model).

Design rules:
- Always show threshold values — demonstrates model understanding
- Never hide spike rate — it is the unique SNN contribution
- Colour-code recommendation box (green=DIRECT, orange=FALLBACK)
- Use st.progress() for spike rate — clean and fast
- Expander explains the LIF decision logic in plain English
"""

import streamlit as st


def render_prediction_panel(pred: dict):
    """
    Render the full SNN prediction engine panel.

    Parameters
    ----------
    pred : dict — snn_prediction section of API response

    Keys used
    ---------
    direction, probability, confidence, spike_rate, spike_rate_pct,
    recommendation, recommendation_text,
    prob_threshold_used, rate_threshold_used
    """
    st.markdown(
        '<div class="panel-title panel-title-accent">BRICSLiquiditySNN &nbsp;·&nbsp; Prediction Engine</div>',
        unsafe_allow_html=True,
    )

    # ── Three metric columns ──────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    # Direction
    direction = pred["direction"]
    col1.metric(
        label       = "Predicted Direction",
        value       = f"{'📈' if direction == 'UP' else '📉'} {direction}",
        delta       = f"Probability: {pred['probability']:.3f}",
        delta_color = "normal" if direction == "UP" else "inverse",
        help        = (
            "SNN's prediction for tomorrow's INR/BRL rate movement. "
            "Based on rate-coded spike accumulation over 10-day window."
        ),
    )

    # Spike rate — unique to SNN
    col2.metric(
        label       = "Avg Spike Rate",
        value       = f"{pred['spike_rate_pct']:.1f}%",
        delta       = "Neuromorphic activity",
        delta_color = "off",
        help        = (
            "Fraction of LIF neurons firing per timestep, "
            "averaged across both LIF layers and all 10 timesteps. "
            "Lower = more energy efficient. "
            "Baseline LSTM equivalent would be 100% (dense)."
        ),
    )

    # Confidence
    col3.metric(
        label       = "Confidence Score",
        value       = f"{pred['confidence']:.3f}",
        delta       = f"Threshold: {pred['prob_threshold_used']}",
        delta_color = (
            "normal" if pred["confidence"] >
            (pred["prob_threshold_used"] - 0.5) * 2
            else "inverse"
        ),
        help        = (
            "Confidence = |probability − 0.5| × 2. "
            "0 = completely uncertain, 1 = fully confident."
        ),
    )

    # ── Recommendation box — colour coded ────────────────────────────
    rec = pred["recommendation"]
    rec_text = pred.get("recommendation_text", "")

    if rec == "DIRECT":
        st.markdown(
            f"""
            <div class="prediction-callout prediction-direct">
                <div class="kicker">Decision Signal</div>
                <p class="headline">✅ Recommendation: Direct Settlement</p>
                <p class="copy">{rec_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="prediction-callout prediction-fallback">
                <div class="kicker">Decision Signal</div>
                <p class="headline">⚠️ Recommendation: Use USD Route</p>
                <p class="copy">{rec_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Spike rate progress bar ───────────────────────────────────────
    st.markdown("**Neuronal Spike Activity:**")

    spike_rate = float(pred["spike_rate"])
    spike_pct  = float(pred["spike_rate_pct"])

    # Colour coding: green=healthy, orange=high, red=too high
    if spike_rate < 0.30:
        bar_label = f"Spike Rate: {spike_pct:.1f}%  — sparse (efficient)"
    elif spike_rate < 0.60:
        bar_label = f"Spike Rate: {spike_pct:.1f}%  — moderate"
    else:
        bar_label = f"Spike Rate: {spike_pct:.1f}%  — dense (less efficient)"

    st.progress(
        min(spike_rate, 1.0),
        text=bar_label,
    )

    # Spike rate context
    col_a, col_b = st.columns(2)
    col_a.caption(
        f"Rate threshold for DIRECT: ≥ {pred['rate_threshold_used'] * 100:.0f}%  "
        f"{'✅ met' if spike_rate >= pred['rate_threshold_used'] else '❌ not met'}"
    )
    col_b.caption(
        f"Prob threshold for DIRECT: ≥ {pred['prob_threshold_used']}  "
        f"{'✅ met' if pred['probability'] >= pred['prob_threshold_used'] else '❌ not met'}"
    )

    # ── Threshold explainer expander ──────────────────────────────────
    with st.expander("🔬 How does the SNN make this decision?"):
        st.markdown(
            """
            ### SNN Architecture (BRICSLiquiditySNN)

            The model processes the last **10 daily INR/BRL price movements**
            as spike trains through two **Leaky Integrate-and-Fire (LIF)**
            neuron layers:

                Price sequence (10 days)
                        ↓ [spike encoding: |return| > 0.3% → spike = 1]
                FC₁(9→64) + BatchNorm + LIF₁(τ=2.0, v_threshold=0.1)
                        ↓ [binary spikes: 0 or 1 per neuron per timestep]
                FC₂(64→32) + BatchNorm + LIF₂(τ=2.0, v_threshold=0.1)
                        ↓ [spike accumulation: sum over 10 timesteps ÷ 10]
                FC₃(32→1) → Sigmoid → Probability

            ### DIRECT route triggered when **both** conditions are met:

            | Condition | Threshold | Why |
            |---|---|---|
            | Prediction probability | ≥ 0.70 | High confidence UP signal |
            | Average spike rate | ≥ 0.10 | Sufficient neuronal activity |

            If **either** condition fails → **USD fallback** (safe default).

            ### Why this is conservative by design

            Only **9.9% of days** triggered direct settlement in backtesting
            (35 of 354 val-period days). This is intentional — the system
            prioritises avoiding incorrect direct-route decisions over
            maximising direct usage.

            ### What spike rate tells us

            - **Low spike rate** (< 15%): few neurons fired — weak signal
              in the price data. Model is uncertain.
            - **Moderate rate** (15–50%): healthy activation — model has
              sufficient information to make a prediction.
            - **High rate** (> 50%): very active — high volatility period.
              Note: current model averages 47.8% (slightly high; future
              work targets 10–30% via adaptive thresholds).

            ### Energy efficiency context

            At 47.8% spike rate, the SNN performs ~3,046 synaptic operations
            per inference vs ~20,480 FLOPs for the LSTM baseline —
            a **10.6× estimated energy reduction** on neuromorphic hardware
            (Blouw et al., 2019).
            """
        )

    # ── Model metadata row ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
        <div class="meta-pill-grid">
            <div class="meta-pill">
                <div class="label">Model</div>
                <div class="value">BRICSLiquiditySNN v1.0</div>
            </div>
            <div class="meta-pill">
                <div class="label">Validation AUC</div>
                <div class="value">0.555 (vs LSTM 0.514)</div>
            </div>
            <div class="meta-pill">
                <div class="label">Parameters</div>
                <div class="value">2,945 (vs LSTM 5,537)</div>
            </div>
            <div class="meta-pill">
                <div class="label">Energy Estimate</div>
                <div class="value">8.95 pJ (vs LSTM 94.46 pJ)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )