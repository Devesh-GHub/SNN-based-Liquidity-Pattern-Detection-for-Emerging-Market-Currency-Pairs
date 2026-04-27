"""dashboard/components/header.py — Title banner with SPANDA brand identity."""

import streamlit as st
import os
import base64


def _load_logo_b64() -> str:
    """Read the SPANDA SVG and return a base64 data-URI."""
    svg_path = os.path.join(os.path.dirname(__file__), "..", "svgviewer-output.svg")
    try:
        with open(os.path.normpath(svg_path), "r", encoding="utf-8") as f:
            svg_content = f.read()
        b64 = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")
        return f"data:image/svg+xml;base64,{b64}"
    except Exception:
        return ""


def render_header():
    logo_uri = _load_logo_b64()
    logo_html = (
        f'<img src="{logo_uri}" alt="SPANDA Logo" class="hero-logo-img">'
        if logo_uri
        else '<div class="hero-logo-placeholder">SPANDA</div>'
    )

    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="hero-inner">
                <div class="hero-left">
                    <p class="hero-kicker">
                        Strategic Finance Intelligence &nbsp;&middot;&nbsp; Patent Showcase Mode
                    </p>
                    <div class="hero-wordmark">
                        <span class="hero-brand">SPANDA</span>
                        <span class="hero-wd-sep">|</span>
                        <span class="hero-product">SNN Settlement Advisor</span>
                    </div>
                    <p class="hero-subtitle">
                        Biologically-inspired Spiking Neural Networks for event-driven
                        INR/BRL liquidity discovery &mdash; policy-grade settlement intelligence
                        and cross-border cost transparency across BRICS nations.
                    </p>
                    <div class="hero-tags">
                        <span class="hero-tag hero-tag-glow">&#9889; ~97% Fee Compression</span>
                        <span class="hero-tag">T+2 &rarr; T+0 Settlement</span>
                        <span class="hero-tag">10.6&times; Energy vs LSTM</span>
                        <span class="hero-tag">BRICSLiquiditySNN v1.0</span>
                        <span class="hero-tag">Val AUC 0.555</span>
                    </div>
                    <div class="hero-kpi-strip">
                        <div class="hero-kpi">
                            <div class="hero-kpi-val">~97%</div>
                            <div class="hero-kpi-lbl">Fee Compression</div>
                        </div>
                        <div class="hero-kpi-vr"></div>
                        <div class="hero-kpi">
                            <div class="hero-kpi-val">T+0</div>
                            <div class="hero-kpi-lbl">Settlement Speed</div>
                        </div>
                        <div class="hero-kpi-vr"></div>
                        <div class="hero-kpi">
                            <div class="hero-kpi-val">10.6&times;</div>
                            <div class="hero-kpi-lbl">Energy Efficiency</div>
                        </div>
                        <div class="hero-kpi-vr"></div>
                        <div class="hero-kpi">
                            <div class="hero-kpi-val">2,945</div>
                            <div class="hero-kpi-lbl">SNN Parameters</div>
                        </div>
                    </div>
                </div>
                <div class="hero-right">
                    {logo_html}
                    <div class="hero-logo-name">SPANDA</div>
                    <div class="hero-logo-sub">Global Unity Banking System</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_customer_banner():
    """Render the customer persona banner."""
    st.markdown(
        """
        <div class="persona-card">
            <div class="persona-flex">
                <div class="persona-text">
                    <div class="persona-head">Prototype Persona &nbsp;·&nbsp; Research Illustration</div>
                    <div class="persona-copy">
                        An Indian textile exporter (Surat, Gujarat) invoicing a São Paulo importer in BRL.
                        Current USD-intermediated routing leaks <strong>≈ 3.6%</strong> per transaction
                        and settles in <strong>T+2 to T+3</strong> — creating avoidable working-capital drag.
                        SPANDA's SNN-powered direct route compresses fees to <strong>0.12%</strong>
                        with <strong>same-day finality</strong>.
                    </div>
                </div>
                <div class="persona-stats">
                    <div class="persona-stat persona-stat-red">
                        <div class="ps-val">3.6%</div>
                        <div class="ps-lbl">Cost Today (USD Route)</div>
                    </div>
                    <div class="persona-stat-arrow">→</div>
                    <div class="persona-stat persona-stat-green">
                        <div class="ps-val">0.12%</div>
                        <div class="ps-lbl">SPANDA Direct Route</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
