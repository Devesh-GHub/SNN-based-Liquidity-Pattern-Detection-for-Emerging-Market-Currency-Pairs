"""
dashboard/app.py
================
BRICS SNN Settlement Advisor — main Streamlit entry point.

Run locally:
    streamlit run dashboard/app.py

Deploy (Streamlit Cloud):
    Main file: dashboard/app.py
    Secret   : API_URL = https://your-render-url.onrender.com
"""

import streamlit as st
import sys
import os
import time
import base64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.config import (
    DEFAULT_AMOUNT, MIN_AMOUNT, MAX_AMOUNT,
    DEFAULT_MONTHLY_TX, DEMO_PRICE_SEQUENCE, PLACEHOLDER_DATA,
)
from dashboard.components.header           import render_header, render_customer_banner
from dashboard.components.before_panel     import render_before_panel
from dashboard.components.after_panel      import render_after_panel
from dashboard.components.savings_panel    import render_savings_panel, render_metrics_row
from dashboard.components.prediction_panel import render_prediction_panel
from dashboard.components.liquidity_panel  import render_liquidity_panel
from dashboard.utils.api_client            import (
    call_predict_api, call_health_api, call_liquidity_api,
    validate_price_input, API_URL,
)

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPANDA — BRICS SNN Settlement Advisor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Sora:wght@400;500;600;700;800&display=swap');

        /* ── CSS Custom Properties ─────────────────────────────────── */
        :root {
            --font-body   : "Plus Jakarta Sans", system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
            --font-display: "Sora", system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
            --bg-base     : #f4f7fb;
            --bg-card     : rgba(255, 255, 255, 0.88);
            --border-soft : rgba(0, 0, 0, 0.05);
            --border-hard : rgba(0, 0, 0, 0.10);
            --text-main   : #0B192C;
            --text-muted  : #5E748E;
            --brand-dark  : #08101a;
            --accent-teal : #00d2ff;
            --accent-green: #10b981;
            --accent-red  : #ef4444;
            --accent-blue : #3a7bd5;
            --accent-orange: #f59e0b;
            --spanda-teal : #00E5CC;
        }

        html, body, [class*="css"], input, textarea, button {
            font-family: var(--font-body) !important;
            color: var(--text-main);
        }
        h1, h2, h3 { font-family: var(--font-display) !important; }

        /* ── App background ───────────────────────────────────────── */
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 10% 10%, rgba(58,123,213,0.07), transparent 30%),
                radial-gradient(circle at 90% 90%, rgba(0,210,255,0.07), transparent 30%),
                var(--bg-base);
        }

        .block-container {
            max-width: 100% !important;
            padding-left : clamp(1rem, 3vw, 3rem);
            padding-right: clamp(1rem, 3vw, 3rem);
            padding-top  : 1.15rem;
            padding-bottom: 3rem;
            animation: fadeIn 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(12px); }
            to   { opacity: 1; transform: translateY(0);    }
        }

        div[data-testid="stHorizontalBlock"] { gap: 1.25rem; align-items: stretch; }

        /* ── Section titles ────────────────────────────────────────── */
        .dashboard-section-title {
            font-family: var(--font-display);
            font-size  : 0.78rem;
            font-weight: 800;
            color      : var(--brand-dark);
            text-transform : uppercase;
            letter-spacing : 0.14em;
            display   : flex;
            align-items: center;
            gap       : 1rem;
            margin    : 2rem 0 1rem 0;
        }
        .dashboard-section-title::after {
            content: ""; flex: 1; height: 1px;
            background: linear-gradient(90deg, var(--border-hard), transparent);
        }

        /* ══════════════════════════════════════════════════════════
           HERO SHELL
        ══════════════════════════════════════════════════════════ */
        .hero-shell {
            background: linear-gradient(135deg, #060d18 0%, #0c1e38 55%, #0a2240 100%);
            border-radius: 24px;
            padding  : 2.75rem 3rem;
            color    : #ffffff;
            box-shadow: 0 24px 48px rgba(6,13,24,0.22),
                        inset 0 1px 0 rgba(255,255,255,0.07);
            margin-bottom : 1.75rem;
            position : relative;
            overflow : hidden;
            border   : 1px solid rgba(0,229,204,0.08);
        }
        .hero-shell::before {
            content  : "";
            position : absolute;
            top: -40%; left: -20%;
            width: 160%; height: 160%;
            background: radial-gradient(ellipse at 30% 50%,
                rgba(0,229,204,0.06) 0%, transparent 55%),
                radial-gradient(ellipse at 80% 20%,
                rgba(58,123,213,0.06) 0%, transparent 45%);
            pointer-events: none;
        }

        /* ── Hero inner two-column layout ─────────────────────── */
        .hero-inner {
            display : flex;
            align-items : flex-start;
            justify-content: space-between;
            gap     : 3rem;
            position: relative;
            z-index : 1;
        }
        .hero-left  { flex: 1; min-width: 0; }
        .hero-right {
            display        : flex;
            flex-direction : column;
            align-items    : center;
            gap            : 0.5rem;
            flex-shrink    : 0;
        }

        /* ── Logo image ───────────────────────────────────────── */
        .hero-logo-img {
            width   : 190px;
            height  : 196px;
            object-fit: contain;
            border-radius: 18px;
            filter  : drop-shadow(0 0 28px rgba(0,229,204,0.45));
            transition: filter 0.4s ease, transform 0.4s ease;
        }
        .hero-logo-img:hover {
            filter   : drop-shadow(0 0 40px rgba(0,229,204,0.65));
            transform: scale(1.02);
        }
        .hero-logo-placeholder {
            width: 190px; height: 196px;
            background: rgba(0,229,204,0.06);
            border: 1px dashed rgba(0,229,204,0.3);
            border-radius: 18px;
            display: flex; align-items: center; justify-content: center;
            color: rgba(0,229,204,0.5);
            font-family: var(--font-display); font-size: 1.2rem; font-weight: 800;
            letter-spacing: 0.25em;
        }
        .hero-logo-name {
            font-family   : var(--font-display);
            font-weight   : 800;
            font-size     : 1.15rem;
            letter-spacing: 0.3em;
            color         : var(--spanda-teal);
            text-align    : center;
            margin-top    : 0.4rem;
        }
        .hero-logo-sub {
            font-size     : 0.6rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color         : rgba(255,255,255,0.35);
            text-align    : center;
        }

        /* ── Hero wordmark ────────────────────────────────────── */
        .hero-kicker {
            font-size     : 0.7rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color         : var(--spanda-teal);
            font-weight   : 700;
            margin-bottom : 0.9rem;
            opacity       : 0.9;
        }
        .hero-wordmark {
            display    : flex;
            align-items: baseline;
            gap        : 0.85rem;
            margin-bottom: 0.85rem;
            flex-wrap  : wrap;
        }
        .hero-brand {
            font-family   : var(--font-display);
            font-weight   : 800;
            font-size     : 3rem;
            color         : var(--spanda-teal);
            letter-spacing: 0.18em;
            line-height   : 1;
        }
        .hero-wd-sep {
            font-size: 1.6rem;
            color    : rgba(255,255,255,0.18);
            line-height: 1.1;
        }
        .hero-product {
            font-family   : var(--font-display);
            font-weight   : 500;
            font-size     : 1.25rem;
            color         : rgba(255,255,255,0.72);
            line-height   : 1.1;
        }
        .hero-subtitle {
            font-size  : 0.95rem;
            color      : rgba(255,255,255,0.65);
            line-height: 1.65;
            max-width  : 700px;
            margin-bottom: 1.4rem;
        }

        /* ── Hero tags ────────────────────────────────────────── */
        .hero-tags {
            display  : flex;
            flex-wrap: wrap;
            gap      : 0.45rem;
            margin-bottom: 1.5rem;
        }
        .hero-tag {
            background  : rgba(255,255,255,0.05);
            border      : 1px solid rgba(255,255,255,0.10);
            padding     : 0.35rem 0.75rem;
            border-radius: 100px;
            font-size   : 0.72rem;
            font-weight : 600;
            color       : rgba(255,255,255,0.85);
            backdrop-filter: blur(4px);
            letter-spacing: 0.02em;
        }
        .hero-tag-glow {
            background   : rgba(0,229,204,0.12) !important;
            border-color : rgba(0,229,204,0.35) !important;
            color        : var(--spanda-teal) !important;
        }

        /* ── KPI Strip ────────────────────────────────────────── */
        .hero-kpi-strip {
            display        : flex;
            align-items    : center;
            background     : rgba(0,0,0,0.3);
            border         : 1px solid rgba(0,229,204,0.1);
            border-radius  : 16px;
            padding        : 1rem 1.5rem;
            backdrop-filter: blur(12px);
            max-width      : 620px;
        }
        .hero-kpi       { flex: 1; text-align: center; }
        .hero-kpi-val   {
            font-family: var(--font-display);
            font-weight: 800;
            font-size  : 1.55rem;
            color      : var(--spanda-teal);
            line-height: 1;
        }
        .hero-kpi-lbl   {
            font-size     : 0.62rem;
            font-weight   : 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color         : rgba(255,255,255,0.45);
            margin-top    : 0.3rem;
        }
        .hero-kpi-vr    {
            width     : 1px;
            height    : 38px;
            background: rgba(255,255,255,0.1);
            margin    : 0 1rem;
            flex-shrink: 0;
        }

        /* ══════════════════════════════════════════════════════════
           PERSONA CARD
        ══════════════════════════════════════════════════════════ */
        .persona-card {
            background  : rgba(255,255,255,0.72);
            border      : 1px solid var(--border-hard);
            padding     : 1.4rem 1.75rem;
            border-radius: 18px;
            margin-bottom: 1.75rem;
            box-shadow  : 0 6px 20px rgba(0,0,0,0.03);
            font-size   : 0.93rem;
            line-height : 1.65;
            backdrop-filter: blur(8px);
        }
        .persona-flex {
            display     : flex;
            align-items : center;
            gap         : 2rem;
            justify-content: space-between;
            flex-wrap   : wrap;
        }
        .persona-text { flex: 1; min-width: 260px; }
        .persona-head {
            font-weight   : 800;
            font-family   : var(--font-display);
            font-size     : 0.75rem;
            color         : var(--brand-dark);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom : 0.5rem;
        }
        .persona-copy { color: var(--text-muted); }
        .persona-stats {
            display    : flex;
            align-items: center;
            gap        : 0.6rem;
            flex-shrink: 0;
        }
        .persona-stat {
            text-align   : center;
            padding      : 0.85rem 1.1rem;
            border-radius: 14px;
            min-width    : 95px;
        }
        .persona-stat-red   { background: rgba(239,68,68,0.07);  border: 1px solid rgba(239,68,68,0.2);  }
        .persona-stat-green { background: rgba(16,185,129,0.07); border: 1px solid rgba(16,185,129,0.2); }
        .persona-stat-arrow {
            font-size : 1.4rem;
            color     : var(--text-muted);
            background: transparent;
            border    : none;
            padding   : 0;
            min-width : unset;
        }
        .ps-val {
            font-family: var(--font-display);
            font-weight: 800;
            font-size  : 1.35rem;
            line-height: 1;
        }
        .persona-stat-red   .ps-val { color: var(--accent-red);   }
        .persona-stat-green .ps-val { color: var(--accent-green); }
        .ps-lbl {
            font-size     : 0.62rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color         : var(--text-muted);
            font-weight   : 700;
            margin-top    : 0.25rem;
        }

        /* ══════════════════════════════════════════════════════════
           STATUS STRIP
        ══════════════════════════════════════════════════════════ */
        .status-strip {
            position  : relative;
            display   : flex;
            align-items: center;
            justify-content: space-between;
            gap       : 1rem;
            padding   : 1.1rem 1.5rem 1.1rem 1.7rem;
            border-radius: 14px;
            border    : 1px solid var(--border-soft);
            background: #fff;
            box-shadow: 0 4px 18px rgba(0,0,0,0.03);
            overflow  : hidden;
        }
        .status-strip::before {
            content  : "";
            position : absolute; left: 0; top: 0; bottom: 0; width: 4px;
            background: linear-gradient(180deg, var(--accent-blue), var(--accent-teal));
        }
        .status-strip .label {
            font-size     : 0.72rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color         : var(--text-muted);
            font-weight   : 700;
            margin-bottom : 0.2rem;
        }
        .status-strip .value {
            font-family: var(--font-display);
            font-size  : 1rem;
            color      : var(--brand-dark);
            font-weight: 800;
        }

        /* ══════════════════════════════════════════════════════════
           METRIC CARDS  (st.metric)
        ══════════════════════════════════════════════════════════ */
        [data-testid="stMetric"] {
            background    : var(--bg-card);
            border-radius : 18px;
            padding       : 1.4rem 1.6rem;
            box-shadow    : 0 6px 22px rgba(0,0,0,0.04);
            border        : 1px solid var(--border-soft);
            backdrop-filter: blur(10px);
            transition    : transform 0.2s, box-shadow 0.2s;
        }
        [data-testid="stMetric"]:hover {
            transform : translateY(-3px);
            box-shadow: 0 10px 32px rgba(0,0,0,0.08);
        }
        [data-testid="stMetricLabel"] {
            font-size     : 0.72rem;
            color         : var(--text-muted);
            font-weight   : 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        [data-testid="stMetricValue"] {
            font-family: var(--font-display) !important;
            font-size  : clamp(1.65rem, 2vw, 2.2rem);
            font-weight: 800;
            color      : var(--brand-dark);
            line-height: 1.1;
            margin     : 0.4rem 0;
        }
        [data-testid="stMetricDelta"] { font-weight: 600; font-size: 0.88rem; }

        /* ══════════════════════════════════════════════════════════
           ROUTE COMPARISON PANELS
        ══════════════════════════════════════════════════════════ */
        div[data-testid="column"]:has(.panel-title-danger),
        div[data-testid="column"]:has(.panel-title-success),
        div[data-testid="column"]:has(.savings-spotlight) {
            border-radius   : 20px;
            border          : 1px solid var(--border-soft);
            background      : rgba(255,255,255,0.65);
            box-shadow      : 0 14px 36px rgba(2,6,23,0.06);
            backdrop-filter : blur(12px);
            padding         : 1.4rem 1.4rem 1.5rem 1.4rem;
        }

        .panel-title {
            font-family  : var(--font-display);
            font-size    : 1.1rem;
            font-weight  : 800;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            display      : inline-block;
        }
        .panel-title-danger  { color: var(--brand-dark); border-bottom: 3px solid var(--accent-orange); }
        .panel-title-success { color: var(--brand-dark); border-bottom: 3px solid var(--spanda-teal);   }
        .panel-title-accent  { color: var(--brand-dark); border-bottom: 3px solid var(--accent-blue);   }

        /* Route pills */
        .route-card {
            padding     : 1.1rem 1.4rem;
            border-radius: 12px;
            font-family : var(--font-display);
            font-weight : 700;
            font-size   : 0.88rem;
            display     : inline-flex;
            align-items : center;
            justify-content: center;
            width       : 100%;
            text-align  : center;
            box-shadow  : inset 0 1px 0 rgba(255,255,255,0.5);
            border      : 1px solid var(--border-soft);
            margin-bottom: 1.25rem;
        }
        .route-card-danger  { background: linear-gradient(135deg,#fff5f5,#fff); color: #c53030; }
        .route-card-success { background: linear-gradient(135deg,#e6fffa,#fff); color: #047857; }

        /* ══════════════════════════════════════════════════════════
           FEE BREAKDOWN TABLE
        ══════════════════════════════════════════════════════════ */
        .premium-table {
            background   : #fff;
            border-radius: 14px;
            overflow     : hidden;
            box-shadow   : 0 4px 16px rgba(0,0,0,0.03);
            border       : 1px solid var(--border-soft);
            margin       : 1rem 0;
            width        : 100%;
            font-size    : 0.83rem;
        }
        .premium-table-row {
            display    : grid;
            grid-template-columns: 2fr 0.9fr 2.6fr;
            gap        : 0.8rem;
            padding    : 14px 18px;
            border-bottom: 1px solid var(--border-soft);
            align-items: center;
        }
        .premium-table-row:last-child { border-bottom: none; }
        .premium-table-header {
            background    : #f8fafc;
            font-weight   : 700;
            color         : var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size     : 0.72rem;
        }
        .premium-table-total {
            background : var(--bg-base);
            font-weight: 800;
            font-size  : 0.95rem;
            border-top : 2px solid var(--border-hard) !important;
        }
        .fee-name       { font-weight: 600; color: var(--brand-dark); }
        .fee-amt        { font-family: var(--font-display); font-weight: 700; text-align: right; }
        .fee-amt-danger { color: #e53e3e; }
        .fee-amt-success{ color: #059669; }
        .fee-note       { color: var(--text-muted); font-size: 0.73rem; padding-left: 8px; line-height: 1.4; }

        /* ══════════════════════════════════════════════════════════
           COST BAR (before_panel)
        ══════════════════════════════════════════════════════════ */
        .cost-bar {
            display      : flex;
            height       : 11px;
            border-radius: 100px;
            overflow     : hidden;
            width        : 100%;
            margin-bottom: 0.6rem;
            background   : rgba(0,0,0,0.05);
        }
        .cost-segment {
            height    : 100%;
            transition: width 0.35s ease;
        }
        .cost-legend {
            display  : flex;
            flex-wrap: wrap;
            gap      : 0.35rem 1rem;
            margin-top: 0.4rem;
        }
        .cost-legend-item {
            font-size  : 0.71rem;
            color      : var(--text-muted);
            display    : flex;
            align-items: center;
            gap        : 0.2rem;
            font-weight: 500;
        }
        .swatch { font-size: 0.82rem; line-height: 1; }

        /* ══════════════════════════════════════════════════════════
           SAVINGS SPOTLIGHT
        ══════════════════════════════════════════════════════════ */
        .savings-spotlight {
            background: linear-gradient(135deg, var(--brand-dark), #1e293b);
            padding   : 2.25rem 1.75rem;
            border-radius: 20px;
            text-align: center;
            color     : white;
            box-shadow: 0 20px 40px rgba(15,23,42,0.22);
            border    : 1px solid rgba(255,255,255,0.05);
            margin-bottom: 1.5rem;
            position  : relative;
            overflow  : hidden;
        }
        .savings-spotlight::after {
            content  : "";
            position : absolute; right: -20%; top: -50%;
            width    : 200px; height: 300px;
            background: radial-gradient(circle, rgba(16,185,129,0.3), transparent 70%);
            filter   : blur(20px);
        }
        .savings-amount {
            font-family: var(--font-display);
            font-size  : 3.5rem;
            font-weight: 800;
            background : linear-gradient(90deg, #34d399, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1;
            margin-bottom: 0.7rem;
        }
        .savings-copy {
            font-size  : 1rem;
            color      : rgba(255,255,255,0.8);
            font-weight: 500;
            font-family: var(--font-display);
        }
        .savings-band {
            margin-top   : 1.1rem;
            display      : inline-block;
            background   : rgba(255,255,255,0.08);
            padding      : 0.5rem 1rem;
            border-radius: 100px;
            font-size    : 0.8rem;
            font-weight  : 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            border       : 1px solid rgba(255,255,255,0.05);
        }

        /* Trajectory card */
        .trajectory-card {
            padding      : 1.35rem;
            text-align   : center;
            border-radius: 16px;
            margin       : 0.85rem 0;
            border       : 1px solid var(--border-soft);
            box-shadow   : 0 4px 14px rgba(0,0,0,0.02);
        }
        .trajectory-direct  { background: linear-gradient(135deg,#ecfdf5,#fff); }
        .trajectory-fallback{ background: linear-gradient(135deg,#fffbeb,#fff); }
        .trajectory-icon    { font-size: 1.8rem; margin-bottom: 0.4rem; }
        .trajectory-label   {
            font-family: var(--font-display); font-weight: 800; font-size: 1.1rem;
            color      : var(--brand-dark); margin-bottom: 0.4rem;
        }
        .trajectory-copy    { font-size: 0.88rem; color: var(--text-muted); }

        /* Comparison chip */
        .comparison-chip {
            background   : linear-gradient(90deg, #0f172a, #1e293b);
            color        : #fff;
            padding      : 1.1rem;
            border-radius: 12px;
            text-align   : center;
            font-family  : var(--font-display);
            font-weight  : 800;
            font-size    : 1.05rem;
            box-shadow   : 0 10px 24px rgba(15,23,42,0.15);
            margin       : 1.25rem 0;
            border       : 1px solid rgba(255,255,255,0.08);
            display      : block;
        }

        /* ══════════════════════════════════════════════════════════
           SNN PREDICTION PANEL
        ══════════════════════════════════════════════════════════ */
        .prediction-callout {
            border-radius: 16px;
            padding      : 1.5rem;
            text-align   : center;
            border       : 1px solid var(--border-soft);
            box-shadow   : 0 10px 30px rgba(0,0,0,0.04);
            margin-top   : 1rem;
            position     : relative;
            overflow     : hidden;
        }
        .prediction-direct  { background: linear-gradient(135deg,#f0fdf4,#fff); border-color: #a7f3d0; }
        .prediction-fallback{ background: linear-gradient(135deg,#fffbeb,#fff); border-color: #fde68a; }
        .prediction-callout .kicker {
            font-size: 0.68rem; font-weight: 800; letter-spacing: 0.1em;
            text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.5rem;
        }
        .prediction-callout .headline {
            font-family: var(--font-display); font-size: 1.35rem; font-weight: 800;
            margin-bottom: 0.5rem; color: var(--brand-dark);
        }
        .prediction-callout .copy { font-size: 0.88rem; color: var(--text-muted); }

        /* Meta pill grid */
        .meta-pill-grid {
            display              : grid;
            grid-template-columns: repeat(4, 1fr);
            gap                  : 1rem;
            margin-top           : 1.5rem;
        }
        .meta-pill {
            background     : var(--bg-card);
            padding        : 1.15rem;
            border-radius  : 14px;
            border         : 1px solid var(--border-soft);
            box-shadow     : 0 4px 14px rgba(0,0,0,0.02);
            backdrop-filter: blur(5px);
            transition     : all 0.2s ease;
        }
        .meta-pill:hover {
            border-color: var(--accent-teal);
            transform   : translateY(-2px);
        }
        .meta-pill .label {
            font-size     : 0.67rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color         : var(--text-muted);
            font-weight   : 700;
            margin-bottom : 0.4rem;
        }
        .meta-pill .value {
            font-family: var(--font-display);
            font-size  : 1rem;
            font-weight: 800;
            color      : var(--brand-dark);
        }

        /* ══════════════════════════════════════════════════════════
           COMPACT STAT CARDS  (savings panel mini-grid)
        ══════════════════════════════════════════════════════════ */
        .smc-grid {
            display       : flex;
            flex-direction: column;
            gap           : 0.55rem;
            margin        : 0.85rem 0;
        }
        .smc-card {
            background   : #fff;
            border-radius: 12px;
            padding      : 0.75rem 1rem;
            border       : 1px solid var(--border-soft);
            box-shadow   : 0 3px 10px rgba(0,0,0,0.03);
        }
        .smc-card-accent {
            background   : linear-gradient(135deg, #f0fdf4, #fff);
            border-color : rgba(16,185,129,0.2);
        }
        .smc-val {
            font-family: var(--font-display);
            font-weight: 800;
            font-size  : 1.15rem;
            color      : var(--brand-dark);
            line-height: 1.1;
        }
        .smc-card-accent .smc-val { color: #059669; }
        .smc-lbl {
            font-size     : 0.67rem;
            font-weight   : 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color         : var(--text-muted);
            margin-top    : 0.2rem;
        }
        .smc-sub {
            font-size : 0.68rem;
            color     : var(--text-muted);
            margin-top: 0.15rem;
        }

        /* ══════════════════════════════════════════════════════════
           SNN SIGNAL CARD  (direction prediction in savings panel)
        ══════════════════════════════════════════════════════════ */
        .signal-card {
            border-radius: 14px;
            padding      : 0.9rem 1rem;
            margin-bottom: 0.75rem;
            border       : 1px solid;
        }
        .signal-card-direct {
            background  : linear-gradient(135deg, #022c22, #064e3b);
            border-color: rgba(16,185,129,0.3);
        }
        .signal-card-fallback {
            background  : linear-gradient(135deg, #1c1507, #2d2010);
            border-color: rgba(245,158,11,0.3);
        }
        .signal-top {
            display        : flex;
            justify-content: space-between;
            align-items    : center;
            margin-bottom  : 0.6rem;
        }
        .signal-label {
            font-family   : var(--font-display);
            font-weight   : 800;
            font-size     : 0.65rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color         : rgba(255,255,255,0.55);
        }
        .signal-route-icon {
            font-size  : 1rem;
            color      : rgba(255,255,255,0.4);
        }
        .signal-row {
            display    : flex;
            align-items: center;
            gap        : 0;
        }
        .signal-item   { flex: 1; text-align: center; }
        .signal-val    {
            font-family: var(--font-display);
            font-weight: 800;
            font-size  : 1.05rem;
            color      : #fff;
            line-height: 1;
        }
        .signal-lbl    {
            font-size     : 0.6rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color         : rgba(255,255,255,0.4);
            margin-top    : 0.2rem;
            font-weight   : 600;
        }
        .signal-vr {
            width     : 1px;
            height    : 30px;
            background: rgba(255,255,255,0.1);
            margin    : 0 0.5rem;
            flex-shrink: 0;
        }
        .signal-note {
            font-size  : 0.62rem;
            color      : rgba(255,255,255,0.3);
            text-align : center;
            margin-top : 0.6rem;
            font-weight: 500;
        }

        /* ══════════════════════════════════════════════════════════
           MISC UTILS
        ══════════════════════════════════════════════════════════ */
        .scale-note {
            font-size : 0.83rem;
            color     : var(--text-muted);
            text-align: center;
            margin-top: 0.85rem;
            line-height: 1.5;
            font-weight: 500;
        }
        hr { border-color: var(--border-hard); margin: 2rem 0; opacity: 0.5; }

        /* ── Progress bars ─────────────────────────────────────── */
        .stProgress > div > div > div { background-color: var(--accent-teal) !important; border-radius: 100px; }
        .stProgress > div             { border-radius: 100px; background-color: rgba(0,0,0,0.05); }

        /* ── Inputs ────────────────────────────────────────────── */
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div,
        div[data-baseweb="select"] > div {
            border-radius: 14px !important;
            border       : 1px solid var(--border-soft) !important;
            background   : rgba(255,255,255,0.92) !important;
            box-shadow   : 0 8px 22px rgba(2,6,23,0.04);
        }
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {
            font-family: var(--font-body) !important;
            font-weight: 600;
        }

        /* ── Buttons ───────────────────────────────────────────── */
        .stButton > button {
            border-radius: 14px;
            font-family  : var(--font-display) !important;
            font-weight  : 800;
            letter-spacing: 0.02em;
        }
        button[kind="primary"],
        button[data-testid="baseButton-primary"] {
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-teal)) !important;
            border    : none !important;
            box-shadow: 0 14px 28px rgba(58,123,213,0.2);
        }
        button[kind="primary"]:hover,
        button[data-testid="baseButton-primary"]:hover {
            filter   : brightness(1.04);
            transform: translateY(-1px);
        }

        div[data-testid="stAlert"] {
            border-radius: 16px;
            border       : 1px solid var(--border-soft);
            box-shadow   : 0 8px 22px rgba(2,6,23,0.04);
        }

        /* ══════════════════════════════════════════════════════════
           SIDEBAR
        ══════════════════════════════════════════════════════════ */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #060d18 0%, #0a1b2e 100%);
            border-right: 1px solid rgba(0,229,204,0.08);
        }
        [data-testid="stSidebar"] * {
            color: rgba(255,255,255,0.85) !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.08) !important;
            opacity: 1 !important;
        }
        [data-testid="stSidebar"] [data-testid="stMetric"] {
            background    : rgba(255,255,255,0.05) !important;
            border-color  : rgba(255,255,255,0.08) !important;
            box-shadow    : none !important;
        }
        [data-testid="stSidebar"] [data-testid="stMetricValue"] {
            color: var(--spanda-teal) !important;
            font-size: 1.35rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
            color: rgba(255,255,255,0.5) !important;
        }
        .sidebar-brand-block {
            text-align : center;
            padding    : 0.5rem 0 1rem 0;
        }
        .sidebar-logo-img {
            width        : 110px;
            height       : 113px;
            object-fit   : contain;
            filter       : drop-shadow(0 0 16px rgba(0,229,204,0.4));
            border-radius: 12px;
            margin-bottom: 0.75rem;
        }
        .sidebar-logo-placeholder {
            width: 110px; height: 113px;
            background: rgba(0,229,204,0.06);
            border: 1px dashed rgba(0,229,204,0.25);
            border-radius: 12px;
            display: inline-flex; align-items: center; justify-content: center;
            color: rgba(0,229,204,0.5);
            font-size: 0.9rem; font-weight: 800; letter-spacing: 0.2em;
            margin-bottom: 0.75rem;
        }
        .sidebar-brand-name {
            font-family   : var(--font-display);
            font-weight   : 800;
            font-size     : 1.4rem;
            letter-spacing: 0.28em;
            color         : var(--spanda-teal) !important;
            margin-bottom : 0.2rem;
        }
        .sidebar-brand-sub {
            font-size     : 0.6rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color         : rgba(255,255,255,0.35) !important;
        }
        .sidebar-pill {
            display      : inline-flex;
            align-items  : center;
            gap          : 0.4rem;
            background   : rgba(0,229,204,0.1);
            border       : 1px solid rgba(0,229,204,0.2);
            padding      : 0.3rem 0.7rem;
            border-radius: 100px;
            font-size    : 0.7rem;
            font-weight  : 700;
            color        : var(--spanda-teal) !important;
            margin-bottom: 0.3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────────────
if "result"      not in st.session_state: st.session_state.result      = None
if "api_error"   not in st.session_state: st.session_state.api_error   = None
if "last_amount" not in st.session_state: st.session_state.last_amount = DEFAULT_AMOUNT
if "demo_mode"   not in st.session_state: st.session_state.demo_mode   = True


def _load_logo_b64() -> str:
    svg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "svgviewer-output.svg"
    )
    try:
        with open(svg_path, "r", encoding="utf-8") as f:
            content = f.read()
        b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")
        return f"data:image/svg+xml;base64,{b64}"
    except Exception:
        return ""


def scale_placeholder(amount: float, monthly_tx: int) -> dict:
    """Scale PLACEHOLDER_DATA to the chosen transaction amount."""
    ratio = amount / 1_000_000
    data  = {k: v for k, v in PLACEHOLDER_DATA.items()}
    data["usd_route_cost"] = round(36180.0 * ratio, 2)
    data["snn_route_cost"] = round(1236.0  * ratio, 2)
    data["savings_inr"]    = round(34944.0 * ratio, 2)
    data["savings_percentage"] = PLACEHOLDER_DATA["savings_percentage"]
    data["usd_route"] = {
        **PLACEHOLDER_DATA["usd_route"],
        "total_cost_inr": round(36180.0 * ratio, 2),
        "itemised": {
            k: round(v * ratio, 2)
            for k, v in PLACEHOLDER_DATA["usd_route"]["itemised"].items()
        },
    }
    data["snn_route"] = {
        **PLACEHOLDER_DATA["snn_route"],
        "total_cost_inr": round(1236.0 * ratio, 2),
        "itemised": {
            k: round(v * ratio, 2)
            for k, v in PLACEHOLDER_DATA["snn_route"]["itemised"].items()
        },
    }
    saving = round(34944.0 * ratio, 2)
    annual = round(saving * monthly_tx * 12, 2)
    data["savings"] = {
        **PLACEHOLDER_DATA["savings"],
        "amount_inr"          : saving,
        "amount_formatted"    : f"₹{saving:,.2f}",
        "annual_estimate_inr" : annual,
        "annual_formatted"    : f"₹{annual:,.2f}",
        "monthly_tx_assumption": monthly_tx,
    }
    return data


def _render_sidebar(logo_uri: str):
    """Render the SPANDA-branded sidebar."""
    with st.sidebar:
        logo_html = (
            f'<img src="{logo_uri}" class="sidebar-logo-img" alt="SPANDA">'
            if logo_uri else
            '<div class="sidebar-logo-placeholder">SPANDA</div>'
        )
        st.markdown(
            f"""
            <div class="sidebar-brand-block">
                {logo_html}
                <div class="sidebar-brand-name">SPANDA</div>
                <div class="sidebar-brand-sub">Global Unity Banking System</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Research claims
        st.markdown("##### Core Innovation")
        st.markdown(
            """
            <div style="font-size:0.82rem;line-height:1.7;color:rgba(255,255,255,0.7)!important;">
            SPANDA applies <strong style="color:#00E5CC!important;">Spiking Neural Networks</strong>
            (biologically inspired, event-driven) to BRICS cross-border settlement routing —
            achieving near-instant, low-cost INR/BRL settlement without USD intermediation.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Model metrics
        st.markdown("##### Model Metrics")
        col1, col2 = st.columns(2)
        col1.metric("Val AUC", "0.555", "vs LSTM 0.514")
        col2.metric("Parameters", "2,945", "vs 5,537")
        col1.metric("Energy", "8.95 pJ", "vs 94.46 pJ")
        col2.metric("Spike Rate", "47.8%", "avg activity")

        st.divider()

        # Key claims for patent
        st.markdown("##### Patent Claims Summary")
        claims = [
            "Event-driven spike encoding of FX price series",
            "LIF neuron decision gate for settlement routing",
            "Dual-threshold confidence filter (prob + rate)",
            "97% fee compression vs SWIFT USD route",
            "T+0 direct INR/BRL settlement finality",
        ]
        for i, claim in enumerate(claims, 1):
            st.markdown(
                f'<div class="sidebar-pill">#{i} {claim}</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # Research citation
        st.markdown("##### Research Context")
        st.markdown(
            """
            <div style="font-size:0.75rem;line-height:1.65;color:rgba(255,255,255,0.5)!important;">
            <em>Spanda</em> (Sanskrit: vibration / primordial pulse) — applied here
            to financial signal processing via neuromorphic computing.<br><br>
            Model trained on synthetic INR/BRL data. Validation-set evaluation.
            Production deployment pending regulatory approval.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.caption("Research prototype — not financial advice.")


def _load_liquidity_local() -> dict | None:
    """
    Load liquidity analysis data from local outputs/ files.

    Used as fallback when the API is unavailable or in demo mode.
    Reads outputs/backtest_results.csv and outputs/backtest_summary.json
    and returns the same structure as GET /liquidity.
    """
    try:
        import json
        import pandas as pd

        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        summary_path  = os.path.join(_root, "outputs", "backtest_summary.json")
        backtest_path = os.path.join(_root, "outputs", "backtest_results.csv")

        if not os.path.exists(summary_path) or not os.path.exists(backtest_path):
            return None

        with open(summary_path) as f:
            summary = json.load(f)

        df = pd.read_csv(backtest_path, index_col="date", parse_dates=True).sort_index()

        timeline = [
            {
                "date"               : str(idx.date()),
                "decision"           : str(row["decision"]),
                "snn_prob"           : round(float(row["snn_prob"]),          4),
                "snn_spike_rate"     : round(float(row["snn_spike_rate"]),    4),
                "saving_day"         : round(float(row["saving_day"]),        2),
                "cumulative_saving"  : round(float(row["cumulative_saving"]), 2),
                "cumulative_baseline": round(float(row["cumulative_baseline"]), 2),
                "cumulative_direct"  : round(float(row["cumulative_direct"]), 2),
                "snn_correct"        : int(row["snn_correct"]),
                "confidence"         : round(float(row["confidence"]),        4),
            }
            for idx, row in df.iterrows()
        ]

        direct_df    = df[df["decision"] == "DIRECT"]
        right_direct = direct_df[direct_df["snn_correct"] == 1]
        wrong_direct = direct_df[direct_df["snn_correct"] == 0]

        wdc = float(wrong_direct["cost_taken"].sum())   if len(wrong_direct) > 0 else 0.0
        wsc = float(wrong_direct["cost_baseline"].sum()) if len(wrong_direct) > 0 else 0.0

        risk_analysis = {
            "n_right_direct"       : int(len(right_direct)),
            "n_wrong_direct"       : int(len(wrong_direct)),
            "avg_conf_correct"     : round(float(right_direct["confidence"].mean()), 4) if len(right_direct) > 0 else 0.0,
            "avg_conf_wrong"       : round(float(wrong_direct["confidence"].mean()),  4) if len(wrong_direct) > 0 else 0.0,
            "wrong_direct_cost"    : round(wdc, 2),
            "wrong_swift_cost"     : round(wsc, 2),
            "even_wrong_saved_money": wdc < wsc,
        }

        # Savings scaling (hardcoded — same formula as cost_engine.py)
        _fee = {
            "usd": {"spread": 0.020, "flat": 1_000, "gst": 0.18,
                    "corr": 25 * 84.0, "brl": 0.015, "iof": 0.0038},
            "snn": {"fee": 0.001, "flat": 200, "gst": 0.18},
        }
        scaling = []
        for amount, label in [(100_000, "₹1 Lakh"), (1_000_000, "₹10 Lakh"), (5_000_000, "₹50 Lakh")]:
            u = _fee["usd"]
            s = _fee["snn"]
            usd_cost = (amount * u["spread"] + u["flat"] + u["flat"] * u["gst"] +
                        u["corr"] + amount * u["brl"] + amount * u["iof"])
            snn_cost = amount * s["fee"] + s["flat"] + s["flat"] * s["gst"]
            saving   = usd_cost - snn_cost
            scaling.append({
                "label"     : label,
                "amount"    : float(amount),
                "usd_cost"  : round(usd_cost, 2),
                "snn_cost"  : round(snn_cost, 2),
                "saving"    : round(saving,   2),
                "saving_pct": round(saving / usd_cost * 100, 1),
            })

        return {
            "summary"        : summary,
            "timeline"       : timeline,
            "risk_analysis"  : risk_analysis,
            "savings_scaling": scaling,
        }
    except Exception:
        return None


def main():
    logo_uri = _load_logo_b64()
    _render_sidebar(logo_uri)

    # ── ROW 1: Hero header ─────────────────────────────────────────────
    render_header()
    render_customer_banner()

    # ── API status strip ───────────────────────────────────────────────
    api_col, mode_col = st.columns([4, 1.3])
    with api_col:
        st.markdown(
            f"""
            <div class="status-strip">
                <div>
                    <div class="label">Active API Endpoint</div>
                    <div class="value">{API_URL}</div>
                </div>
                <div class="hero-tag" style="
                    background:rgba(0,229,204,0.1);
                    border:1px solid rgba(0,229,204,0.25);
                    color:#00d2ff;font-size:0.72rem;
                    padding:0.35rem 0.75rem;border-radius:100px;
                    font-weight:700;letter-spacing:0.04em;">
                    Signal-Driven Routing
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mode_col:
        demo_toggle = st.toggle(
            "Demo mode",
            value=st.session_state.demo_mode,
            help="Demo: uses scaled placeholder data. Off: calls live API.",
        )
        st.session_state.demo_mode = demo_toggle

    st.divider()

    # ── ROW 2: Transaction design studio ──────────────────────────────
    st.markdown(
        '<div class="dashboard-section-title">Transaction Design Studio</div>',
        unsafe_allow_html=True,
    )

    inp1, inp2 = st.columns([4, 2])
    with inp1:
        transaction_amount = st.slider(
            "Transaction Amount (₹)",
            min_value=MIN_AMOUNT,
            max_value=MAX_AMOUNT,
            value=DEFAULT_AMOUNT,
            step=100_000,
            format="₹%d",
            help="Adjust to see how savings scale with transaction size",
        )
        st.caption(f"Selected: ₹{transaction_amount:,.0f}")
    with inp2:
        monthly_tx = st.number_input(
            "Transactions per Month",
            min_value=1,
            max_value=200,
            value=DEFAULT_MONTHLY_TX,
            help="Used for annual saving estimate",
        )

    if not st.session_state.demo_mode:
        st.markdown("**Last 10 INR/BRL closing prices (synthetic):**")
        st.caption(
            "Enter the 10 most recent daily INR/BRL closing rates, "
            "oldest first. Typical range: 15.0 – 18.0."
        )
        price_input_str = st.text_input(
            "Comma-separated prices",
            value=", ".join(map(str, DEMO_PRICE_SEQUENCE)),
            help="Example: 16.42, 16.38, 16.51, ...",
        )
        analyse_btn = st.button(
            "Analyse Settlement Route",
            type="primary",
            use_container_width=True,
        )
        if analyse_btn:
            prices, val_err = validate_price_input(price_input_str)
            if val_err:
                st.error(f"❌ {val_err}")
                st.stop()
            with st.spinner("Running SNN analysis… (first call may take 30 s if API is waking up)"):
                t0 = time.time()
                result, error = call_predict_api(
                    price_sequence=prices,
                    transaction_amount=transaction_amount,
                    monthly_tx_count=monthly_tx,
                )
                elapsed = time.time() - t0
            if error:
                st.session_state.api_error = error
                st.session_state.result    = None
            else:
                st.session_state.result    = result
                st.session_state.api_error = None
                st.success(f"✅ Analysis complete ({elapsed:.1f}s)")
        if st.session_state.api_error:
            st.error(f"⚠️ {st.session_state.api_error}")
            st.info("Switching to demo mode while API is unavailable.")

    st.divider()

    # ── Resolve display data ───────────────────────────────────────────
    if st.session_state.demo_mode:
        data = scale_placeholder(transaction_amount, monthly_tx)
        st.info(
            "**Demo mode** — showing placeholder data scaled to your transaction amount. "
            "Toggle 'Demo mode' off to call the live API."
        )
    elif st.session_state.result is not None:
        data = st.session_state.result
        data["savings"]["monthly_tx_assumption"] = monthly_tx
        data["savings"]["annual_estimate_inr"] = round(
            data["savings_inr"] * monthly_tx * 12, 2
        )
        data["savings"]["annual_formatted"] = (
            f"₹{data['savings']['annual_estimate_inr']:,.2f}"
        )
    else:
        data = scale_placeholder(transaction_amount, monthly_tx)
        if not st.session_state.demo_mode:
            st.info("Enter prices and click **Analyse** to get a live prediction.")

    # ── ROW 3: Executive snapshot ──────────────────────────────────────
    st.markdown(
        '<div class="dashboard-section-title">Executive Snapshot</div>',
        unsafe_allow_html=True,
    )
    render_metrics_row(data)
    st.divider()

    # ── ROW 4: Route comparison theater ───────────────────────────────
    st.markdown(
        '<div class="dashboard-section-title">Route Comparison Theater</div>',
        unsafe_allow_html=True,
    )
    col_before, col_savings, col_after = st.columns([6, 3, 6])

    with col_before:
        render_before_panel(
            transaction_amount=transaction_amount,
            usd_route=data["usd_route"],
        )
    with col_savings:
        render_savings_panel(
            savings=data["savings"],
            recommendation=data.get("recommendation", "USD_FALLBACK"),
            snn_pred=data.get("snn_prediction"),
        )
    with col_after:
        render_after_panel(
            transaction_amount=transaction_amount,
            snn_route=data["snn_route"],
        )

    st.divider()

    # ── ROW 5: Neuromorphic decision core ─────────────────────────────
    st.markdown(
        '<div class="dashboard-section-title">Neuromorphic Decision Core</div>',
        unsafe_allow_html=True,
    )
    render_prediction_panel(data["snn_prediction"])

    st.divider()

    # ── ROW 6: Threshold sensitivity chart ────────────────────────────
    threshold_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "threshold_tradeoff.png",
    )
    if os.path.exists(threshold_path):
        with st.expander("Research: Threshold Sensitivity Analysis"):
            st.image(threshold_path, use_container_width=True)
            st.markdown(
                "**Left axis:** % of transactions routed via direct settlement at each "
                "probability threshold. **Right axis:** precision of direct routing decisions. "
                "Vertical line marks deployed threshold (0.70)."
            )

    # ── ROW 7: Liquidity Analysis ──────────────────────────────────────
    st.markdown(
        '<div class="dashboard-section-title">Liquidity Analysis — Backtest Results</div>',
        unsafe_allow_html=True,
    )

    liq_data = None
    liq_error = None

    if not st.session_state.demo_mode:
        with st.spinner("Loading liquidity analysis…"):
            liq_data, liq_error = call_liquidity_api()

    if liq_data is None:
        # Load from local outputs/ as fallback (works in demo mode and offline)
        liq_data = _load_liquidity_local()

    if liq_data:
        render_liquidity_panel(liq_data)
        if liq_error:
            st.caption(f"Note: Loaded from local cache — {liq_error}")
    else:
        st.info(
            "Liquidity analysis data not found. "
            "Run `notebooks/13_business_logic.ipynb` to generate the backtest results."
        )

    st.divider()

    # ── ROW 8: Assumptions & caveats ──────────────────────────────────
    with st.expander("Assumptions & Honest Caveats", expanded=False):
        for a in data.get("assumptions", []):
            st.markdown(f"- {a}")
        st.markdown(
            "- **Model:** Val AUC=0.555 on synthetic INR/BRL data. "
            "Test set evaluation pending (Month 3 final)."
        )
        st.markdown(
            "- **Energy:** Theoretical estimate "
            "(0.9 pJ/SynOp, Blouw et al., 2019). "
            "Not measured on neuromorphic hardware."
        )
        st.caption(f"Research prototype — not financial advice. API: {API_URL}")


if __name__ == "__main__":
    main()
