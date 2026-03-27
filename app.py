import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from registry import load_predictor, check_model_availability

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Endo Digital Twin | Clinical AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — dark glassmorphism clinical theme
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

/* ══════════════════════════════════════════
   STREAMLIT 1.50 — RELIABLE SELECTORS
   ══════════════════════════════════════════ */

/* Font injection — everything */
*, *::before, *::after {
    font-family: 'Inter', 'Space Grotesk', -apple-system, sans-serif !important;
}

/* App background */
[data-testid="stApp"] {
    background: linear-gradient(160deg, #07071a 0%, #0e0525 45%, #07101f 100%) !important;
}

/* Main content area */
section[data-testid="stMain"] {
    background: transparent !important;
}

/* Block container padding */
[data-testid="block-container"] {
    padding: 1.5rem 2.5rem 3rem !important;
    max-width: 1380px !important;
}

/* Hide Streamlit chrome */
[data-testid="stHeader"]       { display: none !important; }
[data-testid="stToolbar"]      { display: none !important; }
#MainMenu, footer, header      { display: none !important; }

/* ══════════════════════════════════════════
   TABS
   ══════════════════════════════════════════ */
[data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 4px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    margin-bottom: 1.5rem !important;
}
[data-baseweb="tab"] {
    border-radius: 10px !important;
    color: rgba(255,255,255,0.5) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.45rem 1.15rem !important;
    transition: all 0.2s !important;
    background: transparent !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(135deg, #7c3aed 0%, #1d4ed8 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 12px rgba(124,58,237,0.45) !important;
}

/* ── Hero ── */
.hero-section {
    background: linear-gradient(135deg, rgba(124,58,237,0.13) 0%, rgba(6,182,212,0.07) 100%);
    border: 1px solid rgba(192,132,252,0.22);
    border-radius: 24px;
    padding: 3.5rem 2.5rem 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-section::before {
    content: '';
    position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 60% 40%, rgba(192,132,252,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #c084fc 0%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.75rem;
}
.hero-sub {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.55);
    max-width: 640px;
    margin: 0 auto 1.75rem;
    line-height: 1.75;
}
.badge {
    display: inline-block;
    background: rgba(124,58,237,0.18);
    border: 1px solid rgba(192,132,252,0.35);
    border-radius: 20px;
    padding: 0.28rem 0.85rem;
    font-size: 0.78rem;
    color: #c084fc;
    font-weight: 500;
    margin: 0.2rem;
    letter-spacing: 0.04em;
}

/* ── Cards ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.75rem 0;
    transition: border-color 0.3s;
}
.glass-card:hover { border-color: rgba(192,132,252,0.28); }

.info-card {
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin: 0.5rem 0;
    line-height: 1.75;
}
.info-card h3 { margin-top: 0; font-size: 1.05rem; font-weight: 600; }
.info-card p, .info-card ul, .info-card li { font-size: 0.95rem; }

/* ── Prediction result ── */
@keyframes glow {
    0%   { box-shadow: 0 0 12px rgba(192,132,252,0.15); }
    50%  { box-shadow: 0 0 30px rgba(192,132,252,0.45); }
    100% { box-shadow: 0 0 12px rgba(192,132,252,0.15); }
}
.prediction-result {
    background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(6,182,212,0.1));
    border: 1px solid rgba(192,132,252,0.45);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    animation: glow 3s ease-in-out infinite;
}
.pain-score {
    font-size: 4.8rem;
    font-weight: 800;
    line-height: 1;
    margin: 0.4rem 0;
}
.pain-label {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.45);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 500;
}
.pain-delta-down { color: #34d399; font-weight: 700; font-size: 1.05rem; }
.pain-delta-up   { color: #f87171; font-weight: 700; font-size: 1.05rem; }
.pain-delta-flat { color: rgba(255,255,255,0.5); font-weight: 600; font-size: 1.05rem; }

/* ── KPI metric cards ── */
.kpi-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 0.75rem;
    text-align: center;
}
.kpi-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #c084fc;
    line-height: 1.1;
    margin: 0.2rem 0;
}
.kpi-label {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.45);
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.kpi-delta-good { font-size: 0.78rem; color: #34d399; font-weight: 600; }
.kpi-delta-bad  { font-size: 0.78rem; color: #f87171; font-weight: 600; }
.kpi-delta-flat { font-size: 0.78rem; color: rgba(255,255,255,0.4); }

/* ── Section headers ── */
.section-hdr {
    font-size: 1rem;
    font-weight: 600;
    color: #c084fc;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 0.75rem;
    border-bottom: 1px solid rgba(192,132,252,0.2);
    letter-spacing: 0.02em;
}

/* ── Insight / AI boxes ── */
.insight-box {
    background: linear-gradient(135deg, rgba(34,211,238,0.07), rgba(6,182,212,0.04));
    border: 1px solid rgba(34,211,238,0.22);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.75rem 0;
    color: rgba(255,255,255,0.82);
    line-height: 1.75;
    font-size: 0.95rem;
}
.ai-box {
    background: linear-gradient(135deg, rgba(124,58,237,0.1), rgba(109,40,217,0.07));
    border: 1px solid rgba(192,132,252,0.28);
    border-radius: 14px;
    padding: 1.5rem;
    margin: 0.75rem 0;
    color: rgba(255,255,255,0.88);
    line-height: 1.8;
    font-size: 0.96rem;
}
.disclaimer-box {
    background: rgba(251,146,60,0.07);
    border: 1px solid rgba(251,146,60,0.25);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    color: rgba(255,255,255,0.65);
    font-size: 0.84rem;
    margin: 0.75rem 0;
}

/* ══════════════════════════════════════════
   BUTTONS
   ══════════════════════════════════════════ */
.stButton button,
[data-testid="stFormSubmitButton"] button,
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #7c3aed 0%, #1d4ed8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.93rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 16px rgba(124,58,237,0.35) !important;
    letter-spacing: 0.02em !important;
}
.stButton button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    box-shadow: 0 6px 28px rgba(124,58,237,0.6) !important;
    transform: translateY(-1px) !important;
    filter: brightness(1.08) !important;
}

/* ══════════════════════════════════════════
   INPUTS
   ══════════════════════════════════════════ */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 10px !important;
    color: rgba(255,255,255,0.92) !important;
    font-size: 0.94rem !important;
    caret-color: #c084fc !important;
}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
    border-color: rgba(192,132,252,0.6) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.22) !important;
    outline: none !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div {
    background: rgba(255,255,255,0.05) !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 9px !important;
}

/* Metrics */
[data-testid="metric-container"],
[data-testid="stMetric"] {
    background: rgba(124,58,237,0.08) !important;
    border: 1px solid rgba(192,132,252,0.15) !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
}

/* Slider track */
[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #22d3ee) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(6,6,18,0.98) !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: rgba(255,255,255,0.75) !important;
    font-weight: 500 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Info / warning / success alerts */
[data-testid="stAlert"] {
    border-radius: 10px !important;
}

/* Divider */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.07) !important;
    margin: 1.5rem 0 !important;
}

/* Caption */
[data-testid="stCaptionContainer"] p,
.stCaption {
    color: rgba(255,255,255,0.38) !important;
    font-size: 0.81rem !important;
}

/* Stats counter animation */
@keyframes countUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.stat-item {
    animation: countUp 0.6s ease forwards;
}

/* Glowing separator */
.glow-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, #7c3aed, #22d3ee, transparent);
    margin: 2rem 0;
    border: none;
}

/* Pipeline step arrows */
.pipeline-arrow {
    color: rgba(192,132,252,0.5);
    font-size: 1.5rem;
    text-align: center;
    line-height: 1;
    padding-top: 1.5rem;
}

/* ── Feature input group labels ── */
.input-group-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}

/* ── Model badge ── */
.model-badge {
    display: inline-block;
    background: rgba(34,211,238,0.12);
    border: 1px solid rgba(34,211,238,0.3);
    border-radius: 6px;
    padding: 0.15rem 0.65rem;
    font-size: 0.75rem;
    color: #22d3ee;
    font-weight: 600;
    letter-spacing: 0.06em;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DATA_FILE = "checkin_data.csv"
FEATURES = ["sleep", "stress", "activity", "period_phase", "gi", "meds", "mood", "hydration"]
FEATURE_LABELS = {
    "sleep": "Sleep (hrs)",
    "stress": "Stress",
    "activity": "Activity",
    "period_phase": "Period Phase",
    "gi": "GI Symptoms",
    "meds": "NSAID",
    "mood": "Mood",
    "hydration": "Hydration (glasses)",
}
PHASE_MAP = {
    -2: "Ovulation (-2)",
    -1: "Pre-menstrual (-1)",
    0: "Menstrual (0)",
    1: "Post-menstrual (+1)",
    2: "Follicular (+2)",
}

_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk, sans-serif", color="rgba(255,255,255,0.75)", size=12),
    margin=dict(t=50, b=40, l=40, r=20),
)

# ─────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame()


def save_data(df: pd.DataFrame) -> None:
    df.to_csv(DATA_FILE, index=False)


# ─────────────────────────────────────────────
# ML HELPERS
# ─────────────────────────────────────────────

def predict_grid(
    predict_fn, base_dict: Dict, x_name: str, y_name: str,
    xs: np.ndarray, ys: np.ndarray,
) -> np.ndarray:
    grid = np.zeros((len(ys), len(xs)))
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            feats = base_dict.copy()
            feats[x_name] = xv
            feats[y_name] = yv
            grid[i, j] = predict_fn(feats)
    return grid


def compute_feature_impacts(
    predictor,
    baseline: Dict,
    scenario: Dict,
    pain_change: float,
    meta: Dict,
) -> Dict[str, float]:
    """Return per-feature contributions to pain change."""
    key_features = ["sleep", "activity", "hydration", "mood", "stress"]
    impacts: Dict[str, float] = {}
    coefficients = meta.get("coefficients", {})
    feature_importances = meta.get("feature_importances", {})

    if coefficients:
        for f in key_features:
            if f in coefficients and f in baseline and f in scenario:
                delta = scenario[f] - baseline[f]
                impacts[f] = delta * coefficients[f]
    elif feature_importances:
        total = sum(abs(v) for v in feature_importances.values()) or 1.0
        sign = 1 if pain_change >= 0 else -1
        for f in key_features:
            if f in feature_importances and f in baseline and f in scenario:
                delta = scenario[f] - baseline[f]
                rel = abs(feature_importances[f]) / total
                impacts[f] = delta * rel * sign * abs(pain_change)

    return impacts


def pain_color(score: float) -> str:
    if score <= 3:
        return "#34d399"
    if score <= 6:
        return "#fbbf24"
    return "#f87171"


# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────

def chart_shap_waterfall(
    baseline_pain: float,
    feature_impacts: Dict[str, float],
    simulated_pain: float,
) -> go.Figure:
    """SHAP-style waterfall / force chart showing per-feature contributions."""
    # Filter near-zero contributions
    significant = {k: v for k, v in feature_impacts.items() if abs(v) > 0.01}
    # Sort by absolute value descending
    sorted_items = sorted(significant.items(), key=lambda x: abs(x[1]), reverse=True)

    labels = ["Base"] + [FEATURE_LABELS.get(k, k) for k, _ in sorted_items] + ["Prediction"]
    measures = ["absolute"] + ["relative"] * len(sorted_items) + ["total"]
    values = [baseline_pain] + [v for _, v in sorted_items] + [0]

    # Colors: increasing = red (pain up), decreasing = teal (pain down), totals = purple
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            connector=dict(line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot")),
            increasing=dict(marker=dict(color="#f87171", line=dict(width=0))),
            decreasing=dict(marker=dict(color="#34d399", line=dict(width=0))),
            totals=dict(marker=dict(color="#c084fc", line=dict(width=0))),
            text=[f"{v:+.2f}" if i > 0 and i < len(values) - 1 else f"{v:.1f}" for i, v in enumerate(values)],
            textposition="outside",
            textfont=dict(size=11, color="rgba(255,255,255,0.8)"),
            hovertemplate="<b>%{x}</b><br>Contribution: %{y:+.3f} pts<extra></extra>",
        )
    )
    fig.update_layout(
        **_DARK,
        title=dict(text="SHAP Feature Contribution Waterfall", font=dict(size=14, color="rgba(255,255,255,0.8)")),
        yaxis=dict(
            title="Pain Score (0–10)",
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.12)",
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        height=360,
        showlegend=False,
    )
    return fig


def chart_comparison(baseline: float, simulated: float) -> go.Figure:
    colors = [pain_color(baseline), pain_color(simulated)]
    fig = go.Figure(
        go.Bar(
            x=["Baseline", "Your Scenario"],
            y=[baseline, simulated],
            marker=dict(
                color=colors,
                line=dict(width=0),
                opacity=0.85,
            ),
            text=[f"{baseline:.1f}", f"{simulated:.1f}"],
            textposition="auto",
            textfont=dict(color="white", size=14, family="Space Grotesk"),
            hovertemplate="<b>%{x}</b><br>Pain: %{y:.2f}/10<extra></extra>",
        )
    )
    fig.update_layout(
        **_DARK,
        title=dict(text="Baseline vs Your Scenario", font=dict(size=14, color="rgba(255,255,255,0.8)")),
        yaxis=dict(range=[0, 10], title="Pain Level (0–10)", gridcolor="rgba(255,255,255,0.06)"),
        height=300,
        showlegend=False,
    )
    return fig


def chart_radar(user_values: Dict, ideal_values: Dict) -> go.Figure:
    cats = list(user_values.keys())
    u = [user_values[c] for c in cats]
    i = [ideal_values[c] for c in cats]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=u + [u[0]],
            theta=cats + [cats[0]],
            fill="toself",
            name="You",
            line=dict(color="#22d3ee", width=2),
            fillcolor="rgba(34,211,238,0.15)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=i + [i[0]],
            theta=cats + [cats[0]],
            fill="toself",
            name="Ideal",
            line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"),
            fillcolor="rgba(255,255,255,0.04)",
        )
    )
    fig.update_layout(
        **_DARK,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.1)", tickfont=dict(size=9)),
            angularaxis=dict(linecolor="rgba(255,255,255,0.15)", gridcolor="rgba(255,255,255,0.08)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        title=dict(text="Your Profile vs Ideal Balance", font=dict(size=14, color="rgba(255,255,255,0.8)")),
        legend=dict(font=dict(size=11)),
        height=350,
        margin=dict(t=55, b=30, l=50, r=50),
    )
    return fig


def chart_gauge(score: float) -> go.Figure:
    color = pain_color(10 - score / 10)  # invert: higher balance → lower pain proxy
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            number=dict(font=dict(size=40, color="#c084fc", family="Space Grotesk"), suffix=""),
            delta=dict(reference=65, increasing=dict(color="#34d399"), decreasing=dict(color="#f87171")),
            domain={"x": [0, 1], "y": [0, 1]},
            title=dict(text="Well-being Balance Score", font=dict(size=14, color="rgba(255,255,255,0.7)")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="rgba(255,255,255,0.3)", tickfont=dict(size=10)),
                bar=dict(color="#c084fc", thickness=0.22),
                bgcolor="rgba(255,255,255,0.04)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 40], color="rgba(248,113,113,0.15)"),
                    dict(range=[40, 70], color="rgba(251,191,36,0.12)"),
                    dict(range=[70, 100], color="rgba(52,211,153,0.15)"),
                ],
                threshold=dict(line=dict(color="#c084fc", width=2), thickness=0.75, value=score),
            ),
        )
    )
    fig.update_layout(
        **_DARK,
        height=280,
        margin=dict(t=45, b=20, l=20, r=20),
    )
    return fig


def chart_heatmap(
    predictor, base_dict: Dict, x_name: str, y_name: str,
    x_val: float, y_val: float, bx: float, by: float,
) -> go.Figure:
    ranges = {
        "sleep": (3.0, 11.0, 25),
        "stress": (0.0, 10.0, 25),
        "activity": (0.0, 10.0, 25),
        "hydration": (0.0, 12.0, 25),
        "mood": (0.0, 10.0, 25),
    }
    xmin, xmax, xn = ranges[x_name]
    ymin, ymax, yn = ranges[y_name]
    xs = np.linspace(xmin, xmax, xn)
    ys = np.linspace(ymin, ymax, yn)
    grid = predict_grid(predictor.predict, base_dict, x_name, y_name, xs, ys)

    fig = go.Figure(
        go.Heatmap(
            z=grid, x=xs, y=ys,
            colorscale=[
                [0.0,  "#1e3a5f"],
                [0.25, "#2563eb"],
                [0.5,  "#7c3aed"],
                [0.75, "#dc2626"],
                [1.0,  "#7f1d1d"],
            ],
            zmin=0, zmax=10,
            colorbar=dict(
                title=dict(text="Predicted Pain", font=dict(color="rgba(255,255,255,0.6)", size=11)),
                tickfont=dict(color="rgba(255,255,255,0.5)"),
                thickness=14,
            ),
            hovertemplate=(
                f"{FEATURE_LABELS.get(x_name, x_name)}: %{{x:.1f}}<br>"
                f"{FEATURE_LABELS.get(y_name, y_name)}: %{{y:.1f}}<br>"
                "Predicted Pain: %{z:.2f}/10<extra></extra>"
            ),
        )
    )
    # Scenario marker
    fig.add_trace(
        go.Scatter(
            x=[x_val], y=[y_val], mode="markers+text",
            marker=dict(symbol="x", size=14, color="white", line=dict(width=2.5, color="rgba(0,0,0,0.6)")),
            text=["You"], textposition="top center",
            textfont=dict(color="white", size=11),
            name="Your Scenario", showlegend=True,
        )
    )
    # Baseline marker
    fig.add_trace(
        go.Scatter(
            x=[bx], y=[by], mode="markers+text",
            marker=dict(symbol="circle", size=11, color="rgba(255,255,255,0.7)", line=dict(width=2, color="rgba(0,0,0,0.5)")),
            text=["Baseline"], textposition="top center",
            textfont=dict(color="rgba(255,255,255,0.7)", size=11),
            name="Baseline", showlegend=True,
        )
    )
    fig.update_layout(
        **_DARK,
        title=dict(text="Pain Landscape: How Feature Pairs Interact", font=dict(size=14, color="rgba(255,255,255,0.8)")),
        xaxis=dict(title=FEATURE_LABELS.get(x_name, x_name), gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title=FEATURE_LABELS.get(y_name, y_name), gridcolor="rgba(255,255,255,0.04)"),
        legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        height=420,
    )
    return fig


def chart_pain_timeline(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["pain_level"],
            mode="lines+markers",
            line=dict(color="#c084fc", width=2.5),
            marker=dict(size=7, color="#c084fc", line=dict(width=1.5, color="rgba(0,0,0,0.4)")),
            fill="tozeroy",
            fillcolor="rgba(192,132,252,0.08)",
            hovertemplate="<b>%{x}</b><br>Pain: %{y}/10<extra></extra>",
            name="Pain Level",
        )
    )
    fig.update_layout(
        **_DARK,
        title=dict(text="Pain Level Over Time", font=dict(size=14, color="rgba(255,255,255,0.8)")),
        xaxis=dict(title="Date", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Pain (0–10)", range=[0, 10], gridcolor="rgba(255,255,255,0.06)"),
        height=320,
    )
    return fig


def chart_pain_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=df["pain_level"], nbinsx=11,
            marker=dict(
                color="rgba(34,211,238,0.7)",
                line=dict(color="rgba(34,211,238,0.3)", width=1),
            ),
            hovertemplate="Pain %{x}: %{y} entries<extra></extra>",
        )
    )
    fig.update_layout(
        **_DARK,
        title=dict(text="Pain Level Distribution", font=dict(size=14, color="rgba(255,255,255,0.8)")),
        xaxis=dict(title="Pain Level (0–10)", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.06)"),
        height=320,
    )
    return fig


def chart_sleep_vs_pain(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=df["sleep_hours"], y=df["pain_level"],
            mode="markers",
            marker=dict(
                size=8,
                color=df["pain_level"],
                colorscale=[[0, "#22d3ee"], [0.5, "#c084fc"], [1.0, "#f87171"]],
                showscale=True,
                colorbar=dict(title="Pain", thickness=10, tickfont=dict(size=10)),
                opacity=0.8,
                line=dict(width=0),
            ),
            hovertemplate="Sleep: %{x:.1f}h<br>Pain: %{y}/10<extra></extra>",
        )
    )
    # Trend line
    if len(df) >= 5 and df["sleep_hours"].nunique() >= 3:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
        z = np.polyfit(df["sleep_hours"].fillna(8), df["pain_level"].fillna(5), 1)
        p = np.poly1d(z)
        xs = np.linspace(df["sleep_hours"].min(), df["sleep_hours"].max(), 50)
        fig.add_trace(go.Scatter(
            x=xs, y=p(xs),
            mode="lines",
            line=dict(color="rgba(255,255,255,0.25)", width=1.5, dash="dot"),
            name="Trend", showlegend=False,
        ))
    fig.update_layout(
        **_DARK,
        title=dict(text="Sleep vs Pain (scatter)", font=dict(size=14, color="rgba(255,255,255,0.8)")),
        xaxis=dict(title="Sleep Hours", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Pain Level", range=[0, 10], gridcolor="rgba(255,255,255,0.06)"),
        height=320,
    )
    return fig


# ─────────────────────────────────────────────
# CLAUDE AI LAYER
# ─────────────────────────────────────────────

def get_claude_insights(
    symptoms_text: str,
    sim_features: Dict,
    sim_pain: float,
    pain_change: float,
    model_name: str,
) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return (
            "**ANTHROPIC_API_KEY not set.**\n\n"
            "To enable AI insights, add your Anthropic API key to your environment:\n"
            "```\nexport ANTHROPIC_API_KEY=sk-ant-...\n```\n"
            "Then restart the app."
        )
    try:
        import anthropic  # lazy import to keep startup fast if key not set
        client = anthropic.Anthropic(api_key=api_key)

        phase_label = PHASE_MAP.get(int(sim_features.get("period_phase", 0)), "Unknown")
        feature_summary = (
            f"- Sleep: {sim_features.get('sleep', 8):.1f} hours/night\n"
            f"- Stress level: {sim_features.get('stress', 5)}/10\n"
            f"- Physical activity: {sim_features.get('activity', 5)}/10\n"
            f"- Mood: {sim_features.get('mood', 7)}/10\n"
            f"- Hydration: {sim_features.get('hydration', 8):.0f} glasses/day\n"
            f"- Menstrual phase: {phase_label}\n"
            f"- GI symptoms: {'Yes' if sim_features.get('gi', 0) else 'No'}\n"
            f"- NSAID medication: {'Yes' if sim_features.get('meds', 0) else 'No'}\n"
        )

        user_message = (
            f"My endometriosis digital twin simulation results:\n\n"
            f"Lifestyle inputs:\n{feature_summary}\n"
            f"ML model used: {model_name}\n"
            f"Predicted pain: {sim_pain:.1f}/10 "
            f"(change from baseline: {pain_change:+.2f} points)\n\n"
            f"My symptom description: {symptoms_text if symptoms_text.strip() else 'No additional description provided.'}\n\n"
            f"Please give me 3–4 evidence-based, personalized insights about my symptom pattern and "
            f"actionable lifestyle suggestions I can try."
        )

        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=450,
            system=(
                "You are an empathetic AI health assistant specializing in endometriosis symptom management. "
                "You help users understand how their lifestyle factors relate to endometriosis symptom patterns. "
                "Provide 3–4 personalized, evidence-based bullet-point insights (total ~160–200 words). "
                "Be warm, practical, and specific to the user's inputs. "
                "Always end with a single line: 'Remember: This is educational only — please consult your healthcare provider for medical decisions.' "
                "Never give diagnosis or prescribe treatments."
            ),
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text
    except ImportError:
        return "anthropic package not installed. Run: pip install anthropic"
    except Exception as exc:
        return f"AI insights unavailable: {exc}"


# ─────────────────────────────────────────────
# DOCTOR SUMMARY (rule-based, shown alongside AI)
# ─────────────────────────────────────────────

def doctor_summary(
    simulated_pain: float,
    baseline_pain: float,
    pain_change: float,
    pi_halfwidth: float,
    scenario: Dict,
) -> Tuple[str, List[str]]:
    summary = (
        f"Predicted pain: **{simulated_pain:.1f}/10** "
        f"(Δ vs baseline: **{pain_change:+.2f}**, uncertainty ±{pi_halfwidth:.2f})."
    )
    suggestions = []
    if scenario.get("sleep", 8) < 7:
        suggestions.append("Aim for 7–9 hours of sleep — consistent sleep quality is strongly linked to lower pain sensitisation.")
    if scenario.get("stress", 5) >= 7:
        suggestions.append("High stress levels were flagged. Brief daily practices (5–10 min box breathing, gentle yoga) may reduce cortisol-mediated pain amplification.")
    if scenario.get("hydration", 8) < 6:
        suggestions.append("Low hydration can worsen cramping and fatigue. Gradually increase fluid intake to 8–10 glasses/day.")
    if scenario.get("activity", 5) < 3:
        suggestions.append("Very low activity detected. Light movement (10-min walks, stretching) releases endorphins that act as natural analgesics.")
    if scenario.get("mood", 7) <= 4:
        suggestions.append("Low mood may amplify pain perception. Small mood-boosting activities — time outdoors, social connection — can have measurable effects.")
    if not suggestions:
        suggestions.append("Your inputs are near optimal ranges. Maintaining consistent routines and tracking patterns over time remains key.")
    return summary, suggestions


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

def tab_intro() -> None:

    # ── HERO ──────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="hero-section">

  <!-- Top label -->
  <div style="margin-bottom:1.2rem;">
    <span style="
      display:inline-block;
      background:rgba(124,58,237,0.18);
      border:1px solid rgba(192,132,252,0.35);
      border-radius:30px;
      padding:0.3rem 1.1rem;
      font-size:0.76rem;
      color:#c084fc;
      font-weight:600;
      letter-spacing:0.1em;
      text-transform:uppercase;
    ">Clinical AI &nbsp;·&nbsp; Endometriosis Research Tool</span>
  </div>

  <!-- Title -->
  <div class="hero-title">Endo Digital Twin</div>
  <div class="hero-sub">
    Simulate how lifestyle factors shape endometriosis pain — powered by
    interpretable ML models and personalised Claude AI insights.
  </div>

  <!-- Stats grid -->
  <div style="
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:1rem;
    max-width:780px;
    margin:2rem auto 2rem;
  ">
    <div class="stat-item" style="
      background:rgba(124,58,237,0.12);
      border:1px solid rgba(192,132,252,0.25);
      border-radius:14px;
      padding:1.25rem 0.75rem;
      text-align:center;
      animation-delay:0.0s;
    ">
      <div style="font-size:1.85rem;font-weight:800;color:#c084fc;line-height:1;">1 in 10</div>
      <div style="font-size:0.72rem;color:rgba(255,255,255,0.5);margin-top:0.3rem;text-transform:uppercase;letter-spacing:0.06em;">Women Affected</div>
    </div>
    <div class="stat-item" style="
      background:rgba(34,211,238,0.1);
      border:1px solid rgba(34,211,238,0.22);
      border-radius:14px;
      padding:1.25rem 0.75rem;
      text-align:center;
      animation-delay:0.1s;
    ">
      <div style="font-size:1.85rem;font-weight:800;color:#22d3ee;line-height:1;">8</div>
      <div style="font-size:0.72rem;color:rgba(255,255,255,0.5);margin-top:0.3rem;text-transform:uppercase;letter-spacing:0.06em;">Lifestyle Features</div>
    </div>
    <div class="stat-item" style="
      background:rgba(52,211,153,0.1);
      border:1px solid rgba(52,211,153,0.22);
      border-radius:14px;
      padding:1.25rem 0.75rem;
      text-align:center;
      animation-delay:0.2s;
    ">
      <div style="font-size:1.85rem;font-weight:800;color:#34d399;line-height:1;">8,000</div>
      <div style="font-size:0.72rem;color:rgba(255,255,255,0.5);margin-top:0.3rem;text-transform:uppercase;letter-spacing:0.06em;">Training Samples</div>
    </div>
    <div class="stat-item" style="
      background:rgba(251,146,60,0.1);
      border:1px solid rgba(251,146,60,0.22);
      border-radius:14px;
      padding:1.25rem 0.75rem;
      text-align:center;
      animation-delay:0.3s;
    ">
      <div style="font-size:1.85rem;font-weight:800;color:#fb923c;line-height:1;">2</div>
      <div style="font-size:0.72rem;color:rgba(255,255,255,0.5);margin-top:0.3rem;text-transform:uppercase;letter-spacing:0.06em;">ML Models</div>
    </div>
  </div>

  <!-- Tech badges -->
  <div>
    <span class="badge">ElasticNet</span>
    <span class="badge">RandomForest</span>
    <span class="badge">SHAP Waterfall</span>
    <span class="badge">Claude AI</span>
    <span class="badge">Digital Twin</span>
    <span class="badge">AWS EC2</span>
  </div>

</div>
""", unsafe_allow_html=True)

    # ── GLOWING SEPARATOR ─────────────────────────────────────────────────────
    st.markdown('<div class="glow-line"></div>', unsafe_allow_html=True)

    # ── HOW IT WORKS — 3-step pipeline ────────────────────────────────────────
    st.markdown("""
<div style="text-align:center;margin-bottom:1.25rem;">
  <span style="font-size:0.75rem;font-weight:700;color:rgba(255,255,255,0.35);
    text-transform:uppercase;letter-spacing:0.12em;">How It Works</span>
</div>
""", unsafe_allow_html=True)

    s1, arr1, s2, arr2, s3 = st.columns([4, 1, 4, 1, 4])

    with s1:
        st.markdown("""
<div class="glass-card" style="text-align:center;padding:1.6rem 1rem;border-color:rgba(124,58,237,0.3);">
  <div style="font-size:2.2rem;margin-bottom:0.6rem;">🎛️</div>
  <div style="font-weight:700;color:#c084fc;font-size:1rem;margin-bottom:0.5rem;">Step 1 — Input</div>
  <div style="font-size:0.85rem;color:rgba(255,255,255,0.55);line-height:1.6;">
    Adjust 8 lifestyle sliders: sleep, stress, activity, hydration, mood,
    cycle phase, GI symptoms, NSAID use.
  </div>
</div>""", unsafe_allow_html=True)

    with arr1:
        st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)

    with s2:
        st.markdown("""
<div class="glass-card" style="text-align:center;padding:1.6rem 1rem;border-color:rgba(34,211,238,0.3);">
  <div style="font-size:2.2rem;margin-bottom:0.6rem;">🧠</div>
  <div style="font-weight:700;color:#22d3ee;font-size:1rem;margin-bottom:0.5rem;">Step 2 — Predict</div>
  <div style="font-size:0.85rem;color:rgba(255,255,255,0.55);line-height:1.6;">
    ElasticNet or RandomForest model predicts pain (0–10) and SHAP attribution
    shows each feature's exact contribution.
  </div>
</div>""", unsafe_allow_html=True)

    with arr2:
        st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)

    with s3:
        st.markdown("""
<div class="glass-card" style="text-align:center;padding:1.6rem 1rem;border-color:rgba(52,211,153,0.3);">
  <div style="font-size:2.2rem;margin-bottom:0.6rem;">✨</div>
  <div style="font-weight:700;color:#34d399;font-size:1rem;margin-bottom:0.5rem;">Step 3 — Insight</div>
  <div style="font-size:0.85rem;color:rgba(255,255,255,0.55);line-height:1.6;">
    Claude AI reads your inputs and pain score, then delivers personalised,
    evidence-based lifestyle recommendations.
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="glow-line"></div>', unsafe_allow_html=True)

    # ── FEATURE CARDS ─────────────────────────────────────────────────────────
    st.markdown("""
<div style="text-align:center;margin-bottom:1.25rem;">
  <span style="font-size:0.75rem;font-weight:700;color:rgba(255,255,255,0.35);
    text-transform:uppercase;letter-spacing:0.12em;">App Features</span>
</div>
""", unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    feat_cards = [
        ("🧪", "#7c3aed", "rgba(124,58,237,0.12)", "rgba(124,58,237,0.28)",
         "Simulator",
         "Run what-if scenarios. Instantly see how lifestyle changes shift predicted pain with a SHAP waterfall breakdown."),
        ("💬", "#22d3ee", "rgba(34,211,238,0.08)", "rgba(34,211,238,0.22)",
         "AI Insights",
         "Describe symptoms in plain English. Claude AI combines your description with simulation data for personalised advice."),
        ("📊", "#34d399", "rgba(52,211,153,0.08)", "rgba(52,211,153,0.22)",
         "My Tracker",
         "Log daily check-ins. Visualise pain trends, sleep patterns, and stress correlations over time."),
        ("🗺️", "#fb923c", "rgba(251,146,60,0.08)", "rgba(251,146,60,0.22)",
         "Pain Landscape",
         "2D heatmap shows how any pair of features interact across their full range — find your personal leverage points."),
    ]
    for col, (icon, color, bg, border, title, desc) in zip([f1, f2, f3, f4], feat_cards):
        with col:
            st.markdown(f"""
<div class="glass-card" style="
  background:{bg};
  border-color:{border};
  text-align:center;
  padding:1.5rem 1rem;
  height:100%;
">
  <div style="font-size:2rem;margin-bottom:0.6rem;">{icon}</div>
  <div style="font-weight:700;color:{color};font-size:0.97rem;margin-bottom:0.5rem;">{title}</div>
  <div style="font-size:0.82rem;color:rgba(255,255,255,0.52);line-height:1.65;">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="glow-line"></div>', unsafe_allow_html=True)

    # ── ML MODEL COMPARISON TABLE ─────────────────────────────────────────────
    st.markdown("""
<div style="text-align:center;margin-bottom:1.25rem;">
  <span style="font-size:0.75rem;font-weight:700;color:rgba(255,255,255,0.35);
    text-transform:uppercase;letter-spacing:0.12em;">ML Models</span>
</div>
""", unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("""
<div class="glass-card" style="border-color:rgba(192,132,252,0.28);padding:1.5rem;">
  <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem;">
    <div style="
      background:rgba(124,58,237,0.2);border:1px solid rgba(192,132,252,0.4);
      border-radius:8px;padding:0.3rem 0.7rem;font-size:0.75rem;
      color:#c084fc;font-weight:700;letter-spacing:0.05em;
    ">ElasticNet</div>
    <div style="font-size:0.78rem;color:rgba(255,255,255,0.4);">Interpretable Linear Model</div>
  </div>
  <div style="display:flex;flex-direction:column;gap:0.5rem;">
    <div style="display:flex;align-items:center;gap:0.5rem;font-size:0.85rem;color:rgba(255,255,255,0.7);">
      <span style="color:#34d399;font-size:0.7rem;">●</span> StandardScaler + L1/L2 regularisation
    </div>
    <div style="display:flex;align-items:center;gap:0.5rem;font-size:0.85rem;color:rgba(255,255,255,0.7);">
      <span style="color:#34d399;font-size:0.7rem;">●</span> Exact per-feature coefficients → true SHAP
    </div>
    <div style="display:flex;align-items:center;gap:0.5rem;font-size:0.85rem;color:rgba(255,255,255,0.7);">
      <span style="color:#34d399;font-size:0.7rem;">●</span> Best for understanding directional effects
    </div>
    <div style="display:flex;align-items:center;gap:0.5rem;font-size:0.85rem;color:rgba(255,255,255,0.7);">
      <span style="color:#34d399;font-size:0.7rem;">●</span> α = 0.12, L1 ratio = 0.2
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    with m2:
        st.markdown("""
<div class="glass-card" style="border-color:rgba(34,211,238,0.25);padding:1.5rem;">
  <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem;">
    <div style="
      background:rgba(34,211,238,0.12);border:1px solid rgba(34,211,238,0.35);
      border-radius:8px;padding:0.3rem 0.7rem;font-size:0.75rem;
      color:#22d3ee;font-weight:700;letter-spacing:0.05em;
    ">RandomForest</div>
    <div style="font-size:0.78rem;color:rgba(255,255,255,0.4);">Non-linear Ensemble</div>
  </div>
  <div style="display:flex;flex-direction:column;gap:0.5rem;">
    <div style="display:flex;align-items:center;gap:0.5rem;font-size:0.85rem;color:rgba(255,255,255,0.7);">
      <span style="color:#22d3ee;font-size:0.7rem;">●</span> 300 trees, max_depth=10
    </div>
    <div style="display:flex;align-items:center;gap:0.5rem;font-size:0.85rem;color:rgba(255,255,255,0.7);">
      <span style="color:#22d3ee;font-size:0.7rem;">●</span> Captures non-linear feature interactions
    </div>
    <div style="display:flex;align-items:center;gap:0.5rem;font-size:0.85rem;color:rgba(255,255,255,0.7);">
      <span style="color:#22d3ee;font-size:0.7rem;">●</span> Feature importances → SHAP estimates
    </div>
    <div style="display:flex;align-items:center;gap:0.5rem;font-size:0.85rem;color:rgba(255,255,255,0.7);">
      <span style="color:#22d3ee;font-size:0.7rem;">●</span> Better absolute pain level accuracy
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── DISCLAIMER ────────────────────────────────────────────────────────────
    st.markdown("""
<div class="disclaimer-box" style="margin-top:1rem;">
  ⚠️ <strong>Educational purposes only.</strong> This tool is not medical advice.
  Models are trained on synthetic data. Always consult a qualified healthcare
  professional for decisions about your health.
</div>
""", unsafe_allow_html=True)

    # ── References ────────────────────────────────────────────────────────────
    with st.expander("📚 Scientific References", expanded=False):
        st.markdown("""
- Bulun S.E. et al. (2019). *Endometriosis.* Endocrine Reviews, 40(4), 1048–1079.
- Zondervan K.T. et al. (2020). *Endometriosis.* Nature Reviews Disease Primers, 6(1), 9.
- Nodler J.L. et al. (2020). *Lifestyle modifications for endometriosis.* Curr Opin Obstet Gynecol, 32(4).
- Nnoaham K.E. et al. (2011). *Impact of endometriosis on quality of life.* Hum Reprod, 26(10).
""")


# ─────────────────────────────────────────────

def tab_simulator() -> None:
    # Header
    st.markdown(
        """
<div style="margin-bottom:1.5rem;">
    <div style="font-size:1.5rem; font-weight:700; color:rgba(255,255,255,0.9);">🧪 Digital Twin Simulator</div>
    <div style="color:rgba(255,255,255,0.45); font-size:0.93rem; margin-top:0.3rem;">
        Adjust lifestyle factors below to simulate their combined effect on predicted pain.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    model_availability = check_model_availability()
    available_models = [m for m, ok in model_availability.items() if ok]

    if not available_models:
        st.warning(
            "No trained models found. Run the training scripts first:\n"
            "```\npython src/synth.py\npython src/train_elasticnet.py\npython src/train_rf.py\n```"
        )
        return

    # ── Input form ──
    with st.form("sim_form", border=False):
        col_left, col_right = st.columns([1, 3])
        with col_left:
            model_choice = st.selectbox(
                "Model",
                available_models,
                help="ElasticNet → interpretable coefficients. RandomForest → non-linear pattern capture.",
            )

        st.markdown('<div class="section-hdr">Input Your Daily Profile</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="input-group-label">Physical Health</div>', unsafe_allow_html=True)
            sleep_sim = st.slider("😴 Sleep Hours", 3.0, 11.0, 8.0, 0.5)
            activity_sim = st.slider("🏃 Activity Level", 0, 10, 5)
            hydration_sim = st.slider("💧 Hydration (glasses)", 0, 12, 8)

        with c2:
            st.markdown('<div class="input-group-label">Mental & Emotional</div>', unsafe_allow_html=True)
            stress_sim = st.slider("😰 Stress Level", 0, 10, 5)
            mood_sim = st.slider("😊 Mood Level", 0, 10, 7)
            st.markdown("<br>", unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="input-group-label">Symptoms & Cycle</div>', unsafe_allow_html=True)
            period_sim = st.selectbox(
                "📅 Menstrual Phase",
                list(PHASE_MAP.keys()),
                format_func=lambda x: PHASE_MAP[x],
                index=2,
            )
            gi_sim = st.checkbox("🤢 GI Symptoms Present")
            meds_sim = st.checkbox("💊 Taking NSAID")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🔮  Generate Prediction", use_container_width=True, type="primary"
        )

    # ── Results ──
    if not submitted:
        st.markdown(
            '<div class="insight-box" style="text-align:center; padding:2rem;">👆 Adjust the sliders above and click <strong>Generate Prediction</strong> to see your results.</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        predictor = load_predictor(model_choice)
        meta = predictor.get_metadata()

        BASELINE = dict(sleep=8.0, stress=5, activity=5, period_phase=0, gi=0, meds=0, mood=7, hydration=8)
        scenario = dict(
            sleep=sleep_sim,
            stress=stress_sim,
            activity=activity_sim,
            period_phase=period_sim,
            gi=1 if gi_sim else 0,
            meds=1 if meds_sim else 0,
            mood=mood_sim,
            hydration=hydration_sim,
        )

        baseline_pain = predictor.predict(BASELINE)
        simulated_pain = predictor.predict(scenario)
        pain_change = simulated_pain - baseline_pain
        pi_hw = meta.get("pi_halfwidth", 0.7)

        feature_impacts = compute_feature_impacts(predictor, BASELINE, scenario, pain_change, meta)

        # ── Top result card ──
        score_color = pain_color(simulated_pain)
        delta_dir = "↓" if pain_change < -0.05 else ("↑" if pain_change > 0.05 else "→")
        delta_class = "pain-delta-down" if pain_change < -0.05 else ("pain-delta-up" if pain_change > 0.05 else "pain-delta-flat")
        delta_text = f"{delta_dir} {abs(pain_change):.2f} pts vs baseline"

        col_main, col_info = st.columns([1, 2])
        with col_main:
            st.markdown(
                f"""
<div class="prediction-result">
    <div class="pain-label">Predicted Pain Score</div>
    <div class="pain-score" style="color:{score_color};">{simulated_pain:.1f}</div>
    <div class="pain-label">/ 10</div>
    <div style="margin-top:0.75rem;">
        <span class="{delta_class}">{delta_text}</span>
    </div>
    <div style="margin-top:0.5rem; font-size:0.8rem; color:rgba(255,255,255,0.35);">
        Uncertainty ±{pi_hw:.2f} &nbsp;|&nbsp;
        <span class="model-badge">{model_choice}</span>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

        with col_info:
            # KPI row
            sleep_eff = feature_impacts.get("sleep", 0)
            stress_eff = feature_impacts.get("stress", 0)
            act_eff = feature_impacts.get("activity", 0)
            hyd_eff = feature_impacts.get("hydration", 0)
            mood_eff = feature_impacts.get("mood", 0)

            def kpi_html(label: str, val: float, units: str = "") -> str:
                dc = "kpi-delta-good" if val <= 0 else "kpi-delta-bad"
                sign = "+" if val > 0 else ""
                return f"""
<div class="kpi-card">
    <div class="kpi-label">{label}</div>
    <div class="kpi-value">{sign}{val:.2f}</div>
    <div class="kpi-delta-good" style="color:{'#34d399' if val<0 else '#f87171' if val>0 else 'rgba(255,255,255,0.4)'}">
        {"reduces" if val<0 else "increases" if val>0 else "neutral"} pain
    </div>
</div>"""

            k1, k2, k3, k4, k5 = st.columns(5)
            for col, label, val in zip(
                [k1, k2, k3, k4, k5],
                ["Sleep", "Stress", "Activity", "Hydration", "Mood"],
                [sleep_eff, stress_eff, act_eff, hyd_eff, mood_eff],
            ):
                with col:
                    st.markdown(kpi_html(label, val), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SHAP waterfall + Comparison bar ──
        st.markdown('<div class="section-hdr">SHAP Feature Attribution</div>', unsafe_allow_html=True)
        col_shap, col_bar = st.columns([3, 2])
        with col_shap:
            st.plotly_chart(
                chart_shap_waterfall(baseline_pain, feature_impacts, simulated_pain),
                use_container_width=True,
            )
            st.caption(
                "The waterfall shows each feature's marginal contribution to pain change from baseline. "
                "Green bars reduce pain; red bars increase it. ElasticNet values are exact; RandomForest values are importance-weighted estimates."
            )
        with col_bar:
            st.plotly_chart(chart_comparison(baseline_pain, simulated_pain), use_container_width=True)
            # Balance gauge
            balance = (
                (sleep_sim / 11.0) * 20
                + (activity_sim / 10.0) * 15
                + (hydration_sim / 12.0) * 15
                + (mood_sim / 10.0) * 20
                + ((10 - stress_sim) / 10.0) * 30
            )
            balance = max(0, min(100, balance))
            st.plotly_chart(chart_gauge(balance), use_container_width=True)

        st.divider()

        # ── Radar + Heatmap ──
        st.markdown('<div class="section-hdr">Profile & Pain Landscape</div>', unsafe_allow_html=True)
        col_radar, col_heat = st.columns([1, 2])

        with col_radar:
            user_profile = {
                "Sleep": sleep_sim,
                "Activity": activity_sim,
                "Hydration": min(hydration_sim, 10),
                "Mood": mood_sim,
                "Low Stress": 10 - stress_sim,
            }
            ideal_profile = dict(Sleep=8.0, Activity=8.0, Hydration=10.0, Mood=8.0, **{"Low Stress": 8.0})
            st.plotly_chart(chart_radar(user_profile, ideal_profile), use_container_width=True)

        with col_heat:
            available_feat = ["sleep", "stress", "activity", "hydration", "mood"]
            hc1, hc2 = st.columns(2)
            with hc1:
                x_feat = st.selectbox("X axis", available_feat, format_func=lambda x: FEATURE_LABELS.get(x, x), key="hx")
            with hc2:
                y_opts = [f for f in available_feat if f != x_feat]
                y_feat = st.selectbox("Y axis", y_opts, format_func=lambda x: FEATURE_LABELS.get(x, x), key="hy")
            st.plotly_chart(
                chart_heatmap(
                    predictor, scenario.copy(), x_feat, y_feat,
                    float(scenario.get(x_feat, 0)), float(scenario.get(y_feat, 0)),
                    float(BASELINE.get(x_feat, 0)), float(BASELINE.get(y_feat, 0)),
                ),
                use_container_width=True,
            )
            st.caption("Heatmap holds all other features constant at your scenario values. Explore different axis pairs to find leverage points.")

        st.divider()

        # ── Doctor summary ──
        st.markdown('<div class="section-hdr">Clinical Summary (Educational)</div>', unsafe_allow_html=True)
        summary_text, suggestions = doctor_summary(simulated_pain, baseline_pain, pain_change, pi_hw, scenario)
        st.markdown(
            f'<div class="insight-box"><strong>Model Assessment:</strong> {summary_text}</div>',
            unsafe_allow_html=True,
        )
        if suggestions:
            for s in suggestions:
                st.markdown(f"• {s}")

        st.divider()

        # ── AI Insights (Claude) ──
        st.markdown('<div class="section-hdr">🤖 AI Personalised Insights (Claude)</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="color:rgba(255,255,255,0.5); font-size:0.88rem; margin-bottom:0.75rem;">'
            "Optionally describe your current symptoms in plain English. Claude will combine this with your simulation data to generate personalised insights."
            "</div>",
            unsafe_allow_html=True,
        )
        symptoms_input = st.text_area(
            "Symptom description (optional)",
            placeholder="e.g. I've been having sharp lower back pain and bloating the past 3 days, feeling exhausted even after 8 hours of sleep…",
            height=90,
            label_visibility="collapsed",
        )
        if st.button("✨  Generate AI Insights", key="ai_btn"):
            with st.spinner("Consulting Claude AI…"):
                ai_response = get_claude_insights(symptoms_input, scenario, simulated_pain, pain_change, model_choice)
            st.markdown(f'<div class="ai-box">{ai_response}</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="disclaimer-box">📚 Educational suggestions are informed by NIH, Mayo Clinic, ACOG, NICE, and WHO resources. '
            "This is not medical advice. Always consult a qualified healthcare provider for personalised medical guidance.</div>",
            unsafe_allow_html=True,
        )

        # ── 7-day trend snippet ──
        df_hist = load_data()
        if not df_hist.empty and len(df_hist) >= 3:
            df_hist["date"] = pd.to_datetime(df_hist["date"], errors="coerce")
            df_hist = df_hist.dropna(subset=["date"]).sort_values("date")
            recent = df_hist.tail(7)
            if len(recent) >= 2:
                avg_recent = recent["pain_level"].mean()
                avg_prev = df_hist.tail(14).head(7)["pain_level"].mean()
                trend = avg_recent - avg_prev
                icon = "↓" if trend < 0 else "↑"
                col_t = "#34d399" if trend < 0 else "#f87171"
                st.markdown(
                    f'<div class="insight-box" style="margin-top:0.75rem;">📈 <strong>7-day tracker trend:</strong> '
                    f'Your average pain is <span style="color:{col_t};font-weight:600;">{icon} {abs(trend):.1f} pts</span> '
                    f"vs the prior 7 days (avg {avg_recent:.1f}/10).</div>",
                    unsafe_allow_html=True,
                )

    except Exception as exc:
        st.error(f"Error running simulation: {exc}")
        st.exception(exc)


# ─────────────────────────────────────────────

def tab_ai_chat() -> None:
    st.markdown(
        """
<div style="margin-bottom:1.5rem;">
    <div style="font-size:1.5rem; font-weight:700; color:rgba(255,255,255,0.9);">💬 AI Symptom Chat</div>
    <div style="color:rgba(255,255,255,0.45); font-size:0.93rem; margin-top:0.3rem;">
        Describe your endometriosis symptoms in plain English and get personalised, evidence-based insights from Claude AI.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    api_key_set = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    if not api_key_set:
        st.markdown(
            """
<div class="disclaimer-box" style="border-color:rgba(251,191,36,0.4); background:rgba(251,191,36,0.07);">
    <strong>API key not detected.</strong> Set your Anthropic API key to enable this feature:<br>
    <code>export ANTHROPIC_API_KEY=sk-ant-...</code><br>
    Then restart the app with <code>streamlit run app.py</code>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-hdr">Describe Your Symptoms</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        symptoms_free = st.text_area(
            "Describe your symptoms",
            placeholder=(
                "e.g. I've had pelvic pain for 3 days, worse in the morning. "
                "I'm also experiencing fatigue and lower back pain. "
                "I slept about 6 hours last night and my stress has been high this week due to work…"
            ),
            height=140,
            label_visibility="collapsed",
        )
    with col2:
        st.markdown('<div class="input-group-label">Context (optional)</div>', unsafe_allow_html=True)
        ctx_sleep = st.number_input("Sleep last night (hrs)", 0.0, 12.0, 7.0, 0.5)
        ctx_stress = st.slider("Stress today", 0, 10, 5)
        ctx_phase = st.selectbox("Cycle phase", list(PHASE_MAP.keys()), format_func=lambda x: PHASE_MAP[x], index=2)

    if st.button("✨  Get AI Insights", key="chat_btn", disabled=not symptoms_free.strip()):
        with st.spinner("Claude is thinking…"):
            ctx_features = dict(
                sleep=ctx_sleep, stress=ctx_stress, activity=5,
                period_phase=ctx_phase, gi=0, meds=0, mood=6, hydration=8,
            )
            response = get_claude_insights(
                symptoms_free, ctx_features,
                sim_pain=5.0, pain_change=0.0, model_name="context-only",
            )
        st.markdown('<div class="section-hdr">AI Response</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ai-box">{response}</div>', unsafe_allow_html=True)
    elif not symptoms_free.strip():
        st.markdown(
            '<div class="insight-box" style="text-align:center;">Type your symptoms above and click <strong>Get AI Insights</strong>.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="disclaimer-box">This AI-generated content is for educational purposes only and does not constitute medical advice. '
        "Always consult a qualified healthcare professional for medical decisions related to endometriosis.</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────

def tab_tracker() -> None:
    st.markdown(
        """
<div style="margin-bottom:1.5rem;">
    <div style="font-size:1.5rem; font-weight:700; color:rgba(255,255,255,0.9);">📊 My Symptom Tracker</div>
    <div style="color:rgba(255,255,255,0.45); font-size:0.93rem; margin-top:0.3rem;">
        Log your daily check-ins and visualise patterns over time.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    df = load_data()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        avg_pain = df["pain_level"].mean()
        max_pain = df["pain_level"].max()
        min_pain = df["pain_level"].min()
        total = len(df)

        m1, m2, m3, m4 = st.columns(4)
        metrics = [
            ("Average Pain", f"{avg_pain:.1f}", "/10"),
            ("Peak Pain", f"{max_pain:.0f}", "/10"),
            ("Best Day", f"{min_pain:.0f}", "/10"),
            ("Entries Logged", str(total), ""),
        ]
        for col, (label, val, unit) in zip([m1, m2, m3, m4], metrics):
            with col:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                    f'<div class="kpi-value">{val}<span style="font-size:1rem;color:rgba(255,255,255,0.4);">{unit}</span></div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(chart_pain_timeline(df), use_container_width=True)
        with col2:
            st.plotly_chart(chart_pain_distribution(df), use_container_width=True)

        if "sleep_hours" in df.columns and df["sleep_hours"].notna().sum() >= 3:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(chart_sleep_vs_pain(df), use_container_width=True)
            with col2:
                # Stress vs pain scatter
                if "stress_level" in df.columns:
                    fig_st = go.Figure(
                        go.Scatter(
                            x=df["stress_level"], y=df["pain_level"],
                            mode="markers",
                            marker=dict(
                                size=8, color=df["pain_level"],
                                colorscale=[[0, "#22d3ee"], [0.5, "#c084fc"], [1, "#f87171"]],
                                opacity=0.8, showscale=False, line=dict(width=0),
                            ),
                            hovertemplate="Stress: %{x}/10<br>Pain: %{y}/10<extra></extra>",
                        )
                    )
                    fig_st.update_layout(
                        **_DARK,
                        title=dict(text="Stress vs Pain", font=dict(size=14, color="rgba(255,255,255,0.8)")),
                        xaxis=dict(title="Stress Level (0–10)", gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(title="Pain Level (0–10)", range=[0, 10], gridcolor="rgba(255,255,255,0.06)"),
                        height=320,
                    )
                    st.plotly_chart(fig_st, use_container_width=True)

        st.markdown('<div class="section-hdr">Recent Entries</div>', unsafe_allow_html=True)
        show_cols = ["date", "pain_level", "mood", "sleep_hours", "stress_level", "period_phase"]
        show_cols = [c for c in show_cols if c in df.columns]
        disp = df[show_cols].copy()
        disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(disp.tail(12).sort_values("date", ascending=False), width="stretch", hide_index=True)
    else:
        st.markdown(
            '<div class="insight-box" style="text-align:center; padding:2rem;">No tracking data yet. Log your first entry below.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-hdr">📝 Log Today</div>', unsafe_allow_html=True)

    with st.form("checkin_form", border=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            pain_level = st.slider("Pain Level (0–10)", 0, 10, 5)
            sleep_hours = st.number_input("Sleep Hours", 0.0, 24.0, 8.0, 0.5)
            activity_level = st.slider("Activity (0–10)", 0, 10, 5)
        with c2:
            mood = st.selectbox("Mood", ["Terrible", "Poor", "Okay", "Good", "Excellent"], index=2)
            stress_level = st.slider("Stress (0–10)", 0, 10, 5)
            period_phase = st.selectbox(
                "Menstrual Phase",
                ["-2 (Ovulation)", "-1 (Pre-menstrual)", "0 (Menstrual)", "+1 (Post-menstrual)", "+2 (Follicular)"],
            )
        with c3:
            gi_symptoms = st.selectbox("GI Symptoms", ["No", "Yes"])
            nsaid_taken = st.checkbox("Took NSAID")
            other_meds = st.text_input("Other notes (optional)", placeholder="e.g. heat pad, rest day")

        save_btn = st.form_submit_button("💾  Save Entry")
        if save_btn:
            new_entry = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M"),
                "pain_level": pain_level,
                "mood": mood,
                "sleep_hours": sleep_hours,
                "stress_level": stress_level,
                "activity_level": activity_level,
                "period_phase": period_phase,
                "gi_symptoms": gi_symptoms,
                "nsaid_taken": nsaid_taken,
                "other_meds": other_meds,
            }
            existing = load_data()
            updated = pd.concat([existing, pd.DataFrame([new_entry])], ignore_index=True)
            save_data(updated)
            st.success("✅ Entry saved!")
            st.rerun()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🏠  Home", "🧪  Simulator", "💬  AI Insights", "📊  My Tracker"]
    )
    with tab1:
        tab_intro()
    with tab2:
        tab_simulator()
    with tab3:
        tab_ai_chat()
    with tab4:
        tab_tracker()


if __name__ == "__main__":
    main()
