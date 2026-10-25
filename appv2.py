
"""
FIRE HARMONIZER — PROGRESSIVE DISCLOSURE DASHBOARD
==================================================

A public-friendly Streamlit interface over the prepared M7 science table
and M8 reconstructed-event table.

Layout
------
1. Burning Activity Calendar — the challenge deliverable, first.
2. Historical Monitoring Signal — selected-year context for unusual activity and critical periods.
3. Long-term Context — All Years trend.
4. Advanced Scientific Filters — collapsed by default.
5. Investigate Specific Events — guided top-five event surface.
6. Three event-detail tabs:
      - Event Overview
      - Lifecycle Progression
      - Deep Science Context
7. M10 Static Scientific Plates — collapsed at the bottom.

Important
---------
This app never regenerates the static M10 PNGs. Generate those separately
with generate_charts.py.
"""

from __future__ import annotations

from pathlib import Path
import html
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# PATHS / CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SCIENCE_CANDIDATES = [
    BASE_DIR / "output" / "harmonized_science_all_years.csv",
    BASE_DIR / "data" / "output" / "harmonized_science_all_years.csv",
]

EVENT_CANDIDATES = [
    BASE_DIR / "data" / "output" / "critical_periods.csv",
    BASE_DIR / "output" / "critical_periods.csv",
]

CHART_DIR_CANDIDATES = [
    BASE_DIR / "output" / "charts",
    BASE_DIR / "data" / "output" / "charts",
]

M10_CHARTS = [
    (
        "01_longitudinal_evidence.png",
        "Long-term evidence",
        "Daily activity, radiative anomaly, and cross-sensor concordance across the full record.",
    ),
    (
        "02_m4_vs_m8_paradigm_shift.png",
        "M4 → M8 shift",
        "The move from isolated threshold periods to reconstructed multi-sector events.",
    ),
    (
        "03_concordance_breakdown.png",
        "Sensor concordance",
        "Activity versus radiative anomaly, with cross-sensor agreement retained as separate evidence.",
    ),
    (
        "04_mega_event_lifecycle.png",
        "Featured event lifecycle",
        "A detailed offline M10 reference plate for a major reconstructed event.",
    ),
]

GRID_SIZE = 5
EXPECTED_SECTORS = [
    f"C{i:02d}" for i in range(1, GRID_SIZE * GRID_SIZE + 1)
]
ANOMALY_THRESHOLD = 2.0
CONCORDANCE_HIGH_THRESHOLD = 0.80
CONCORDANCE_MODERATE_THRESHOLD = 0.50

# ============================================================
# CHART DESIGN TOKENS
# ============================================================

CHART_TEXT = "#0f172a"
CHART_MUTED = "#64748b"
CHART_ACCENT = "#ea580c"
CHART_AMBER = "#f59e0b"
CHART_GRID = "#e2e8f0"
CHART_PAPER = "#ffffff"
CHART_TEMPLATE = "plotly_white"
CHART_FILL = "rgba(234, 88, 12, 0.08)"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fire Harmonizer — All Years",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# LIGHT, PUBLIC-FACING STYLING
# ============================================================

st.markdown(
    """
    <style>
        :root {
            --fh-bg: #f8fafc;
            --fh-surface: #ffffff;
            --fh-surface-soft: #fff7ed;
            --fh-border: #e2e8f0;
            --fh-text: #0f172a;
            --fh-muted: #64748b;
            --fh-accent: #c2410c;
            --fh-accent-bright: #ea580c;
            --fh-accent-soft: #fff7ed;
            --fh-amber-border: #fed7aa;
            --fh-grid: #f1f5f9;
        }

        .stApp {
            background: var(--fh-bg);
            color: var(--fh-text);
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1.7rem;
            padding-bottom: 3.5rem;
        }

        h1, h2, h3, h4 {
            color: var(--fh-text);
            letter-spacing: -0.02em;
        }

        p, label, [data-testid="stMarkdownContainer"] {
            color: var(--fh-text);
        }

        .hero {
            background: linear-gradient(135deg, #ffffff 0%, #fff7ed 100%);
            border: 1px solid var(--fh-amber-border);
            border-radius: 20px;
            padding: 1.45rem 1.55rem 1.4rem 1.55rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 10px 30px rgba(234, 88, 12, 0.08);
            position: relative;
            overflow: hidden;
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 230px;
            height: 230px;
            right: -65px;
            top: -90px;
            border-radius: 50%;
            background: rgba(234, 88, 12, 0.08);
            filter: blur(28px);
            pointer-events: none;
        }

        .hero-kicker {
            color: var(--fh-accent);
            font-size: 0.77rem;
            font-weight: 800;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }

        .hero-title {
            color: var(--fh-text);
            font-size: 2.35rem;
            font-weight: 800;
            line-height: 1.05;
            margin: 0;
        }

        .hero-copy {
            color: var(--fh-muted);
            font-size: 1rem;
            line-height: 1.55;
            max-width: 900px;
            margin-top: 0.62rem;
        }

        .section-kicker {
            color: var(--fh-accent);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            margin-top: 1.35rem;
            margin-bottom: 0.28rem;
        }

        .section-description {
            color: var(--fh-muted);
            font-size: 0.91rem;
            line-height: 1.48;
            margin-bottom: 0.7rem;
        }

        .kpi-card {
            background: #ffffff;
            border: 1px solid var(--fh-border);
            border-radius: 15px;
            min-height: 105px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 5px 18px rgba(15, 23, 42, 0.04);
            transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
        }

        .kpi-card:hover {
            transform: translateY(-2px);
            border-color: #cbd5e1;
            box-shadow: 0 9px 24px rgba(15, 23, 42, 0.07);
        }

        .kpi-label {
            color: var(--fh-muted);
            font-size: 0.77rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .kpi-value {
            color: var(--fh-text);
            font-size: 1.46rem;
            font-weight: 800;
            line-height: 1.15;
        }

        .kpi-note {
            color: var(--fh-muted);
            font-size: 0.74rem;
            margin-top: 0.28rem;
        }

        .event-card {
            background: #ffffff;
            border: 1px solid var(--fh-border);
            border-radius: 15px;
            padding: 1rem 1.1rem;
            box-shadow: 0 5px 18px rgba(15, 23, 42, 0.035);
            transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
        }

        .event-card:hover {
            transform: translateY(-2px);
            border-color: #cbd5e1;
            box-shadow: 0 9px 24px rgba(15, 23, 42, 0.07);
        }

        .event-card-title {
            color: var(--fh-text);
            font-size: 1.02rem;
            font-weight: 780;
            margin-bottom: 0.16rem;
        }

        .event-card-copy {
            color: var(--fh-muted);
            font-size: 0.84rem;
            line-height: 1.45;
        }

        .context-box {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 13px;
            padding: 0.86rem 0.98rem;
            color: #9a3412;
            font-size: 0.84rem;
            line-height: 1.5;
        }

        .footer-note {
            background: #ffffff;
            border: 1px solid var(--fh-border);
            border-radius: 13px;
            padding: 0.82rem 0.98rem;
            color: var(--fh-muted);
            font-size: 0.82rem;
            line-height: 1.5;
        }

        /* Native Streamlit controls */
        div[data-testid="stExpander"] {
            border: 1px solid var(--fh-border);
            border-radius: 14px;
            background: #ffffff;
            overflow: hidden;
        }

        div[data-testid="stExpander"] summary {
            color: var(--fh-text);
        }

        div[data-testid="stSelectbox"] div[role="combobox"] {
            background: #ffffff;
            border-color: var(--fh-border);
            border-radius: 10px;
        }

        div[data-testid="stSelectbox"] div[role="combobox"]:focus-within {
            border-color: var(--fh-accent-bright);
            box-shadow: 0 0 0 2px rgba(234, 88, 12, 0.10);
        }

        div[data-testid="stSelectbox"] input,
        div[data-testid="stSlider"] label,
        div[data-testid="stExpander"] summary,
        div[data-testid="stMarkdownContainer"] p {
            color: var(--fh-text) !important;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--fh-border);
            border-radius: 12px;
            overflow: hidden;
        }

        /* Warm native tabs: active state is amber-tinted with terracotta text. */
        div[data-testid="stTabs"] [role="tablist"] {
            gap: 0.35rem;
            border-bottom: 1px solid var(--fh-border);
        }

        div[data-testid="stTabs"] button[role="tab"] {
            color: var(--fh-muted);
            border: 1px solid transparent;
            border-bottom: 2px solid transparent;
            border-radius: 10px 10px 0 0;
            padding: 0.6rem 0.9rem;
            font-weight: 700;
            background: transparent;
        }

        div[data-testid="stTabs"] button[role="tab"]:hover {
            color: var(--fh-accent);
            background: #fffbf7;
        }

        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: var(--fh-accent);
            background: #fff7ed;
            border-color: #fed7aa;
            border-bottom-color: #c2410c;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: #c2410c;
        }

        /* Slider accent */
        div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
            background: var(--fh-accent-bright);
            border-color: var(--fh-accent-bright);
        }

        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
            background: #ea580c;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ACCESSIBLE STREAMLIT WIDGET STYLING — CONTRAST OVERRIDES
# ============================================================

# These overrides intentionally follow the original theme CSS so that control
# states remain readable when Streamlit's active theme or BaseWeb defaults vary.
st.markdown(
    r"""
    <style>
        :root {
            --fh-bg: #f8fafc;
            --fh-surface: #ffffff;
            --fh-text: #0f172a;
            --fh-muted: #475569;
            --fh-border: #cbd5e1;
            --fh-accent: #c2410c;
            --fh-accent-hover: #9a3412;
            --fh-accent-soft: #fff7ed;
            --fh-accent-selected: #ffedd5;
            --fh-focus: rgba(234, 88, 12, 0.24);
        }

        /* Explicitly keep the dashboard surface light and the default text dark. */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            background: var(--fh-bg) !important;
            color: var(--fh-text) !important;
        }

        .stApp p,
        .stApp li,
        .stApp label,
        .stApp legend,
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stWidgetLabel"],
        .stApp [data-testid="stCaptionContainer"] {
            color: var(--fh-text);
        }

        .stApp [data-testid="stCaptionContainer"] p,
        .stApp small {
            color: var(--fh-muted) !important;
        }

        .stApp a {
            color: var(--fh-accent-hover) !important;
            text-decoration-thickness: 1px;
            text-underline-offset: 2px;
        }
        .stApp a:hover {
            color: #7c2d12 !important;
        }

        /* Selectbox and multiselect trigger: white surface, legible text. */
        div[data-testid="stSelectbox"] label,
        div[data-testid="stMultiSelect"] label,
        div[data-testid="stDateInput"] label,
        div[data-testid="stTimeInput"] label,
        div[data-testid="stTextInput"] label,
        div[data-testid="stNumberInput"] label,
        div[data-testid="stSlider"] label {
            color: var(--fh-text) !important;
            font-weight: 650 !important;
        }

        div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
        div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
            background-color: #ffffff !important;
            border-color: var(--fh-border) !important;
            color: var(--fh-text) !important;
            border-radius: 10px !important;
        }

        div[data-testid="stSelectbox"] [data-baseweb="select"] *,
        div[data-testid="stMultiSelect"] [data-baseweb="select"] * {
            color: var(--fh-text) !important;
        }

        /* Dropdown popovers and menu options (where the black-text issue often appears). */
        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        div[data-baseweb="popover"] > div,
        ul[role="listbox"] {
            background-color: #ffffff !important;
            color: var(--fh-text) !important;
            border-color: var(--fh-border) !important;
        }

        [role="option"] {
            background-color: #ffffff !important;
            color: var(--fh-text) !important;
        }
        [role="option"] * {
            color: inherit !important;
        }
        [role="option"]:hover,
        [role="option"][data-highlighted="true"] {
            background-color: var(--fh-accent-soft) !important;
            color: var(--fh-accent-hover) !important;
        }
        [role="option"][aria-selected="true"] {
            background-color: var(--fh-accent-selected) !important;
            color: var(--fh-accent-hover) !important;
            font-weight: 650 !important;
        }

        /* Standard action buttons remain white/dark for a calm control surface. */
        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button,
        div[data-testid="stFormSubmitButton"] button {
            background-color: #ffffff !important;
            color: var(--fh-text) !important;
            border: 1px solid var(--fh-border) !important;
            border-radius: 10px !important;
            font-weight: 650 !important;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            transition: background-color 0.15s ease, border-color 0.15s ease,
                        color 0.15s ease, transform 0.15s ease;
        }
        div[data-testid="stButton"] button *,
        div[data-testid="stDownloadButton"] button *,
        div[data-testid="stFormSubmitButton"] button * {
            color: inherit !important;
        }
        div[data-testid="stButton"] button:hover,
        div[data-testid="stDownloadButton"] button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            background-color: var(--fh-accent-soft) !important;
            color: var(--fh-accent-hover) !important;
            border-color: #fdba74 !important;
            transform: translateY(-1px);
        }

        /* Explicit primary-button variant, if the app adds one later. */
        div[data-testid="stButton"] button[kind="primary"],
        div[data-testid="stDownloadButton"] button[kind="primary"],
        div[data-testid="stFormSubmitButton"] button[kind="primary"],
        div[data-testid="stButton"] button[data-testid="stBaseButton-primary"] {
            background-color: var(--fh-accent) !important;
            color: #ffffff !important;
            border-color: var(--fh-accent) !important;
        }
        div[data-testid="stButton"] button[kind="primary"] *,
        div[data-testid="stButton"] button[data-testid="stBaseButton-primary"] * {
            color: #ffffff !important;
        }
        div[data-testid="stButton"] button[kind="primary"]:hover,
        div[data-testid="stButton"] button[data-testid="stBaseButton-primary"]:hover {
            background-color: var(--fh-accent-hover) !important;
            border-color: var(--fh-accent-hover) !important;
        }

        /* Text, number, date, time and multiline inputs. */
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stDateInput"] input,
        div[data-testid="stTimeInput"] input,
        div[data-testid="stTextArea"] textarea {
            background-color: #ffffff !important;
            color: var(--fh-text) !important;
            -webkit-text-fill-color: var(--fh-text) !important;
            border: 1px solid var(--fh-border) !important;
            border-radius: 9px !important;
        }
        div[data-testid="stTextInput"] input::placeholder,
        div[data-testid="stTextArea"] textarea::placeholder {
            color: #64748b !important;
            -webkit-text-fill-color: #64748b !important;
            opacity: 1 !important;
        }
        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stNumberInput"] input:focus,
        div[data-testid="stDateInput"] input:focus,
        div[data-testid="stTimeInput"] input:focus,
        div[data-testid="stTextArea"] textarea:focus {
            border-color: var(--fh-accent) !important;
            box-shadow: 0 0 0 3px var(--fh-focus) !important;
        }

        /* Number-input +/- buttons. */
        div[data-testid="stNumberInput"] button {
            background-color: var(--fh-accent-soft) !important;
            color: var(--fh-accent-hover) !important;
            border-color: var(--fh-border) !important;
        }
        div[data-testid="stNumberInput"] button svg {
            color: var(--fh-accent-hover) !important;
            fill: currentColor !important;
        }

        /* Checkbox, radio and toggle copy. */
        div[data-testid="stCheckbox"] label,
        div[data-testid="stRadio"] label,
        div[data-testid="stToggle"] label,
        div[data-testid="stCheckbox"] label *,
        div[data-testid="stRadio"] label *,
        div[data-testid="stToggle"] label * {
            color: var(--fh-text) !important;
        }
        div[data-testid="stCheckbox"] input,
        div[data-testid="stRadio"] input {
            accent-color: var(--fh-accent) !important;
        }

        /* Slider thumb, track, labels and values. */
        div[data-testid="stSlider"] [role="slider"] {
            background-color: var(--fh-accent) !important;
            border-color: #ffffff !important;
            box-shadow: 0 0 0 2px var(--fh-accent) !important;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
            background-color: var(--fh-accent) !important;
        }
        div[data-testid="stSlider"] [data-testid="stThumbValue"],
        div[data-testid="stSlider"] [data-testid="stMarkdownContainer"] {
            color: var(--fh-text) !important;
        }

        /* Expander headers, body and disclosure icons. */
        div[data-testid="stExpander"] {
            background-color: #ffffff !important;
            border: 1px solid var(--fh-border) !important;
            border-radius: 12px !important;
            overflow: hidden;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary * {
            color: var(--fh-text) !important;
        }
        div[data-testid="stExpander"] summary:hover {
            background-color: var(--fh-accent-soft) !important;
        }
        div[data-testid="stExpander"] summary svg {
            color: var(--fh-text) !important;
            fill: currentColor !important;
        }
        div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            background-color: #ffffff !important;
            color: var(--fh-text) !important;
        }

        /* Tabs: clearly distinguish inactive and active states. */
        div[data-testid="stTabs"] [role="tablist"] {
            gap: 0.35rem;
            border-bottom: 1px solid var(--fh-border) !important;
        }
        div[data-testid="stTabs"] button[role="tab"] {
            color: var(--fh-muted) !important;
            background-color: transparent !important;
            border: 1px solid transparent !important;
            border-bottom: 2px solid transparent !important;
            border-radius: 10px 10px 0 0;
            font-weight: 700 !important;
        }
        div[data-testid="stTabs"] button[role="tab"] * {
            color: inherit !important;
        }
        div[data-testid="stTabs"] button[role="tab"]:hover {
            background-color: var(--fh-accent-soft) !important;
            color: var(--fh-accent-hover) !important;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            background-color: var(--fh-accent-selected) !important;
            color: var(--fh-accent-hover) !important;
            border-color: #fed7aa !important;
            border-bottom-color: var(--fh-accent) !important;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: var(--fh-accent) !important;
        }

        /* Segmented controls, if used in a future version of the app. */
        div[data-testid="stSegmentedControl"] button {
            background-color: #ffffff !important;
            color: var(--fh-text) !important;
            border-color: var(--fh-border) !important;
        }
        div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
            background-color: var(--fh-accent-selected) !important;
            color: var(--fh-accent-hover) !important;
            border-color: #fdba74 !important;
        }

        /* Metrics, tables and alerts should retain high contrast. */
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: var(--fh-text) !important;
        }
        div[data-testid="stDataFrame"] {
            background-color: #ffffff !important;
            border: 1px solid var(--fh-border) !important;
            border-radius: 12px !important;
            overflow: hidden;
        }
        div[data-testid="stAlert"] {
            border-radius: 10px;
        }

        /* Strong, visible keyboard focus for interactive controls. */
        .stApp button:focus-visible,
        .stApp input:focus-visible,
        .stApp textarea:focus-visible,
        .stApp [role="combobox"]:focus-visible,
        .stApp [role="slider"]:focus-visible,
        .stApp [role="tab"]:focus-visible {
            outline: 3px solid var(--fh-focus) !important;
            outline-offset: 2px !important;
        }

        /* Disabled controls remain legible instead of becoming black-on-black. */
        .stApp button:disabled,
        .stApp input:disabled,
        .stApp textarea:disabled {
            opacity: 0.72 !important;
            cursor: not-allowed !important;
        }
    

        /* FINAL FIX: scientific-terms expander header contrast.
           Use the testid without assuming its host element is a div. */
        .stApp [data-testid="stExpander"],
        .stApp [data-testid="stExpander"] details,
        .stApp details[data-testid="stExpander"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border-color: #cbd5e1 !important;
        }

        .stApp [data-testid="stExpander"] summary,
        .stApp [data-testid="stExpander"] summary:hover,
        .stApp [data-testid="stExpander"] summary:focus,
        .stApp [data-testid="stExpander"] summary:focus-visible,
        .stApp [data-testid="stExpander"] summary:active,
        .stApp details[data-testid="stExpander"] > summary,
        .stApp details[data-testid="stExpander"] > summary:hover {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
            opacity: 1 !important;
            box-shadow: none !important;
        }

        .stApp [data-testid="stExpander"] summary *,
        .stApp [data-testid="stExpander"] summary p,
        .stApp [data-testid="stExpander"] summary span,
        .stApp [data-testid="stExpander"] summary div,
        .stApp [data-testid="stExpander"] summary label,
        .stApp details[data-testid="stExpander"] > summary * {
            background-color: transparent !important;
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
            opacity: 1 !important;
        }

        .stApp [data-testid="stExpander"] summary svg,
        .stApp [data-testid="stExpander"] summary svg * {
            color: #0f172a !important;
            fill: #0f172a !important;
            stroke: #0f172a !important;
            opacity: 1 !important;
        }

        .stApp [data-testid="stExpander"] summary:hover,
        .stApp details[data-testid="stExpander"] > summary:hover {
            background: #fff7ed !important;
            background-color: #fff7ed !important;
            cursor: pointer;
        }

        .stApp [data-testid="stExpander"] summary:focus-visible,
        .stApp details[data-testid="stExpander"] > summary:focus-visible {
            outline: 3px solid rgba(234, 88, 12, 0.30) !important;
            outline-offset: -3px !important;
        }

        .stApp [data-testid="stExpander"] [data-testid="stExpanderDetails"],
        .stApp [data-testid="stExpander"] [data-testid="stExpanderDetails"] * {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BASIC HELPERS
# ============================================================


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def fmt_sigma(value: float | int | None, decimals: int = 2) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "—"
    if not np.isfinite(numeric):
        return "—"
    return f"{numeric:.{decimals}f}σ"


def fmt_number(value: float | int | None, decimals: int = 2) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "—"
    if not np.isfinite(numeric):
        return "—"
    return f"{numeric:,.{decimals}f}"


def safe_date_text(value) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return "—"
    return parsed.strftime("%d %b %Y")


def parse_sector_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    text = str(value).strip()
    if not text:
        return []

    sectors = [item.strip().upper() for item in text.replace(",", ";").split(";")]
    return [item for item in sectors if item in EXPECTED_SECTORS]


def parse_analogue_ids(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    return [
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    ]


def concordance_label(value: float | int | None) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "Unavailable"

    if not np.isfinite(numeric):
        return "Unavailable"
    if numeric >= CONCORDANCE_HIGH_THRESHOLD:
        return "High cross-sensor agreement"
    if numeric >= CONCORDANCE_MODERATE_THRESHOLD:
        return "Moderate cross-sensor agreement"
    return "Lower cross-sensor agreement"


def weighted_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else np.nan


def render_readable_html_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    """Render a compact, high-contrast HTML table for public-facing outputs.

    This avoids embedded grid theme mismatches where the table background and
    text colors can become unreadable under a host's Streamlit theme.
    All cell content is HTML-escaped because values come from CSV/JSON files.
    """
    if frame is None or frame.empty:
        return (
            '<div style="padding:0.9rem;border:1px solid #cbd5e1;border-radius:10px;'
            'background:#ffffff;color:#475569;">No rows are available.</div>'
        )

    shown = frame.copy()
    if max_rows is not None:
        shown = shown.head(max_rows)

    table_style = (
        'width:100%;border-collapse:collapse;background:#ffffff;color:#0f172a;'
        'font-size:0.88rem;line-height:1.45;'
    )
    head_style = (
        'text-align:left;padding:0.65rem 0.75rem;background:#fff7ed;color:#9a3412;'
        'border:1px solid #fed7aa;font-weight:750;white-space:nowrap;'
    )
    cell_style = (
        'text-align:left;padding:0.58rem 0.75rem;color:#0f172a;'
        'border:1px solid #e2e8f0;vertical-align:top;'
    )
    parts = [
        '<div style="overflow-x:auto;width:100%;background:#ffffff;border:1px solid #cbd5e1;'
        'border-radius:11px;margin:0.35rem 0 0.8rem 0;">',
        f'<table style="{table_style}"><thead><tr>',
    ]
    for column in shown.columns:
        parts.append(f'<th style="{head_style}">{html.escape(str(column))}</th>')
    parts.append('</tr></thead><tbody>')
    for row_index, (_, row) in enumerate(shown.iterrows()):
        row_bg = '#ffffff' if row_index % 2 == 0 else '#f8fafc'
        parts.append('<tr>')
        for value in row.tolist():
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                display = '—'
            else:
                display = str(value)
            parts.append(
                f'<td style="{cell_style}background:{row_bg};">{html.escape(display)}</td>'
            )
        parts.append('</tr>')
    parts.append('</tbody></table></div>')
    if max_rows is not None and len(frame) > max_rows:
        parts.append(
            f'<div style="color:#475569;font-size:0.8rem;">Showing {max_rows} of {len(frame)} rows.</div>'
        )
    return ''.join(parts)


def find_chart_directory() -> Path | None:
    for directory in CHART_DIR_CANDIDATES:
        if directory.exists():
            return directory
    return None


# ============================================================
# M7 LOAD
# ============================================================

@st.cache_data(show_spinner="Loading Fire Harmonizer science data...")
def load_science() -> tuple[pd.DataFrame, str]:
    path = first_existing(SCIENCE_CANDIDATES)
    if path is None:
        raise FileNotFoundError(
            "Could not find the M7 science dataset.\n\nChecked:\n"
            + "\n".join(str(item) for item in SCIENCE_CANDIDATES)
        )

    df = pd.read_csv(path)

    required = ["acq_date", "sector", "m7_C"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "The M7 science dataset is missing required columns:\n"
            + "\n".join(missing)
        )

    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.dropna(subset=["acq_date", "sector"]).copy()
    df["sector"] = df["sector"].astype(str).str.strip().str.upper()

    if "year" not in df.columns:
        df["year"] = df["acq_date"].dt.year

    numeric_columns = [
        "m7_C",
        "m7_F",
        "m7_G",
        "m7_S",
        "modis_count",
        "viirs_count",
        "m7_modis_count_z",
        "m7_viirs_count_z",
        "u_modis_observation_count",
        "u_viirs_observation_count",
        "u_modis_mean_confidence",
        "u_viirs_mean_confidence",
        "u_modis_mean_scan",
        "u_viirs_mean_scan",
        "u_modis_mean_track",
        "u_viirs_mean_track",
        "u_modis_mean_cloud",
        "u_viirs_mean_cloud",
        "u_modis_day_fraction",
        "u_viirs_day_fraction",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "u_sensor_state" not in df.columns:
        modis_values = df["modis_count"] if "modis_count" in df.columns else pd.Series(0, index=df.index)
        viirs_values = df["viirs_count"] if "viirs_count" in df.columns else pd.Series(0, index=df.index)
        modis_detect = pd.to_numeric(modis_values, errors="coerce").fillna(0) > 0
        viirs_detect = pd.to_numeric(viirs_values, errors="coerce").fillna(0) > 0
        df["u_sensor_state"] = np.select(
            [
                modis_detect & viirs_detect,
                modis_detect & ~viirs_detect,
                viirs_detect & ~modis_detect,
            ],
            ["both_sensors", "modis_only", "viirs_only"],
            default="neither",
        )

    return (
        df.sort_values(["acq_date", "sector"]).reset_index(drop=True),
        str(path),
    )


# ============================================================
# M8 EVENT LOAD
# ============================================================

@st.cache_data(show_spinner="Loading reconstructed event archive...")
def load_events() -> tuple[pd.DataFrame, str]:
    path = first_existing(EVENT_CANDIDATES)
    if path is None:
        raise FileNotFoundError(
            "Could not find the M8 event table.\n\nChecked:\n"
            + "\n".join(str(item) for item in EVENT_CANDIDATES)
        )

    df = pd.read_csv(path)

    required = [
        "event_id",
        "emergence_date",
        "decay_date",
        "duration_days",
        "spatial_footprint",
        "peak_date",
        "peak_C",
        "peak_F",
        "mean_G",
        "mean_S",
        "expansion_flag",
        "sectors_involved",
        "top_3_analog_event_ids",
        "top_3_analog_distances",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "The M8 event table is missing required columns:\n"
            + "\n".join(missing)
        )

    for column in ["emergence_date", "decay_date", "peak_date"]:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    for column in [
        "duration_days",
        "spatial_footprint",
        "peak_C",
        "peak_F",
        "mean_G",
        "mean_S",
        "max_daily_spatial_footprint",
        "node_count",
        "duration_days_normalized",
        "peak_C_normalized",
        "peak_F_normalized",
        "spatial_footprint_normalized",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["expansion_flag"] = (
        df["expansion_flag"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    return (
        df.sort_values(["emergence_date", "decay_date", "event_id"])
        .reset_index(drop=True),
        str(path),
    )


# ============================================================
# EVENT-LEVEL RECONSTRUCTION — RETAINED FOR DEEP EXPLORATION
# ============================================================


def build_event_node_profile(
    science: pd.DataFrame,
    event: pd.Series,
) -> pd.DataFrame:
    """Reconstruct qualifying DATE × SECTOR nodes from the M7 envelope."""
    emergence = pd.Timestamp(event["emergence_date"])
    decay = pd.Timestamp(event["decay_date"])
    sectors = set(parse_sector_list(event["sectors_involved"]))

    if not sectors or pd.isna(emergence) or pd.isna(decay):
        return pd.DataFrame(columns=science.columns)

    mask = (
        (science["acq_date"] >= emergence)
        & (science["acq_date"] <= decay)
        & (science["sector"].isin(sectors))
        & (pd.to_numeric(science["m7_C"], errors="coerce") >= ANOMALY_THRESHOLD)
    )

    return (
        science.loc[mask]
        .copy()
        .sort_values(["acq_date", "sector"])
        .reset_index(drop=True)
    )


def aggregate_daily_event_profile(nodes: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reconstructed event nodes into one row per event day."""
    if nodes.empty:
        return pd.DataFrame(
            columns=[
                "acq_date",
                "C",
                "F",
                "G",
                "S",
                "spatial_footprint",
                "modis_count",
                "viirs_count",
            ]
        )

    work = nodes.copy()

    agg = {
        "C": ("m7_C", "max"),
        "F": ("m7_F", "max") if "m7_F" in work.columns else ("m7_C", "max"),
        "G": ("m7_G", "mean") if "m7_G" in work.columns else ("m7_C", "mean"),
        "S": ("m7_S", "mean") if "m7_S" in work.columns else ("m7_C", "mean"),
        "spatial_footprint": ("sector", "nunique"),
    }

    if "modis_count" in work.columns:
        agg["modis_count"] = ("modis_count", "sum")
    if "viirs_count" in work.columns:
        agg["viirs_count"] = ("viirs_count", "sum")

    daily = (
        work.groupby("acq_date", as_index=False)
        .agg(**agg)
        .sort_values("acq_date")
        .reset_index(drop=True)
    )

    for column in ["G", "S"]:
        if column in daily.columns:
            daily[column] = pd.to_numeric(daily[column], errors="coerce")

    return daily


def sector_to_position(sector: str) -> tuple[int, int]:
    number = int(str(sector).replace("C", ""))
    index = number - 1
    return index // GRID_SIZE, index % GRID_SIZE


def build_footprint_figure(nodes: pd.DataFrame) -> go.Figure:
    """Build the M8 reconstructed event footprint on the 5×5 project grid."""
    z = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype=float)
    text = [["" for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    sector_peaks = (
        nodes.groupby("sector")["m7_C"].max().to_dict()
        if not nodes.empty
        else {}
    )

    for row_index in range(GRID_SIZE):
        for col_index in range(GRID_SIZE):
            number = (GRID_SIZE - 1 - row_index) * GRID_SIZE + col_index + 1
            sector = f"C{number:02d}"
            text[row_index][col_index] = sector

            if sector in sector_peaks and np.isfinite(sector_peaks[sector]):
                z[row_index, col_index] = float(sector_peaks[sector])

    figure = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["1", "2", "3", "4", "5"],
            y=["C21–C25", "C16–C20", "C11–C15", "C06–C10", "C01–C05"],
            text=text,
            texttemplate="<b>%{text}</b>",
            textfont=dict(size=13, color=CHART_TEXT),
            hovertemplate=(
                "<b>Sector %{text}</b><br>"
                "Peak C: %{z:.2f}σ<extra></extra>"
            ),
            colorbar=dict(
                thickness=16,
                title=dict(
                    text="<b>Peak C (σ)</b>",
                    font=dict(size=13, color=CHART_TEXT),
                ),
                tickfont=dict(size=12, color=CHART_TEXT),
                outlinecolor=CHART_GRID,
                outlinewidth=1,
            ),
            hoverongaps=False,
            zsmooth=False,
            colorscale="OrRd",
            xgap=3,
            ygap=3,
        )
    )

    figure.update_layout(
        title=dict(
            text="<b>5×5 reconstructed event footprint</b>",
            font=dict(size=18, color=CHART_TEXT),
            x=0.02,
            xanchor="left",
        ),
        height=500,
        margin=dict(l=18, r=18, t=70, b=40),
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_PAPER,
        plot_bgcolor=CHART_PAPER,
        font=dict(size=13, color=CHART_TEXT),
        xaxis=dict(
            title=dict(text="<b>Grid column</b>", font=dict(size=13, color=CHART_TEXT)),
            tickfont=dict(size=12, color=CHART_TEXT),
            gridcolor=CHART_GRID,
            zerolinecolor=CHART_GRID,
            showline=True,
            linecolor=CHART_GRID,
        ),
        yaxis=dict(
            title=dict(text="<b>Grid row</b>", font=dict(size=13, color=CHART_TEXT)),
            tickfont=dict(size=11, color=CHART_TEXT),
            gridcolor=CHART_GRID,
            zerolinecolor=CHART_GRID,
            showline=True,
            linecolor=CHART_GRID,
        ),
    )
    return figure


def build_lifecycle_figure(daily: pd.DataFrame, peak_date: pd.Timestamp) -> go.Figure:
    """Build the event lifecycle with clean, aligned anomaly and concordance axes."""
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    if daily.empty:
        return figure

    c_values = pd.to_numeric(daily["C"], errors="coerce")
    f_values = pd.to_numeric(daily["F"], errors="coerce")
    g_values = pd.to_numeric(daily["G"], errors="coerce")

    anomaly_values = pd.concat([c_values, f_values], ignore_index=True).dropna()
    if anomaly_values.empty:
        anomaly_range = [-1.0, 3.0]
    else:
        low = float(anomaly_values.min())
        high = float(anomaly_values.max())
        if np.isclose(low, high):
            pad = max(1.0, abs(high) * 0.15)
        else:
            pad = max(0.5, (high - low) * 0.10)
        anomaly_range = [low - pad, high + pad]

    figure.add_trace(
        go.Scatter(
            x=daily["acq_date"],
            y=c_values,
            mode="lines+markers",
            name="C — Activity anomaly",
            line=dict(color=CHART_ACCENT, width=3.0),
            marker=dict(color=CHART_ACCENT, size=6),
            hovertemplate="%{x|%Y-%m-%d}<br><b>C:</b> %{y:.2f}σ<extra></extra>",
        ),
        secondary_y=False,
    )

    figure.add_trace(
        go.Scatter(
            x=daily["acq_date"],
            y=f_values,
            mode="lines+markers",
            name="F — FRP anomaly",
            line=dict(color=CHART_AMBER, dash="dash", width=2.7),
            marker=dict(color=CHART_AMBER, size=5),
            hovertemplate="%{x|%Y-%m-%d}<br><b>F:</b> %{y:.2f}σ<extra></extra>",
        ),
        secondary_y=False,
    )

    figure.add_trace(
        go.Scatter(
            x=daily["acq_date"],
            y=g_values,
            mode="lines+markers",
            name="G — Sensor concordance",
            line=dict(color=CHART_TEXT, width=2.7),
            marker=dict(color=CHART_TEXT, size=5),
            hovertemplate="%{x|%Y-%m-%d}<br><b>G:</b> %{y:.3f}<extra></extra>",
        ),
        secondary_y=True,
    )

    if pd.notna(peak_date):
        # Use an explicit paper-spanning shape instead of add_vline().
        # Some Plotly versions attempt arithmetic with pandas Timestamp
        # values inside add_vline(), which can fail on datetime axes.
        figure.add_shape(
            type="line",
            x0=peak_date,
            x1=peak_date,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(
                color=CHART_TEXT,
                dash="dot",
                width=2.2,
            ),
        )
        figure.add_annotation(
            x=peak_date,
            y=1.0,
            xref="x",
            yref="paper",
            text="<b>Peak date</b>",
            showarrow=False,
            yanchor="bottom",
            yshift=6,
            font=dict(size=12, color=CHART_TEXT),
            bgcolor="#ffffff",
            bordercolor=CHART_GRID,
            borderwidth=1,
            borderpad=4,
        )

    figure.add_hline(
        y=ANOMALY_THRESHOLD,
        line_color=CHART_ACCENT,
        line_dash="dot",
        line_width=1.8,
        annotation=dict(
            text="<b>C = 2σ threshold</b>",
            font=dict(size=11, color=CHART_TEXT),
            bgcolor="#fff7ed",
            bordercolor="#fed7aa",
            borderwidth=1,
            borderpad=4,
            xanchor="right",
        ),
    )

    figure.update_layout(
        title=dict(
            text="<b>Event lifecycle — daily evidence progression</b>",
            font=dict(size=18, color=CHART_TEXT),
            x=0.02,
            xanchor="left",
        ),
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_PAPER,
        plot_bgcolor=CHART_PAPER,
        font=dict(size=13, color=CHART_TEXT),
        height=540,
        hovermode="x unified",
        margin=dict(l=75, r=95, t=95, b=55),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
            font=dict(size=12, color=CHART_TEXT),
            bgcolor="#ffffff",
            bordercolor=CHART_GRID,
            borderwidth=1,
        ),
    )

    figure.update_xaxes(
        title=dict(text="<b>Date</b>", font=dict(size=13, color=CHART_TEXT)),
        tickfont=dict(size=12, color=CHART_TEXT),
        gridcolor=CHART_GRID,
        zerolinecolor=CHART_GRID,
        showline=True,
        linecolor=CHART_GRID,
    )
    figure.update_yaxes(
        title=dict(text="<b>Anomaly (σ)</b>", font=dict(size=13, color=CHART_TEXT)),
        tickfont=dict(size=12, color=CHART_TEXT),
        range=anomaly_range,
        gridcolor=CHART_GRID,
        zerolinecolor=CHART_GRID,
        showline=True,
        linecolor=CHART_GRID,
        automargin=True,
        secondary_y=False,
    )
    figure.update_yaxes(
        title=dict(text="<b>G — Concordance</b>", font=dict(size=13, color=CHART_TEXT)),
        tickfont=dict(size=12, color=CHART_TEXT),
        range=[0, 1.05],
        dtick=0.2,
        gridcolor="#ffffff",
        zeroline=False,
        showline=True,
        linecolor=CHART_TEXT,
        automargin=True,
        secondary_y=True,
    )

    return figure


# ============================================================
# DEEP-SCIENCE TABLES
# ============================================================


def build_evidence_table(event: pd.Series, nodes: pd.DataFrame) -> pd.DataFrame:
    """Build the transparent C/F/G/S/U evidence table."""
    g_value = pd.to_numeric(event.get("mean_G"), errors="coerce")
    s_value = pd.to_numeric(event.get("mean_S"), errors="coerce")

    context_parts: list[str] = []

    if "u_sensor_state" in nodes.columns and not nodes.empty:
        state_counts = nodes["u_sensor_state"].astype(str).value_counts()
        if not state_counts.empty:
            context_parts.append(f"Dominant sensor state: {state_counts.index[0]}")

    modis_obs = weighted_mean(nodes.get("u_modis_observation_count", pd.Series(dtype=float)))
    viirs_obs = weighted_mean(nodes.get("u_viirs_observation_count", pd.Series(dtype=float)))
    if np.isfinite(modis_obs):
        context_parts.append(f"MODIS retained context/count field mean: {modis_obs:.1f}")
    if np.isfinite(viirs_obs):
        context_parts.append(f"VIIRS retained context/count field mean: {viirs_obs:.1f}")

    modis_conf = weighted_mean(nodes.get("u_modis_mean_confidence", pd.Series(dtype=float)))
    viirs_conf = weighted_mean(nodes.get("u_viirs_mean_confidence", pd.Series(dtype=float)))
    if np.isfinite(modis_conf):
        context_parts.append(f"MODIS mean confidence: {modis_conf:.1f}")
    if np.isfinite(viirs_conf):
        context_parts.append(f"VIIRS mean confidence: {viirs_conf:.1f}")

    u_text = (
        "; ".join(context_parts)
        if context_parts
        else "Context fields unavailable in this science export."
    )

    return pd.DataFrame(
        [
            {
                "Evidence dimension": "C — Activity",
                "Observed / event value": event["peak_C"],
                "Unit": "σ",
                "Definition": "Peak score showing how unusual the hotspot-detection count is versus its historical reference.",
                "Interpretation": "Statistical unusualness, not a number of fires or direct danger rating.",
            },
            {
                "Evidence dimension": "F — Radiative",
                "Observed / event value": event["peak_F"],
                "Unit": "σ",
                "Definition": "Peak score showing how unusual summed sector-day Fire Radiative Power (FRP) is versus its historical reference.",
                "Interpretation": "FRP is power, not total fire energy; F can rise with detection count.",
            },
            {
                "Evidence dimension": "G — Sensor concordance",
                "Observed / event value": g_value,
                "Unit": "0–1",
                "Definition": "Bounded agreement between standardized MODIS and VIIRS count signals; G reuses count evidence also used by C.",
                "Interpretation": concordance_label(g_value) + "; not ground truth or an independent test of fire presence",
            },
            {
                "Evidence dimension": "S — Spatial coherence",
                "Observed / event value": s_value,
                "Unit": "0–1",
                "Definition": "Share of available neighboring sectors also crossing the activity-anomaly threshold; derived from C, not independent confirmation.",
                "Interpretation": "Shows neighboring-sector participation within the coarse grid; it is related to C, not independent confirmation.",
            },
            {
                "Evidence dimension": "U — Observation context",
                "Observed / event value": u_text,
                "Unit": "Context",
                "Definition": "Retained context descriptors such as sensor state, associated counts, confidence, geometry, and acquisition regime where available.",
                "Interpretation": "Context for reading the record; not a probability, overpass-opportunity measure, or universal confidence percentage.",
            },
        ]
    )


def build_analogue_table(target: pd.Series, events: pd.DataFrame) -> pd.DataFrame:
    """Build the deterministic target-versus-nearest-analogue table."""
    analogue_ids = parse_analogue_ids(target["top_3_analog_event_ids"])
    selected_ids = [target["event_id"], *analogue_ids[:3]]

    subset = events[events["event_id"].isin(selected_ids)].copy()
    if subset.empty:
        return pd.DataFrame()

    order_map = {event_id: index for index, event_id in enumerate(selected_ids)}
    subset["_order"] = subset["event_id"].map(order_map)
    subset = subset.sort_values("_order").drop(columns=["_order"])

    label_map = {target["event_id"]: "Target"}
    for index, event_id in enumerate(analogue_ids[:3], start=1):
        label_map[event_id] = f"Analogue {index}"

    output = subset[
        [
            "event_id",
            "duration_days",
            "spatial_footprint",
            "peak_C",
            "peak_F",
            "mean_G",
            "mean_S",
            "expansion_flag",
        ]
    ].copy()

    output.insert(0, "Relation", output["event_id"].map(label_map))
    return output.rename(
        columns={
            "event_id": "Event ID",
            "duration_days": "Duration (days)",
            "spatial_footprint": "Spatial footprint (sectors)",
            "peak_C": "Peak C (σ)",
            "peak_F": "Peak F (σ)",
            "mean_G": "Mean G",
            "mean_S": "Mean S",
            "expansion_flag": "Expanded?",
        }
    )


# ============================================================
# BURNING ACTIVITY CALENDAR
# ============================================================

CALENDAR_METRICS = {
    "Activity anomaly — C": {
        "column": "m7_C",
        "label": "C — Activity anomaly",
        "unit": "σ",
        "colorscale": "YlOrRd",
        "description": "How unusual the sector-day detection count is against its historical reference; a statistical signal, not a fire count or severity score.",
    },
    "FRP anomaly — F": {
        "column": "m7_F",
        "label": "F — FRP anomaly",
        "unit": "σ",
        "colorscale": "YlOrRd",
        "description": "How unusual summed Fire Radiative Power is against its historical reference; FRP is power, not total fire energy.",
    },
    "Sensor concordance — G": {
        "column": "m7_G",
        "label": "G — Cross-sensor concordance",
        "unit": "0–1",
        "colorscale": "Blues",
        "description": "Agreement between standardized MODIS and VIIRS count signals; related to C and not ground-truth confirmation.",
    },
    "Spatial coherence — S": {
        "column": "m7_S",
        "label": "S — Spatial coherence",
        "unit": "0–1",
        "colorscale": "Oranges",
        "description": "Share of neighboring sectors crossing the C threshold; derived from C, not independent confirmation.",
    },
}


def build_burning_activity_calendar(
    science: pd.DataFrame,
    selected_year: int,
    metric_name: str,
) -> go.Figure:
    """Build the challenge-facing year × sector burning-activity calendar."""
    spec = CALENDAR_METRICS[metric_name]
    column = spec["column"]

    work = science[
        science["acq_date"].dt.year.eq(selected_year)
    ][["acq_date", "sector", column]].copy()

    if work.empty:
        figure = go.Figure()
        figure.update_layout(
            title=dict(
                text=f"<b>Burning Activity Calendar — {selected_year}</b>",
                font=dict(size=19, color=CHART_TEXT),
                x=0.02,
                xanchor="left",
            ),
            template=CHART_TEMPLATE,
            height=530,
            paper_bgcolor=CHART_PAPER,
            plot_bgcolor=CHART_PAPER,
        )
        return figure

    work["date"] = pd.to_datetime(work["acq_date"], errors="coerce")
    work["value"] = pd.to_numeric(work[column], errors="coerce")

    # The science matrix should normally contain one row per date × sector.
    # If duplicates exist, mean them rather than silently taking one record.
    pivot = (
        work.dropna(subset=["date"])
        .groupby(["sector", "date"], as_index=False)["value"]
        .mean()
        .pivot(index="sector", columns="date", values="value")
    )

    ordered_sectors = [s for s in EXPECTED_SECTORS if s in pivot.index]
    pivot = pivot.reindex(ordered_sectors)

    dates = list(pivot.columns)
    z = pivot.to_numpy(dtype=float)

    numeric = pd.to_numeric(work["value"], errors="coerce").dropna()
    if numeric.empty:
        zmin, zmax = 0.0, 1.0
    elif metric_name == "Activity anomaly — C":
        zmin = 0.0
        upper = float(numeric.quantile(0.99))
        zmax = max(ANOMALY_THRESHOLD, upper, 1.0)
    elif metric_name == "FRP anomaly — F":
        low = float(numeric.quantile(0.01))
        high = float(numeric.quantile(0.99))
        span = max(0.5, high - low)
        zmin, zmax = low - 0.05 * span, high + 0.05 * span
    else:
        zmin, zmax = 0.0, 1.0

    hover_name = spec["label"]
    unit = spec["unit"]

    figure = go.Figure(
        data=go.Heatmap(
            x=dates,
            y=ordered_sectors,
            z=z,
            colorscale=spec["colorscale"],
            zmin=zmin,
            zmax=zmax,
            connectgaps=False,
            hoverongaps=False,
            xgap=1,
            ygap=2,
            colorbar=dict(
                title=dict(
                    text=f"<b>{hover_name} ({unit})</b>",
                    font=dict(size=12, color=CHART_TEXT),
                ),
                tickfont=dict(size=11, color=CHART_TEXT),
                outlinecolor=CHART_GRID,
                outlinewidth=1,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{x|%d %b %Y}<br>"
                f"<b>{hover_name}:</b> %{{z:.3f}} {unit}<extra></extra>"
            ),
        )
    )

    month_starts = pd.date_range(
        f"{selected_year}-01-01",
        f"{selected_year}-12-01",
        freq="MS",
    )

    for month in month_starts:
        if dates:
            figure.add_shape(
                type="line",
                x0=month,
                x1=month,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(
                    color=CHART_GRID,
                    width=1,
                    dash="dot",
                ),
            )

    figure.update_layout(
        title=dict(
            text=f"<b>Burning Activity Calendar — {selected_year}</b>",
            font=dict(size=19, color=CHART_TEXT),
            x=0.02,
            xanchor="left",
        ),
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_PAPER,
        plot_bgcolor=CHART_PAPER,
        font=dict(size=13, color=CHART_TEXT),
        height=610,
        margin=dict(l=75, r=100, t=95, b=62),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11, color=CHART_TEXT),
            bgcolor="#ffffff",
            bordercolor=CHART_GRID,
            borderwidth=1,
        ),
        xaxis=dict(
            title=dict(
                text="<b>Observation date</b>",
                font=dict(size=13, color=CHART_TEXT),
            ),
            tickformat="%b",
            dtick="M1",
            tickfont=dict(size=11, color=CHART_TEXT),
            showgrid=False,
            showline=True,
            linecolor=CHART_GRID,
            rangeslider=dict(visible=True, thickness=0.06),
        ),
        yaxis=dict(
            title=dict(
                text="<b>Sector</b>",
                font=dict(size=13, color=CHART_TEXT),
            ),
            tickfont=dict(size=11, color=CHART_TEXT),
            categoryorder="array",
            categoryarray=ordered_sectors,
            showgrid=False,
            showline=True,
            linecolor=CHART_GRID,
        ),
    )

    return figure


def annual_monitoring_summary(
    science: pd.DataFrame,
    events: pd.DataFrame,
    selected_year: int,
) -> dict[str, object]:
    """Compute concise selected-year indicators for the monitoring callout."""
    year_science = science[
        science["acq_date"].dt.year.eq(selected_year)
    ].copy()

    c_values = pd.to_numeric(year_science["m7_C"], errors="coerce")
    active = c_values >= ANOMALY_THRESHOLD

    year_events = filter_events_for_year_and_controls(
        events,
        selected_year,
        min_duration=1,
        min_footprint=1,
        min_peak_c=ANOMALY_THRESHOLD,
    )

    if year_science.empty or c_values.dropna().empty:
        return {
            "active_sector_days": 0,
            "peak_c": np.nan,
            "peak_date": pd.NaT,
            "event_count": len(year_events),
        }

    peak_index = c_values.idxmax()
    return {
        "active_sector_days": int(active.sum()),
        "peak_c": float(c_values.max()),
        "peak_date": year_science.loc[peak_index, "acq_date"],
        "event_count": len(year_events),
    }


# ============================================================
# PUBLIC MACRO DASHBOARD
# ============================================================


def build_macro_event_trend(events: pd.DataFrame) -> pd.DataFrame:
    """Count reconstructed M8 events by emergence month."""
    if events.empty:
        return pd.DataFrame(columns=["month", "events"])

    work = events[["emergence_date"]].dropna().copy()
    work["month"] = work["emergence_date"].dt.to_period("M").dt.to_timestamp()

    return (
        work.groupby("month", as_index=False)
        .size()
        .rename(columns={"size": "events"})
        .sort_values("month")
        .reset_index(drop=True)
    )


def build_macro_figure(trend: pd.DataFrame) -> go.Figure:
    """Build the public macro trend with explicit typography and clean scaling."""
    figure = go.Figure()
    if trend.empty:
        return figure

    event_values = pd.to_numeric(trend["events"], errors="coerce").fillna(0)
    max_events = float(event_values.max()) if not event_values.empty else 0.0
    y_upper = max(1.0, max_events * 1.12)

    figure.add_trace(
        go.Scatter(
            x=trend["month"],
            y=event_values,
            mode="lines",
            name="Events emerging",
            line=dict(color=CHART_ACCENT, width=3.0),
            fill="tozeroy",
            fillcolor=CHART_FILL,
            hovertemplate=(
                "%{x|%b %Y}<br>"
                "<b>Reconstructed events:</b> %{y}<extra></extra>"
            ),
        )
    )

    rolling = event_values.rolling(12, min_periods=1).mean()
    figure.add_trace(
        go.Scatter(
            x=trend["month"],
            y=rolling,
            mode="lines",
            name="12-month rolling average",
            line=dict(color=CHART_TEXT, width=3.0),
            hovertemplate=(
                "%{x|%b %Y}<br>"
                "<b>12-month average:</b> %{y:.1f}<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title=dict(
            text="<b>All Years — reconstructed fire-event trend</b>",
            font=dict(size=19, color=CHART_TEXT),
            x=0.02,
            xanchor="left",
        ),
        template=CHART_TEMPLATE,
        height=470,
        paper_bgcolor=CHART_PAPER,
        plot_bgcolor=CHART_PAPER,
        font=dict(size=13, color=CHART_TEXT),
        margin=dict(l=75, r=35, t=90, b=60),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=12, color=CHART_TEXT),
            bgcolor="#ffffff",
            bordercolor=CHART_GRID,
            borderwidth=1,
        ),
        xaxis=dict(
            title=dict(
                text="<b>Emergence month</b>",
                font=dict(size=13, color=CHART_TEXT),
            ),
            tickfont=dict(size=12, color=CHART_TEXT),
            showgrid=True,
            gridcolor=CHART_GRID,
            zerolinecolor=CHART_GRID,
            showline=True,
            linecolor=CHART_GRID,
        ),
        yaxis=dict(
            title=dict(
                text="<b>Reconstructed events</b>",
                font=dict(size=13, color=CHART_TEXT),
            ),
            tickfont=dict(size=12, color=CHART_TEXT),
            range=[0, y_upper],
            rangemode="tozero",
            showgrid=True,
            gridcolor=CHART_GRID,
            zerolinecolor=CHART_GRID,
            showline=True,
            linecolor=CHART_GRID,
        ),
    )
    return figure


def filter_events_for_year_and_controls(
    events: pd.DataFrame,
    selected_year: int,
    min_duration: int,
    min_footprint: int,
    min_peak_c: float,
) -> pd.DataFrame:
    """Filter events with overlap-based year semantics and advanced controls."""
    year_start = pd.Timestamp(f"{selected_year}-01-01")
    year_end = pd.Timestamp(f"{selected_year}-12-31")

    result = events[
        (events["emergence_date"] <= year_end)
        & (events["decay_date"] >= year_start)
    ].copy()

    result = result[
        result["duration_days"] >= min_duration
    ]
    result = result[
        result["spatial_footprint"] >= min_footprint
    ]
    result = result[
        result["peak_C"] >= min_peak_c
    ]

    return result.reset_index(drop=True)


def rank_largest_events(events: pd.DataFrame) -> pd.DataFrame:
    """Rank event candidates by footprint first, then duration and activity."""
    return (
        events.sort_values(
            [
                "spatial_footprint",
                "duration_days",
                "peak_C",
                "peak_F",
                "event_id",
            ],
            ascending=[False, False, False, False, True],
        )
        .reset_index(drop=True)
    )


def format_event_label(event: pd.Series) -> str:
    return (
        f"{event['event_id']} | "
        f"{safe_date_text(event['emergence_date'])} → "
        f"{safe_date_text(event['decay_date'])} | "
        f"{int(event['spatial_footprint'])} sectors"
    )


# ============================================================
# LOAD DATA
# ============================================================

try:
    science, science_source = load_science()
except Exception as error:
    st.error(str(error))
    st.stop()

try:
    events, events_source = load_events()
except Exception as error:
    st.error(str(error))
    st.stop()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">NASA FIRMS · Broad study window · Burning Activity Calendar</div>
        <div class="hero-title">Fire Harmonizer</div>
        <div class="hero-copy">
            Explore NASA FIRMS hotspot detections from MODIS and Suomi NPP VIIRS in one daily calendar. Compare historical patterns across a broad study window, then inspect the evidence behind unusual periods and reconstructed event groupings.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">A plain-language introduction</div>', unsafe_allow_html=True)
st.markdown(
    """
    NASA's Fire Information for Resource Management System (FIRMS) provides satellite active-fire hotspot detections. Fire Harmonizer brings together two sensor data streams used in the project: MODIS and the Visible Infrared Imaging Radiometer Suite (VIIRS) aboard Suomi NPP. A hotspot is a satellite detection associated with a thermal signal; it is not necessarily one unique ground fire, an exact fire boundary, or a measurement of burned area. Since MODIS and VIIRS observe with different instruments and sampling patterns, their raw detection counts are kept distinguishable and compared in historical context.

    The project organizes the archive into a daily 5×5 grid over a broad geographic study window of about **14.1 million km²** (approximately 82°W–44°W and 20°S–10°N). This wide view can help the public explore historical patterns, give educators concrete examples for learning about satellites and data uncertainty, and help researchers or analysts decide which periods merit closer examination. The wide coverage is a strength for regional comparisons, but the 25 grid sectors are very large and cannot locate individual fires precisely. The app is for historical exploration and research support—not a live emergency-alert service—and an empty or missing cell does not prove that no burning occurred.
    """
)

with st.expander("What the scientific terms mean", expanded=True):
    st.markdown(
        """
        **Hotspot detection (NASA FIRMS):** a satellite-reported location with a thermal signal consistent with active burning under the product's detection rules. It is evidence to investigate, not a guaranteed unique fire or a fire perimeter.

        **MODIS and VIIRS:** the two satellite-instrument data streams being compared. They differ in sensor design and sampling, so one sensor's raw count should not simply be treated as equal to the other's. In this project, VIIRS is from Suomi NPP.

        **C — Activity anomaly:** how unusual the sector-day's hotspot-detection count is compared with its historical reference. The “σ” symbol means standard deviations on a standardized scale. C is a measure of statistical unusualness, not the number of fires, a burned-area estimate, or a direct measure of danger. Very sparse or near-zero-variance reference data can produce extreme scores, so inspect the underlying context.

        **F — FRP anomaly:** how unusual the summed Fire Radiative Power (FRP) is for a sector-day compared with its reference. FRP describes the rate of radiative energy release (power), not the total energy released over an entire fire. Summed FRP can rise simply because more detections were recorded, so F is related to—but not identical to—C.

        **G — Cross-sensor concordance:** how closely the standardized MODIS and VIIRS count signals align. Higher G indicates closer agreement in those signals; it is not a probability that a fire is real or ground-truth confirmation. G reuses count-score information that also contributes to C, so it is not an independent test.

        **S — Spatial coherence / adjacent-sector support:** how much neighboring grid sectors also cross the activity-anomaly threshold. It helps show whether unusual values are spatially clustered, but it is derived from C and is not independent confirmation. “Spatial” here refers to the project's coarse grid, not precise fire coordinates.

        **U — Observation context:** retained descriptors such as sensor state, associated counts, confidence, viewing geometry, or daytime context where available. These can help interpret the data's limitations, but U is not a universal confidence percentage and the sensor-state flag does not establish whether a satellite had an overpass opportunity. No recorded detection does not prove no burning.

        **M8 reconstructed event:** a statistical grouping of qualifying sector-days connected through the project's time-and-neighbor rules. It is an analytical event structure, not a verified individual fire, burn scar, fire perimeter, or official incident declaration.

        **Historical monitoring signal:** a way to prioritize questions about past records. The baseline can include later dates than the event being viewed, so this archive should not be presented as a real-time or forecast system.
        """
    )


with st.expander("5×5 sector geography reference", expanded=False):
    st.markdown(
        """
        Use this reference to interpret the project's C01–C25 grid labels. Bounds and centres describe the rectangular analysis grid, while the geographic descriptions are approximate orientation aids—not official administrative boundaries or a precise Amazon biome mask. Several western cells include Pacific Ocean and Andean areas; eastern and southern cells include areas outside the Amazon rainforest, including parts of the Cerrado and Pantanal.
        """
    )
    sector_geography = pd.DataFrame([
        {"Sector": "C01", "Latitude bounds": "4°N–10°N", "Longitude bounds": "82.0°W–74.4°W", "Approximate centre (lat, lon)": "7°N, 78.2°W", "Approximate geographic region": "Pacific-facing western Colombia and adjacent Pacific Ocean"},
        {"Sector": "C02", "Latitude bounds": "4°N–10°N", "Longitude bounds": "74.4°W–66.8°W", "Approximate centre (lat, lon)": "7°N, 70.6°W", "Approximate geographic region": "Northern Colombia and western Venezuela"},
        {"Sector": "C03", "Latitude bounds": "4°N–10°N", "Longitude bounds": "66.8°W–59.2°W", "Approximate centre (lat, lon)": "7°N, 63.0°W", "Approximate geographic region": "Eastern Venezuela and western Guiana Shield"},
        {"Sector": "C04", "Latitude bounds": "4°N–10°N", "Longitude bounds": "59.2°W–51.6°W", "Approximate centre (lat, lon)": "7°N, 55.4°W", "Approximate geographic region": "Guyana, Suriname and French Guiana"},
        {"Sector": "C05", "Latitude bounds": "4°N–10°N", "Longitude bounds": "51.6°W–44.0°W", "Approximate centre (lat, lon)": "7°N, 47.8°W", "Approximate geographic region": "Amapá and far northern Brazil, adjoining Atlantic coast"},
        {"Sector": "C06", "Latitude bounds": "2°S–4°N", "Longitude bounds": "82.0°W–74.4°W", "Approximate centre (lat, lon)": "1°N, 78.2°W", "Approximate geographic region": "Southern Colombia, Ecuador and western Andean foothills"},
        {"Sector": "C07", "Latitude bounds": "2°S–4°N", "Longitude bounds": "74.4°W–66.8°W", "Approximate centre (lat, lon)": "1°N, 70.6°W", "Approximate geographic region": "Central and southern Colombia and adjoining Amazon borderlands"},
        {"Sector": "C08", "Latitude bounds": "2°S–4°N", "Longitude bounds": "66.8°W–59.2°W", "Approximate centre (lat, lon)": "1°N, 63.0°W", "Approximate geographic region": "Eastern Colombia, southern Venezuela and northwestern Amazonia"},
        {"Sector": "C09", "Latitude bounds": "2°S–4°N", "Longitude bounds": "59.2°W–51.6°W", "Approximate centre (lat, lon)": "1°N, 55.4°W", "Approximate geographic region": "Guianas and northern Brazil"},
        {"Sector": "C10", "Latitude bounds": "2°S–4°N", "Longitude bounds": "51.6°W–44.0°W", "Approximate centre (lat, lon)": "1°N, 47.8°W", "Approximate geographic region": "Amapá, northern Pará and the lower Amazon region"},
        {"Sector": "C11", "Latitude bounds": "8°S–2°S", "Longitude bounds": "82.0°W–74.4°W", "Approximate centre (lat, lon)": "5°S, 78.2°W", "Approximate geographic region": "Pacific-facing Peru, western Andes and adjacent Pacific Ocean"},
        {"Sector": "C12", "Latitude bounds": "8°S–2°S", "Longitude bounds": "74.4°W–66.8°W", "Approximate centre (lat, lon)": "5°S, 70.6°W", "Approximate geographic region": "Peruvian Amazon and western Amazon borderlands of Bolivia"},
        {"Sector": "C13", "Latitude bounds": "8°S–2°S", "Longitude bounds": "66.8°W–59.2°W", "Approximate centre (lat, lon)": "5°S, 63.0°W", "Approximate geographic region": "Southwestern Amazonia, including northern Bolivia and western Brazil"},
        {"Sector": "C14", "Latitude bounds": "8°S–2°S", "Longitude bounds": "59.2°W–51.6°W", "Approximate centre (lat, lon)": "5°S, 55.4°W", "Approximate geographic region": "Central and southern Amazon region, west-central Brazil"},
        {"Sector": "C15", "Latitude bounds": "8°S–2°S", "Longitude bounds": "51.6°W–44.0°W", "Approximate centre (lat, lon)": "5°S, 47.8°W", "Approximate geographic region": "Eastern Amazonia and the Pará–Tocantins transition"},
        {"Sector": "C16", "Latitude bounds": "14°S–8°S", "Longitude bounds": "82.0°W–74.4°W", "Approximate centre (lat, lon)": "11°S, 78.2°W", "Approximate geographic region": "Southern Peru's Pacific coast, western Andes and adjacent Pacific Ocean"},
        {"Sector": "C17", "Latitude bounds": "14°S–8°S", "Longitude bounds": "74.4°W–66.8°W", "Approximate centre (lat, lon)": "11°S, 70.6°W", "Approximate geographic region": "Southern Peru and western Bolivia"},
        {"Sector": "C18", "Latitude bounds": "14°S–8°S", "Longitude bounds": "66.8°W–59.2°W", "Approximate centre (lat, lon)": "11°S, 63.0°W", "Approximate geographic region": "Bolivian lowlands and southwestern Brazilian Amazonia"},
        {"Sector": "C19", "Latitude bounds": "14°S–8°S", "Longitude bounds": "59.2°W–51.6°W", "Approximate centre (lat, lon)": "11°S, 55.4°W", "Approximate geographic region": "West-central Brazil and the southern Amazon–Cerrado transition"},
        {"Sector": "C20", "Latitude bounds": "14°S–8°S", "Longitude bounds": "51.6°W–44.0°W", "Approximate centre (lat, lon)": "11°S, 47.8°W", "Approximate geographic region": "Central Brazil, including parts of the Cerrado and Tocantins–Goiás region"},
        {"Sector": "C21", "Latitude bounds": "20°S–14°S", "Longitude bounds": "82.0°W–74.4°W", "Approximate centre (lat, lon)": "17°S, 78.2°W", "Approximate geographic region": "Southern Peru, western Andes and adjacent Pacific Ocean"},
        {"Sector": "C22", "Latitude bounds": "20°S–14°S", "Longitude bounds": "74.4°W–66.8°W", "Approximate centre (lat, lon)": "17°S, 70.6°W", "Approximate geographic region": "Southern Peru and western to central Bolivia"},
        {"Sector": "C23", "Latitude bounds": "20°S–14°S", "Longitude bounds": "66.8°W–59.2°W", "Approximate centre (lat, lon)": "17°S, 63.0°W", "Approximate geographic region": "Eastern Bolivia and southwestern Brazil"},
        {"Sector": "C24", "Latitude bounds": "20°S–14°S", "Longitude bounds": "59.2°W–51.6°W", "Approximate centre (lat, lon)": "17°S, 55.4°W", "Approximate geographic region": "South-central Brazil, including the Pantanal and Cerrado transition"},
        {"Sector": "C25", "Latitude bounds": "20°S–14°S", "Longitude bounds": "51.6°W–44.0°W", "Approximate centre (lat, lon)": "17°S, 47.8°W", "Approximate geographic region": "Central-eastern Brazil, including the Cerrado and Brazilian Highlands"},
    ])
    # Use a plain HTML table here rather than Streamlit's dataframe grid.
    # This avoids a blank/unreadable grid when the host theme or injected CSS
    # conflicts with the interactive dataframe renderer.
    st.markdown(
        render_readable_html_table(sector_geography),
        unsafe_allow_html=True,
    )
    st.caption("Grid sectors are rectangular analytical bins. Their geographic labels are approximate; the grid is not a precise fire-location layer or an official biome boundary.")


# ============================================================
# STEP 1 — BURNING ACTIVITY CALENDAR
# ============================================================

# Determine years from the science record so the calendar always reflects
# the actual stored coverage rather than a hard-coded current-year list.
years = sorted(
    science["acq_date"].dt.year.dropna().astype(int).unique().tolist()
)
years = [year for year in years if 2012 <= year <= 2026] or [2026]
default_year = 2026 if 2026 in years else years[-1]

st.markdown('<div class="section-kicker">Burning Activity Calendar</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">This is the challenge-facing view: a harmonized day-by-sector calendar showing when and where satellite-observed burning activity occurs. Blank cells mean no stored observation record for that sector-day; they are not proof of zero burning.</div>',
    unsafe_allow_html=True,
)

calendar_controls = st.columns([0.9, 1.35, 1.75])

with calendar_controls[0]:
    selected_year = st.selectbox(
        "Calendar year",
        years,
        index=years.index(default_year),
        help="Choose the historical year to inspect. 2026 is a partial year in the current archive.",
    )

with calendar_controls[1]:
    calendar_metric = st.selectbox(
        "Calendar metric",
        list(CALENDAR_METRICS.keys()),
        index=0,
        help="C is the primary activity-anomaly view. F, G, and S show related evidence dimensions; G and S are not independent confirmation.",
    )

with calendar_controls[2]:
    st.markdown(
        '<div class="context-box"><strong>Historical monitoring role</strong><br>Use the calendar to focus attention on unusual activity and historically persistent periods. The application is for historical scientific review and research support, not a certified live emergency-alert service.</div>',
        unsafe_allow_html=True,
    )

calendar_figure = build_burning_activity_calendar(
    science,
    selected_year,
    calendar_metric,
)

st.plotly_chart(
    calendar_figure,
    use_container_width=True,
    config={"displayModeBar": False, "responsive": True},
)

summary = annual_monitoring_summary(
    science,
    events,
    selected_year,
)

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">C-active sector-days</div><div class="kpi-value">{summary["active_sector_days"]:,}</div><div class="kpi-note">C ≥ {ANOMALY_THRESHOLD:.0f} anomaly evidence</div></div>',
        unsafe_allow_html=True,
    )

with s2:
    peak_c = summary["peak_c"]
    peak_c_text = f"{peak_c:.2f}σ" if np.isfinite(peak_c) else "—"
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Peak activity anomaly</div><div class="kpi-value">{peak_c_text}</div><div class="kpi-note">Maximum C in {selected_year}</div></div>',
        unsafe_allow_html=True,
    )

with s3:
    peak_date = summary["peak_date"]
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Peak date</div><div class="kpi-value">{safe_date_text(peak_date)}</div><div class="kpi-note">Highest C sector-day</div></div>',
        unsafe_allow_html=True,
    )

with s4:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Reconstructed events</div><div class="kpi-value">{int(summary["event_count"]):,}</div><div class="kpi-note">M8 events overlapping {selected_year}</div></div>',
        unsafe_allow_html=True,
    )

st.caption(
    CALENDAR_METRICS[calendar_metric]["description"]
    + " Missing values are left missing where the source table has no value; a missing cell is not proof that no burning occurred."
)


# ============================================================
# STEP 2 — LONG-TERM CONTEXT
# ============================================================

st.markdown('<div class="section-kicker">Long-term Context</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">After the selected-year calendar, the full archive provides the long-term context needed to distinguish unusual periods from ordinary recurrence.</div>',
    unsafe_allow_html=True,
)

start_date = science["acq_date"].min()
end_date = science["acq_date"].max()
sector_count = int(science["sector"].nunique())
total_event_count = int(len(events))

trend = build_macro_event_trend(events)
peak_month_count = int(trend["events"].max()) if not trend.empty else 0

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Reconstructed events</div><div class="kpi-value">{total_event_count:,}</div><div class="kpi-note">M8 event archive</div></div>',
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Sectors represented</div><div class="kpi-value">{sector_count:,}</div><div class="kpi-note">5×5 study grid sectors present</div></div>',
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Record covered</div><div class="kpi-value">{safe_date_text(start_date)}</div><div class="kpi-note">through {safe_date_text(end_date)}</div></div>',
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Busiest emergence month</div><div class="kpi-value">{peak_month_count:,}</div><div class="kpi-note">Reconstructed events in one month</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-kicker">All Years event trend</div>', unsafe_allow_html=True)
macro_figure = build_macro_figure(trend)
st.plotly_chart(
    macro_figure,
    use_container_width=True,
    config={"displayModeBar": False, "responsive": True},
)


# ============================================================
# STEP 3 — ADVANCED SCIENTIFIC FILTERS
# ============================================================

with st.expander("Advanced Scientific Filters", expanded=False):
    st.caption(
        "These controls narrow the event archive for investigation. They do not change the All Years macro dashboard above."
    )

    max_duration = max(1, int(pd.to_numeric(events["duration_days"], errors="coerce").fillna(1).max()))
    max_footprint = max(1, int(pd.to_numeric(events["spatial_footprint"], errors="coerce").fillna(1).max()))
    peak_c_values = pd.to_numeric(events["peak_C"], errors="coerce").dropna()
    peak_c_max = float(peak_c_values.max()) if not peak_c_values.empty else ANOMALY_THRESHOLD
    peak_c_min_absolute = float(peak_c_values.min()) if not peak_c_values.empty else ANOMALY_THRESHOLD

    f1, f2, f3 = st.columns(3)

    with f1:
        min_duration = st.slider(
            "Minimum duration (days)",
            min_value=1,
            max_value=max_duration,
            value=1,
        )

    with f2:
        min_footprint = st.slider(
            "Minimum footprint (sectors)",
            min_value=1,
            max_value=max_footprint,
            value=1,
        )

    with f3:
        min_peak_c = st.slider(
            "Minimum peak anomaly (C)",
            min_value=float(min(ANOMALY_THRESHOLD, peak_c_min_absolute)),
            max_value=float(max(ANOMALY_THRESHOLD, peak_c_max)),
            value=float(ANOMALY_THRESHOLD),
            step=0.1,
        )


# ============================================================
# STEP 4 — GUIDED EVENT SELECTION
# ============================================================

st.markdown('<div class="section-kicker">Investigate Specific Events</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">The calendar above identifies when activity is concentrated. Start with the five largest reconstructed events in that same year, then open the map-like 5×5 footprint, lifecycle, and deep evidence only when useful.</div>',
    unsafe_allow_html=True,
)

filtered_events = filter_events_for_year_and_controls(
    events,
    selected_year,
    min_duration,
    min_footprint,
    min_peak_c,
)

featured_events = rank_largest_events(filtered_events).head(5).copy()

if featured_events.empty:
    st.warning(
        "No reconstructed events match the current scientific filters. Expand the Advanced Scientific Filters and lower one or more thresholds."
    )
else:
    display = featured_events[
        [
            "event_id",
            "emergence_date",
            "decay_date",
            "duration_days",
            "spatial_footprint",
            "peak_C",
            "peak_F",
        ]
    ].copy()
    display.insert(0, "Rank", range(1, len(display) + 1))
    display = display.rename(
        columns={
            "event_id": "Event",
            "emergence_date": "Emergence",
            "decay_date": "Decay",
            "duration_days": "Duration (days)",
            "spatial_footprint": "Footprint (sectors)",
            "peak_C": "Peak C (σ)",
            "peak_F": "Peak F (σ)",
        }
    )
    display["Emergence"] = pd.to_datetime(display["Emergence"]).dt.strftime("%d %b %Y")
    display["Decay"] = pd.to_datetime(display["Decay"]).dt.strftime("%d %b %Y")
    display["Peak C (σ)"] = pd.to_numeric(display["Peak C (σ)"], errors="coerce").round(2)
    display["Peak F (σ)"] = pd.to_numeric(display["Peak F (σ)"], errors="coerce").round(2)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Largest is ranked by spatial footprint first, then duration, peak C, and peak F."
    )

    selected_event_id = st.selectbox(
        "Open an event for detailed investigation",
        featured_events["event_id"].tolist(),
        format_func=lambda event_id: format_event_label(
            featured_events.loc[featured_events["event_id"] == event_id].iloc[0]
        ),
        key=f"event_selector_{selected_year}_{min_duration}_{min_footprint}_{min_peak_c}",
    )

    selected_event = featured_events[
        featured_events["event_id"] == selected_event_id
    ].iloc[0]

    event_nodes = build_event_node_profile(science, selected_event)
    daily_event = aggregate_daily_event_profile(event_nodes)

    st.markdown(
        f'<div class="event-card"><div class="event-card-title">{selected_event["event_id"]}</div><div class="event-card-copy">{safe_date_text(selected_event["emergence_date"])} → {safe_date_text(selected_event["decay_date"])} · {int(selected_event["spatial_footprint"])} sectors · peak C {fmt_sigma(selected_event["peak_C"])}</div></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ========================================================
    # STEP 5 — THREE EVENT TABS
    # ========================================================

    tab_overview, tab_lifecycle, tab_deep = st.tabs(
        [
            "Event Overview",
            "Lifecycle Progression",
            "Deep Science Context",
        ]
    )

    # --------------------------------------------------------
    # EVENT OVERVIEW
    # --------------------------------------------------------
    with tab_overview:
        o1, o2, o3, o4, o5 = st.columns(5)

        o1.metric("Duration", f"{int(selected_event['duration_days'])} days")
        o2.metric("Spatial footprint", f"{int(selected_event['spatial_footprint'])} sectors")
        o3.metric("Peak activity C", fmt_sigma(selected_event["peak_C"]))
        o4.metric("Peak radiative F", fmt_sigma(selected_event["peak_F"]))

        g_value = pd.to_numeric(selected_event["mean_G"], errors="coerce")
        g_text = f"{g_value:.3f}" if np.isfinite(g_value) else "—"
        o5.metric(
            "Mean sensor G",
            g_text,
            help="G is a bounded cross-sensor concordance descriptor, not a probability or ground-truth confidence score.",
        )

        st.caption(f"{concordance_label(g_value)}")

        st.subheader("Event evidence and footprint")
        evidence_table = build_evidence_table(selected_event, event_nodes)
        st.dataframe(evidence_table, use_container_width=True, hide_index=True)

        foot_col, note_col = st.columns([1.35, 1])
        with foot_col:
            st.plotly_chart(
                build_footprint_figure(event_nodes),
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True},
            )
        with note_col:
            st.markdown(
                """
                <div class="context-box">
                    <strong>How to read the footprint</strong><br>
                    The grid shows where the reconstructed event's qualifying
                    M7 sector-day nodes occurred. Cell intensity represents
                    the maximum activity anomaly C recorded for that sector
                    within the event envelope.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(" ")
            involved = parse_sector_list(selected_event["sectors_involved"])
            st.write("**Sectors involved**")
            st.write(", ".join(involved) if involved else "No sectors recorded.")
            st.write("**Peak date**")
            st.write(safe_date_text(selected_event["peak_date"]))
            st.write("**Expansion flag**")
            st.write("Yes" if bool(selected_event["expansion_flag"]) else "No")

    # --------------------------------------------------------
    # LIFECYCLE PROGRESSION
    # --------------------------------------------------------
    with tab_lifecycle:
        st.subheader("Daily event progression")

        if daily_event.empty:
            st.info(
                "The selected event does not have reconstructable daily M7 nodes in the available science export."
            )
        else:
            st.plotly_chart(
                build_lifecycle_figure(
                    daily_event,
                    pd.Timestamp(selected_event["peak_date"]),
                ),
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True},
            )

            l1, l2, l3, l4 = st.columns(4)
            l1.metric("Observed event days", len(daily_event))
            l2.metric("Maximum daily footprint", f"{int(daily_event['spatial_footprint'].max())} sectors")
            l3.metric("Lifecycle peak C", fmt_sigma(pd.to_numeric(daily_event["C"], errors="coerce").max()))
            l4.metric("Lifecycle peak F", fmt_sigma(pd.to_numeric(daily_event["F"], errors="coerce").max()))

            lifecycle_display = daily_event.rename(
                columns={
                    "acq_date": "Date",
                    "C": "C (σ)",
                    "F": "F (σ)",
                    "G": "G",
                    "S": "S",
                    "spatial_footprint": "Active sectors",
                    "modis_count": "MODIS detections",
                    "viirs_count": "VIIRS detections",
                }
            ).copy()

            lifecycle_display["Date"] = pd.to_datetime(
                lifecycle_display["Date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")

            st.dataframe(
                lifecycle_display,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "The lifecycle is reconstructed from qualifying M7 sector-day nodes inside the stored M8 event envelope."
            )

    # --------------------------------------------------------
    # DEEP SCIENCE CONTEXT
    # --------------------------------------------------------
    with tab_deep:
        st.subheader("Historical analogues")

        analogue_ids = parse_analogue_ids(selected_event["top_3_analog_event_ids"])
        analogue_distances = [
            item.strip()
            for item in str(selected_event["top_3_analog_distances"]).split(";")
            if item.strip()
        ]

        if not analogue_ids:
            st.info("No historical analogues are stored for this event.")
        else:
            comparison = build_analogue_table(selected_event, events)
            if not comparison.empty:
                st.dataframe(comparison, use_container_width=True, hide_index=True)

            distance_rows = []
            for index, analogue_id in enumerate(analogue_ids[:3]):
                distance_rows.append(
                    {
                        "Analogue rank": index + 1,
                        "Event ID": analogue_id,
                        "Euclidean distance": (
                            analogue_distances[index]
                            if index < len(analogue_distances)
                            else "—"
                        ),
                    }
                )
            st.dataframe(
                pd.DataFrame(distance_rows),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "M8 compares duration, peak C, peak F, and spatial footprint after independent 0–1 normalization. The result is deterministic similarity, not causal equivalence."
            )

        st.subheader("Observation context U")
        st.markdown(
            """
            <div class="context-box">
                <strong>U is context, not a confidence score.</strong><br>
                Retained sensor-state flags and available counts, detection
                confidence, viewing geometry, and acquisition descriptors can
                help interpret the record. They do not establish satellite
                overpass opportunity or prove that no burning occurred. U is
                kept separate from C, F, G, and S rather than being turned
                into an arbitrary probability.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if event_nodes.empty:
            st.info("No event nodes are available for observation-context inspection.")
        else:
            context_rows = []

            def add_context_row(
                name: str,
                description: str,
                series_a: pd.Series | None,
                series_b: pd.Series | None = None,
                decimals: int = 3,
            ) -> None:
                values = []
                if series_a is not None:
                    values.append(pd.to_numeric(series_a, errors="coerce"))
                if series_b is not None:
                    values.append(pd.to_numeric(series_b, errors="coerce"))

                if values:
                    combined = pd.concat(values, ignore_index=True).dropna()
                    value = float(combined.mean()) if not combined.empty else np.nan
                else:
                    value = np.nan

                context_rows.append(
                    {
                        "Context dimension": name,
                        "Observed context": (
                            f"{value:.{decimals}f}" if np.isfinite(value) else "Unavailable"
                        ),
                        "Definition": description,
                    }
                )

            add_context_row(
                "MODIS retained count/context per node",
                "Mean retained MODIS context/count field at qualifying event nodes; not a measure of overpass opportunity.",
                event_nodes.get("u_modis_observation_count"),
                decimals=1,
            )
            add_context_row(
                "VIIRS retained count/context per node",
                "Mean retained VIIRS context/count field at qualifying event nodes; not a measure of overpass opportunity.",
                event_nodes.get("u_viirs_observation_count"),
                decimals=1,
            )
            add_context_row(
                "MODIS confidence",
                "Mean retained MODIS detection-confidence descriptor.",
                event_nodes.get("u_modis_mean_confidence"),
                decimals=2,
            )
            add_context_row(
                "VIIRS confidence",
                "Mean retained VIIRS detection-confidence descriptor.",
                event_nodes.get("u_viirs_mean_confidence"),
                decimals=2,
            )
            add_context_row(
                "Mean scan",
                "Mean scan/viewing-geometry descriptor across available sensors.",
                event_nodes.get("u_modis_mean_scan"),
                event_nodes.get("u_viirs_mean_scan"),
                decimals=2,
            )
            add_context_row(
                "Mean track",
                "Mean track/viewing-geometry descriptor across available sensors.",
                event_nodes.get("u_modis_mean_track"),
                event_nodes.get("u_viirs_mean_track"),
                decimals=2,
            )
            add_context_row(
                "Daytime fraction",
                "Fraction of retained observations classified as daytime.",
                event_nodes.get("u_modis_day_fraction"),
                event_nodes.get("u_viirs_day_fraction"),
                decimals=3,
            )

            if "u_modis_mean_cloud" in event_nodes.columns or "u_viirs_mean_cloud" in event_nodes.columns:
                add_context_row(
                    "Cloud context",
                    "Optional retained cloud-related context when present in the source products.",
                    event_nodes.get("u_modis_mean_cloud"),
                    event_nodes.get("u_viirs_mean_cloud"),
                    decimals=3,
                )

            st.dataframe(
                pd.DataFrame(context_rows),
                use_container_width=True,
                hide_index=True,
            )

            if "u_sensor_state" in event_nodes.columns:
                state_counts = (
                    event_nodes["u_sensor_state"]
                    .astype(str)
                    .value_counts()
                    .rename_axis("Sensor state")
                    .reset_index(name="Sector-days")
                )
                st.write("**Sensor-state distribution**")
                st.dataframe(state_counts, use_container_width=True, hide_index=True)

            st.caption(
                "Missing optional context fields are shown as unavailable rather than filled with invented values."
            )


# ============================================================
# METHOD NOTE
# ============================================================

with st.expander("How to interpret Fire Harmonizer", expanded=False):
    st.markdown(
        """
        **C — Activity anomaly:** how unusual the detected-hotspot count is against its historical reference. The score is not a number of fires or a direct danger rating.

        **F — FRP anomaly:** how unusual summed Fire Radiative Power is against its historical reference. FRP is power, not total fire energy, and can rise with the number of detections.

        **G — Cross-sensor concordance:** agreement between standardized MODIS and VIIRS count signals. It reuses count information behind C, so it is not independent confirmation, ground truth, or a probability.

        **S — Spatial coherence:** neighboring sectors also crossing the C threshold. S is derived from C and is not independent confirmation.

        **U — Observation context:** retained descriptors such as sensor state, associated counts, confidence, and viewing context where available. It is not a confidence percentage or measure of satellite overpass opportunity.

        **M8 reconstructed events:** connected DATE × SECTOR anomaly nodes. They are analytical groupings, not verified fires, ground-truth perimeters, or emergency declarations. The historical baseline can include later dates, so outputs are not real-time warnings.
        """
    )

st.markdown(
    """
    <div class="footer-note">
        <strong>Important:</strong> Fire Harmonizer analyzes satellite fire
        detections and derived statistical evidence. It does not represent
        burned-area measurements, a live emergency service, or ground-truth
        confirmation of individual fires.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OPTIONAL — LIGHTWEIGHT NEAR-TERM ACTIVITY WATCHLIST
# ============================================================

st.markdown('<div class="section-kicker">Near-term activity watchlist</div>', unsafe_allow_html=True)
with st.expander("Experimental 7-day activity watchlist", expanded=False):
    watchlist_candidates = [
        BASE_DIR / "output" / "model" / "next7d_watchlist.csv",
        BASE_DIR / "data" / "output" / "model" / "next7d_watchlist.csv",
    ]
    metrics_candidates = [
        BASE_DIR / "output" / "model" / "model_metrics.json",
        BASE_DIR / "data" / "output" / "model" / "model_metrics.json",
    ]
    watchlist_path = next((p for p in watchlist_candidates if p.exists()), None)
    metrics_path = next((p for p in metrics_candidates if p.exists()), None)

    st.caption(
        "Optional add-on. It ranks sectors by a lightweight model or persistence baseline, "
        "selected using a chronological validation period. The score is not a probability."
    )
    if watchlist_path is None:
        st.info(
            "No model output found yet. From the project root, run:\n\n"
            "`python fire_harmonizer_light_model.py --science output/harmonized_science_all_years.csv --output-dir output/model --horizon 7`"
        )
    else:
        try:
            watch_df = pd.read_csv(watchlist_path, low_memory=False)
            if watch_df.empty:
                st.warning("The saved watchlist is empty. Re-run the model to create sector scores.")
            else:
                as_of = str(watch_df.iloc[0].get("as_of_date", "unknown"))
                forecast_start = str(watch_df.iloc[0].get("forecast_start", "unknown"))
                forecast_end = str(watch_df.iloc[0].get("forecast_end", "unknown"))
                selected_method = str(watch_df.iloc[0].get("selected_method", "unknown"))
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Data available through", as_of)
                k2.metric("Forecast window begins", forecast_start)
                k3.metric("Forecast window ends", forecast_end)
                k4.metric(
                    "Selected approach",
                    "ML" if selected_method == "hist_gradient_boosting" else (
                        "Persistence" if selected_method == "persistence_baseline" else selected_method
                    ),
                )

                # Normalize the watchlist columns and render a high-contrast table.
                # The chart can work while a themed dataframe grid appears blank, so
                # this table uses explicit white backgrounds and dark text.
                required_watch_columns = {"sector", "model_score"}
                missing_watch_columns = required_watch_columns - set(watch_df.columns)
                if missing_watch_columns:
                    st.error(
                        "The watchlist CSV is missing required columns: "
                        + ", ".join(sorted(missing_watch_columns))
                        + f". File loaded: {watchlist_path}"
                    )
                    st.caption("Available columns: " + ", ".join(map(str, watch_df.columns)))
                else:
                    watch_display = watch_df.copy()
                    if "rank" not in watch_display.columns:
                        watch_display["rank"] = np.arange(1, len(watch_display) + 1)
                    if "watchlist_flag" not in watch_display.columns:
                        watch_display["watchlist_flag"] = np.nan
                    if "selected_method" not in watch_display.columns:
                        watch_display["selected_method"] = selected_method
                    watch_display["rank"] = pd.to_numeric(watch_display["rank"], errors="coerce")
                    watch_display["model_score"] = pd.to_numeric(watch_display["model_score"], errors="coerce")
                    watch_display["watchlist_flag"] = pd.to_numeric(
                        watch_display["watchlist_flag"], errors="coerce"
                    )
                    watch_display = watch_display.sort_values(
                        ["rank", "sector"], ascending=[True, True], na_position="last"
                    )
                    watch_display["rank"] = watch_display["rank"].apply(
                        lambda x: str(int(x)) if pd.notna(x) else "—"
                    )
                    watch_display["model_score"] = watch_display["model_score"].apply(
                        lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                    )
                    watch_display["watchlist_flag"] = watch_display["watchlist_flag"].map(
                        lambda x: ("Above threshold" if x == 1 else "Below threshold" if x == 0 else "—")
                    )
                    watch_display["selected_method"] = watch_display["selected_method"].map(
                        lambda x: (
                            "ML" if str(x) == "hist_gradient_boosting" else
                            "Persistence" if str(x) == "persistence_baseline" else str(x)
                        )
                    )
                    watch_display = watch_display[
                        ["rank", "sector", "model_score", "watchlist_flag", "selected_method"]
                    ].rename(
                        columns={
                            "rank": "Rank",
                            "sector": "Sector",
                            "model_score": "Ranking score (not probability)",
                            "watchlist_flag": "Validation-threshold flag",
                            "selected_method": "Selected approach",
                        }
                    )
                    st.markdown(render_readable_html_table(watch_display), unsafe_allow_html=True)

                    chart_source = watch_df.copy()
                    chart_source["model_score"] = pd.to_numeric(chart_source["model_score"], errors="coerce")
                    chart_source = chart_source.dropna(subset=["model_score"])
                    if chart_source.empty:
                        st.info("The watchlist contains no numeric scores to plot.")
                    else:
                        plot_df = chart_source.sort_values("model_score", ascending=True).set_index("sector")[
                            ["model_score"]
                        ]
                        st.bar_chart(plot_df, use_container_width=True)

                if metrics_path is not None:
                    import json
                    try:
                        model_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                        st.subheader("Chronological back-test metrics")

                        # Parse the schema actually written by fire_harmonizer_light_model.py.
                        # It nests selection metadata under final_method_selection and
                        # aggregate validation results under walk_forward_backtest.
                        selection = model_metrics.get("final_method_selection", {})
                        backtest = model_metrics.get("walk_forward_backtest", {})
                        chosen = selection.get(
                            "selected_method", model_metrics.get("selected_method", selected_method)
                        )
                        reason = selection.get(
                            "reason", model_metrics.get("selection_reason", "Not recorded in metrics file")
                        )
                        chosen_label = (
                            "ML" if chosen == "hist_gradient_boosting" else
                            "Persistence" if chosen == "persistence_baseline" else str(chosen)
                        )
                        st.write("**Validation-selected approach:**", chosen_label)
                        st.write("**Validation-based selection reason:**", reason)

                        if backtest:
                            agg_rows = [{
                                "Held-out rows": backtest.get("rows", "—"),
                                "Positive rate": (
                                    f"{100 * float(backtest['positive_rate']):.1f}%"
                                    if backtest.get("positive_rate") is not None else "—"
                                ),
                                "Precision": (
                                    f"{float(backtest['precision']):.3f}"
                                    if backtest.get("precision") is not None else "—"
                                ),
                                "Recall": (
                                    f"{float(backtest['recall']):.3f}"
                                    if backtest.get("recall") is not None else "—"
                                ),
                                "F1": (
                                    f"{float(backtest['f1']):.3f}"
                                    if backtest.get("f1") is not None else "—"
                                ),
                                "Average precision": (
                                    f"{float(backtest['average_precision']):.3f}"
                                    if backtest.get("average_precision") is not None else "—"
                                ),
                                "Test years": ", ".join(map(str, backtest.get("test_years", []))) or "—",
                            }]
                            st.markdown(
                                render_readable_html_table(pd.DataFrame(agg_rows)),
                                unsafe_allow_html=True,
                            )
                        else:
                            legacy = model_metrics.get("test_metrics_by_method", {})
                            if legacy:
                                st.markdown(
                                    render_readable_html_table(pd.DataFrame(legacy).T.reset_index().rename(
                                        columns={"index": "Method"}
                                    )),
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.warning(
                                    "No walk-forward metrics were found in this JSON. "
                                    "Re-run fire_harmonizer_light_model.py to regenerate the model report."
                                )

                        annual_rows = model_metrics.get("annual_fold_summary", [])
                        if annual_rows:
                            st.write("**Annual walk-forward results**")
                            annual_df = pd.DataFrame(annual_rows)
                            preferred_annual = [
                                "test_year", "validation_year", "selected_method", "train_rows",
                                "validation_rows", "test_rows", "test_precision", "test_recall",
                                "test_f1", "test_average_precision", "threshold",
                            ]
                            annual_df = annual_df[[c for c in preferred_annual if c in annual_df.columns]].copy()
                            annual_df = annual_df.rename(columns={
                                "test_year": "Test year",
                                "validation_year": "Validation year",
                                "selected_method": "Selected method",
                                "train_rows": "Training rows",
                                "validation_rows": "Validation rows",
                                "test_rows": "Test rows",
                                "test_precision": "Precision",
                                "test_recall": "Recall",
                                "test_f1": "F1",
                                "test_average_precision": "Average precision",
                                "threshold": "Threshold",
                            })
                            for col in ["Precision", "Recall", "F1", "Average precision", "Threshold"]:
                                if col in annual_df.columns:
                                    annual_df[col] = annual_df[col].apply(
                                        lambda x: f"{float(x):.3f}" if pd.notna(x) else "—"
                                    )
                            st.markdown(
                                render_readable_html_table(annual_df),
                                unsafe_allow_html=True,
                            )
                    except Exception as exc:
                        st.error(f"Metrics file could not be read: {exc}")
                else:
                    st.info(
                        "No model_metrics.json was found. The watchlist is available, but chronological "
                        "back-test metrics will appear only after running the model script successfully."
                    )
        except Exception as exc:
            st.error(f"Could not read the saved watchlist: {exc}")

    st.warning(
        "Research-only interpretation: the target is whether the existing M7 C score reaches the "
        "C-active threshold in the next seven days for the same sector. It does not predict confirmed fires, "
        "burned area, fire severity, or emergency risk. Because historical M7 baselines may include later dates, "
        "this is not a live early-warning product."
    )


# ============================================================
# STEP 6 — M10 STATIC CHARTS AT THE VERY BOTTOM
# ============================================================

st.markdown('<div class="section-kicker">Scientific plates</div>', unsafe_allow_html=True)

with st.expander("M10 Static Scientific Plates", expanded=False):
    st.caption(
        "These figures are generated offline by generate_charts.py. Streamlit only reads the saved PNG files here."
    )

    chart_directory = find_chart_directory()

    for row_start in range(0, len(M10_CHARTS), 2):
        columns = st.columns(2, gap="large")

        for column, (filename, title, description) in zip(
            columns,
            M10_CHARTS[row_start:row_start + 2],
        ):
            with column:
                st.markdown(
                    f'<div class="event-card"><div class="event-card-title">{title}</div><div class="event-card-copy">{description}</div></div>',
                    unsafe_allow_html=True,
                )

                image_path = (
                    chart_directory / filename
                    if chart_directory is not None
                    else None
                )

                if image_path is not None and image_path.exists():
                    st.image(str(image_path), use_container_width=True)
                else:
                    st.info(f"Pre-generated plate not found: {filename}")

