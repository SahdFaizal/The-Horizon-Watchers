"""
FIRE HARMONIZER - MILESTONE 10
================================

STANDALONE OFFLINE STATIC CHART GENERATION

Purpose
-------
Generate the final high-resolution static figures for the Fire Harmonizer
project using the M7 science dataset and M8 reconstructed-event dataset.

Inputs
------
Preferred M7:
    output/harmonized_science_all_years.csv

Preferred M8:
    output/critical_periods.csv

Fallback:
    data/output/harmonized_science_all_years.csv
    data/output/critical_periods.csv

Outputs
-------
    output/charts/01_longitudinal_evidence.png
    output/charts/02_m4_vs_m8_paradigm_shift.png
    output/charts/03_concordance_breakdown.png
    output/charts/04_mega_event_lifecycle.png

Rendering
---------
All figures are generated with Plotly and exported as PNG using Kaleido.

The script is fully offline and independent of the Streamlit UI:
    - no NASA/API requests
    - no Streamlit execution
    - no machine learning
    - no database

Core libraries
--------------
    numpy
    pandas
    streamlit
    plotly.graph_objects
    plotly.express

Plotly PNG export additionally requires:
    kaleido

Scientific interpretation
-------------------------
These figures visualize the project's observational/statistical products.
They do not convert active-fire detections into ground-truth fire
perimeters, burned area, or emergency declarations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import importlib.util
import time

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SCIENCE_CANDIDATES = [
    BASE_DIR / "output" / "harmonized_science_all_years.csv",
    BASE_DIR / "data" / "output" / "harmonized_science_all_years.csv",
]

EVENT_CANDIDATES = [
    BASE_DIR / "output" / "critical_periods.csv",
    BASE_DIR / "data" / "output" / "critical_periods.csv",
]

CHART_DIR = BASE_DIR / "output" / "charts"


# ============================================================
# PROJECT CONSTANTS
# ============================================================

ANOMALY_THRESHOLD = 2.0
LEGACY_M4_MIN_DAYS = 3
GRID_SIZE = 5
DEFAULT_MEGA_EVENT_ID = "EVT-2024-1610"

# Plotly static-export dimensions.
# High width/height plus scale=2 gives crisp PNGs suitable for
# project pages, documentation, judging screenshots, and slides.
LONGITUDINAL_SIZE = (1500, 1150)
PARADIGM_SIZE = (1550, 900)
CONCORDANCE_SIZE = (1450, 900)
LIFECYCLE_SIZE = (1550, 1000)

EXPORT_SCALE = 2

# Plotly's white publication-style theme replaces the old
# Matplotlib/Seaborn styling layer.
PLOTLY_TEMPLATE = "plotly_white"


# ============================================================
# RUNTIME CHECKS
# ============================================================


def require_kaleido() -> None:
    """
    Ensure Plotly's static image export engine is available.

    Plotly itself can build figures without Kaleido, but PNG export
    requires an image-export backend. Failing early is preferable to
    producing only interactive HTML when the requested output is PNG.
    """
    if importlib.util.find_spec("kaleido") is None:
        raise RuntimeError(
            "Plotly PNG export requires Kaleido, but Kaleido is not installed.\n\n"
            "Install it with:\n"
            "    pip install kaleido\n\n"
            "Then rerun:\n"
            "    python make_chartsv2.py"
        )


# ============================================================
# PATH HELPERS
# ============================================================


def first_existing(paths: list[Path]) -> Path | None:
    """Return the first existing path from an ordered candidate list."""
    for path in paths:
        if path.exists():
            return path
    return None


def require_path(
    paths: list[Path],
    description: str,
) -> Path:
    """Resolve a required input path or raise an informative error."""
    path = first_existing(paths)

    if path is None:
        checked = "\n".join(
            f"  {candidate}"
            for candidate in paths
        )

        raise FileNotFoundError(
            f"Could not find {description}.\n\n"
            f"Checked:\n{checked}"
        )

    return path


# ============================================================
# DATA LOADING
# ============================================================


def load_science(path: Path) -> pd.DataFrame:
    """
    Load and normalize the M7 all-years science dataset.

    M7's preferred evidence fields are:
        m7_C
        m7_F
        m7_G
        m7_S

    Legacy M3 fields are retained when available:
        anomaly_z
        modis_z
        viirs_z
        harmonized_activity
    """
    df = pd.read_csv(path)

    required = [
        "acq_date",
        "sector",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "M7 science dataset is missing required columns:\n"
            + "\n".join(missing)
        )

    df["acq_date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["acq_date", "sector"],
    ).copy()

    df["sector"] = (
        df["sector"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    numeric_columns = [
        "modis_count",
        "viirs_count",
        "modis_frp",
        "viirs_frp",
        "total_frp",
        "modis_z",
        "viirs_z",
        "harmonized_activity",
        "sensor_discrepancy",
        "anomaly_z",
        "m7_C",
        "m7_F",
        "m7_G",
        "m7_S",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    m7_missing = [
        column
        for column in [
            "m7_C",
            "m7_F",
            "m7_G",
        ]
        if column not in df.columns
    ]

    if m7_missing:
        raise ValueError(
            "M7 science dataset is missing final evidence fields:\n"
            + "\n".join(m7_missing)
        )

    if "m7_S" not in df.columns:
        df["m7_S"] = np.nan

    return (
        df.sort_values(
            ["acq_date", "sector"],
        )
        .reset_index(drop=True)
    )


def load_events(path: Path) -> pd.DataFrame:
    """
    Load the M8 reconstructed-event table.
    """
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
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "M8 event dataset is missing required columns:\n"
            + "\n".join(missing)
        )

    for column in [
        "emergence_date",
        "decay_date",
        "peak_date",
    ]:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

    for column in [
        "duration_days",
        "spatial_footprint",
        "peak_C",
        "peak_F",
        "mean_G",
        "mean_S",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["expansion_flag"] = (
        df["expansion_flag"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    return (
        df.sort_values(
            [
                "emergence_date",
                "decay_date",
                "event_id",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# COLUMN RESOLUTION
# ============================================================


def resolve_legacy_activity_column(
    science: pd.DataFrame,
) -> str:
    """
    Select the closest available representation of the old M4 anomaly.

    Priority:
        1. M3 anomaly_z
        2. M7 m7_C
    """
    if "anomaly_z" in science.columns:
        return "anomaly_z"

    return "m7_C"


# ============================================================
# DAILY BASIN AGGREGATION
# ============================================================


def build_daily_longitudinal_profile(
    science: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate the M7 sector-day matrix to a basin-wide daily evidence profile.

    Daily statistics:
        C = mean M7 activity anomaly
        F = mean M7 radiative anomaly
        G = mean M7 cross-sensor concordance
    """
    daily = (
        science.groupby(
            "acq_date",
            as_index=False,
        )
        .agg(
            C=("m7_C", "mean"),
            F=("m7_F", "mean"),
            G=("m7_G", "mean"),
            sector_count=("sector", "nunique"),
        )
        .sort_values("acq_date")
        .reset_index(drop=True)
    )

    return daily


# ============================================================
# FIGURE EXPORT
# ============================================================


def write_png(
    figure: go.Figure,
    output_path: Path,
    width: int,
    height: int,
) -> None:
    """
    Export a Plotly figure as a high-resolution PNG.

    Kaleido is Plotly's static image export backend.
    """
    figure.write_image(
        str(output_path),
        format="png",
        width=width,
        height=height,
        scale=EXPORT_SCALE,
    )


# ============================================================
# FIGURE 1
# ============================================================


def chart_01_longitudinal_evidence(
    science: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate the three-panel longitudinal evidence chart.

    Panel 1:
        Daily Activity Anomaly C

    Panel 2:
        Daily Radiative Anomaly F

    Panel 3:
        Daily Cross-Sensor Concordance G
    """
    daily = build_daily_longitudinal_profile(
        science
    )

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=[
            "Daily Activity Anomaly (C)",
            "Daily Radiative Anomaly (F)",
            "Daily Cross-Sensor Concordance (G)",
        ],
    )

    # --------------------------------------------------------
    # C
    # --------------------------------------------------------
    figure.add_trace(
        go.Scatter(
            x=daily["acq_date"],
            y=daily["C"],
            mode="lines",
            name="Activity anomaly C",
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "C: %{y:.3f}σ"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    figure.add_hline(
        y=ANOMALY_THRESHOLD,
        line_dash="dash",
        line_width=1.1,
        annotation_text="C = 2σ",
        annotation_position="top right",
        row=1,
        col=1,
    )

    # --------------------------------------------------------
    # F
    # --------------------------------------------------------
    figure.add_trace(
        go.Scatter(
            x=daily["acq_date"],
            y=daily["F"],
            mode="lines",
            name="Radiative anomaly F",
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "F: %{y:.3f}σ"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    figure.add_hline(
        y=0,
        line_dash="dot",
        line_width=1,
        row=2,
        col=1,
    )

    # --------------------------------------------------------
    # G
    # --------------------------------------------------------
    figure.add_trace(
        go.Scatter(
            x=daily["acq_date"],
            y=daily["G"],
            mode="lines",
            name="Sensor concordance G",
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "G: %{y:.3f}"
                "<extra></extra>"
            ),
        ),
        row=3,
        col=1,
    )

    figure.add_hline(
        y=0.80,
        line_dash="dash",
        line_width=1.0,
        annotation_text="High concordance = 0.80",
        annotation_position="top right",
        row=3,
        col=1,
    )

    figure.update_yaxes(
        title_text="C (σ)",
        row=1,
        col=1,
    )

    figure.update_yaxes(
        title_text="F (σ)",
        row=2,
        col=1,
    )

    figure.update_yaxes(
        title_text="G",
        range=[0, 1],
        row=3,
        col=1,
    )

    figure.update_xaxes(
        title_text="Acquisition date",
        row=3,
        col=1,
    )

    figure.update_layout(
        template=PLOTLY_TEMPLATE,
        title={
            "text": (
                "Fire Harmonizer — "
                "Longitudinal Multi-Evidence Record"
            ),
            "x": 0.5,
        },
        height=LONGITUDINAL_SIZE[1],
        hovermode="x unified",
        margin=dict(
            l=70,
            r=30,
            t=95,
            b=55,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
    )

    write_png(
        figure,
        output_path,
        *LONGITUDINAL_SIZE,
    )


# ============================================================
# LEGACY M4 RECONSTRUCTION
# ============================================================


def legacy_m4_durations(
    science: pd.DataFrame,
) -> pd.Series:
    """
    Reconstruct the old M4 single-sector duration distribution.

    Legacy logic:
        - evaluate one sector at a time
        - anomaly >= 2
        - require >= 3 consecutive calendar days
    """
    activity_column = resolve_legacy_activity_column(
        science
    )

    work = science[
        [
            "acq_date",
            "sector",
            activity_column,
        ]
    ].copy()

    work[activity_column] = pd.to_numeric(
        work[activity_column],
        errors="coerce",
    )

    qualifying = work[
        work[activity_column] >= ANOMALY_THRESHOLD
    ].copy()

    if qualifying.empty:
        return pd.Series(
            dtype=float,
            name="legacy_duration_days",
        )

    durations: list[int] = []

    for sector, group in qualifying.groupby(
        "sector",
        sort=True,
    ):
        dates = (
            pd.to_datetime(
                group["acq_date"],
            )
            .drop_duplicates()
            .sort_values()
        )

        gap = dates.diff().dt.days

        run_id = (
            gap.fillna(1)
            .ne(1)
            .cumsum()
        )

        for _, run in dates.groupby(run_id):
            duration = int(
                (run.iloc[-1] - run.iloc[0]).days
                + 1
            )

            if duration >= LEGACY_M4_MIN_DAYS:
                durations.append(duration)

    return pd.Series(
        durations,
        dtype=float,
        name="legacy_duration_days",
    )


# ============================================================
# FIGURE 2
# ============================================================


def chart_02_paradigm_shift(
    science: pd.DataFrame,
    events: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Compare legacy M4 single-sector periods against M8 reconstructed events.

    Panels:
        1. Legacy M4 duration distribution
        2. M8 event duration distribution
        3. M8 spatial footprint distribution
    """
    old_durations = legacy_m4_durations(
        science
    )

    new_durations = (
        pd.to_numeric(
            events["duration_days"],
            errors="coerce",
        )
        .dropna()
    )

    footprints = (
        pd.to_numeric(
            events["spatial_footprint"],
            errors="coerce",
        )
        .dropna()
    )

    figure = make_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=0.08,
        subplot_titles=[
            "Legacy M4 — Single-sector periods",
            "M8 — Reconstructed event duration",
            "M8 — Spatial footprint",
        ],
    )

    # --------------------------------------------------------
    # Legacy M4 duration
    # --------------------------------------------------------
    if not old_durations.empty:
        figure.add_trace(
            go.Histogram(
                x=old_durations,
                name="Legacy M4",
                nbinsx=min(
                    35,
                    max(
                        8,
                        int(old_durations.nunique()),
                    ),
                ),
                hovertemplate=(
                    "Duration: %{x} days<br>"
                    "Count: %{y}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # --------------------------------------------------------
    # M8 duration
    # --------------------------------------------------------
    if not new_durations.empty:
        figure.add_trace(
            go.Histogram(
                x=new_durations,
                name="M8 duration",
                nbinsx=min(
                    35,
                    max(
                        8,
                        int(new_durations.nunique()),
                    ),
                ),
                hovertemplate=(
                    "Duration: %{x} days<br>"
                    "Count: %{y}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    # --------------------------------------------------------
    # M8 footprint
    # --------------------------------------------------------
    if not footprints.empty:
        figure.add_trace(
            go.Histogram(
                x=footprints,
                name="M8 footprint",
                xbins=dict(
                    start=0.5,
                    end=max(
                        1.5,
                        float(footprints.max()) + 0.5,
                    ),
                    size=1,
                ),
                hovertemplate=(
                    "Sectors: %{x}<br>"
                    "Count: %{y}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=3,
        )

    figure.update_xaxes(
        title_text="Duration (days)",
        row=1,
        col=1,
    )

    figure.update_xaxes(
        title_text="Duration (days)",
        row=1,
        col=2,
    )

    figure.update_xaxes(
        title_text="Unique sectors",
        row=1,
        col=3,
    )

    figure.update_yaxes(
        title_text="Frequency",
        row=1,
        col=1,
    )

    figure.update_yaxes(
        title_text="Frequency",
        row=1,
        col=2,
    )

    figure.update_yaxes(
        title_text="Frequency",
        row=1,
        col=3,
    )

    figure.update_layout(
        template=PLOTLY_TEMPLATE,
        title={
            "text": "Fire Harmonizer — M4 → M8 Paradigm Shift",
            "x": 0.5,
        },
        height=PARADIGM_SIZE[1],
        showlegend=False,
        margin=dict(
            l=65,
            r=30,
            t=90,
            b=55,
        ),
        bargap=0.08,
    )

    write_png(
        figure,
        output_path,
        *PARADIGM_SIZE,
    )


# ============================================================
# FIGURE 3
# ============================================================


def chart_03_concordance_breakdown(
    science: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot M7 C versus F for sector-day observations, color-coded by G.
    """
    frame = science[
        [
            "m7_C",
            "m7_F",
            "m7_G",
        ]
    ].copy()

    frame = frame.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if frame.empty:
        raise ValueError(
            "No finite M7 C/F/G values are available "
            "for the concordance plot."
        )

    # The full M7 matrix is ~131,900 rows in the project architecture.
    # Still, a deterministic cap keeps the static raster lightweight if
    # a substantially larger future archive is used.
    max_points = 120_000

    if len(frame) > max_points:
        step = int(
            np.ceil(
                len(frame)
                / max_points
            )
        )

        frame = frame.iloc[
            ::step
        ].copy()

    figure = px.scatter(
        frame,
        x="m7_C",
        y="m7_F",
        color="m7_G",
        color_continuous_scale="Viridis",
        labels={
            "m7_C": "Activity Anomaly C (σ)",
            "m7_F": "Radiative Anomaly F (σ)",
            "m7_G": "Cross-Sensor Concordance G",
        },
        title="Fire Harmonizer — Concordance Breakdown",
    )

    figure.update_traces(
        marker=dict(
            size=6,
            opacity=0.58,
        ),
        hovertemplate=(
            "C: %{x:.3f}σ<br>"
            "F: %{y:.3f}σ<br>"
            "G: %{marker.color:.3f}"
            "<extra></extra>"
        ),
    )

    figure.add_vline(
        x=ANOMALY_THRESHOLD,
        line_dash="dash",
        line_width=1.1,
        annotation_text="C = 2σ",
        annotation_position="top right",
    )

    figure.add_hline(
        y=0,
        line_dash="dot",
        line_width=1,
    )

    figure.update_layout(
        template=PLOTLY_TEMPLATE,
        height=CONCORDANCE_SIZE[1],
        margin=dict(
            l=70,
            r=35,
            t=80,
            b=60,
        ),
        coloraxis_colorbar=dict(
            title="G",
        ),
    )

    write_png(
        figure,
        output_path,
        *CONCORDANCE_SIZE,
    )


# ============================================================
# M8 EVENT HELPERS
# ============================================================


def parse_sector_list(value: object) -> list[str]:
    """Parse M8's semicolon-separated sectors_involved field."""
    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    sectors = [
        item.strip().upper()
        for item in str(value).split(";")
        if item.strip()
    ]

    valid = [
        item
        for item in sectors
        if (
            item.startswith("C")
            and item[1:].isdigit()
            and 1 <= int(item[1:]) <= GRID_SIZE * GRID_SIZE
        )
    ]

    return sorted(
        set(valid),
        key=lambda item: int(
            item.replace("C", "")
        ),
    )


def find_mega_event(
    events: pd.DataFrame,
) -> pd.Series:
    """
    Prefer EVT-2024-1610 when present.

    If it is unavailable, deterministically fall back to the event with
    the largest spatial footprint.
    """
    named = events[
        events["event_id"] == DEFAULT_MEGA_EVENT_ID
    ]

    if not named.empty:
        return named.iloc[0]

    ordered = (
        events.sort_values(
            [
                "spatial_footprint",
                "duration_days",
                "event_id",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
    )

    if ordered.empty:
        raise ValueError(
            "The M8 event dataset contains no reconstructed events."
        )

    return ordered.iloc[0]


def build_event_daily_profile(
    science: pd.DataFrame,
    event: pd.Series,
) -> pd.DataFrame:
    """
    Recover the selected M8 event's daily profile from the M7 science table.

    Included rows satisfy:
        emergence_date <= date <= decay_date
        sector in sectors_involved
        m7_C >= 2
    """
    emergence = pd.Timestamp(
        event["emergence_date"]
    )

    decay = pd.Timestamp(
        event["decay_date"]
    )

    sectors = parse_sector_list(
        event["sectors_involved"]
    )

    if not sectors:
        return pd.DataFrame()

    mask = (
        (science["acq_date"] >= emergence)
        & (science["acq_date"] <= decay)
        & (science["sector"].isin(sectors))
        & (
            pd.to_numeric(
                science["m7_C"],
                errors="coerce",
            )
            >= ANOMALY_THRESHOLD
        )
    )

    nodes = (
        science.loc[mask]
        .copy()
        .sort_values(
            [
                "acq_date",
                "sector",
            ]
        )
    )

    if nodes.empty:
        return pd.DataFrame()

    daily = (
        nodes.groupby(
            "acq_date",
            as_index=False,
        )
        .agg(
            C=("m7_C", "max"),
            F=("m7_F", "max"),
            G=("m7_G", "mean"),
            active_sectors=("sector", "nunique"),
        )
        .sort_values("acq_date")
        .reset_index(drop=True)
    )

    return daily


# ============================================================
# FIGURE 4
# ============================================================


def chart_04_mega_event_lifecycle(
    science: pd.DataFrame,
    events: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Detail the requested mega-event lifecycle.

    Preference:
        EVT-2024-1610

    Fallback:
        largest spatial-footprint event

    Top panel:
        C + F

    Bottom panel:
        G + daily active-sector count on secondary axis
    """
    event = find_mega_event(
        events
    )

    daily = build_event_daily_profile(
        science,
        event,
    )

    if daily.empty:
        raise ValueError(
            f"No reconstructable M7 daily profile was found for "
            f"{event['event_id']}."
        )

    peak_date = pd.Timestamp(
        event["peak_date"]
    )

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.11,
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": True}],
        ],
        subplot_titles=[
            "Activity and Radiative Evolution",
            "Sensor Concordance and Spatial Footprint",
        ],
    )

    # --------------------------------------------------------
    # Top: C + F
    # --------------------------------------------------------
    figure.add_trace(
        go.Scatter(
            x=daily["acq_date"],
            y=daily["C"],
            mode="lines+markers",
            name="C — Activity anomaly",
            line=dict(width=2),
            marker=dict(size=5),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "C: %{y:.3f}σ"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=daily["acq_date"],
            y=daily["F"],
            mode="lines+markers",
            name="F — Radiative anomaly",
            line=dict(
                width=1.8,
                dash="dash",
            ),
            marker=dict(size=4),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "F: %{y:.3f}σ"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    figure.add_hline(
        y=ANOMALY_THRESHOLD,
        line_dash="dot",
        line_width=1,
        annotation_text="C = 2σ",
        annotation_position="bottom right",
        row=1,
        col=1,
    )

    # --------------------------------------------------------
    # Bottom: G + active sectors
    # --------------------------------------------------------
    figure.add_trace(
        go.Scatter(
            x=daily["acq_date"],
            y=daily["G"],
            mode="lines+markers",
            name="G — Sensor concordance",
            line=dict(width=2),
            marker=dict(size=5),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "G: %{y:.3f}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    figure.add_trace(
        go.Scatter(
            x=daily["acq_date"],
            y=daily["active_sectors"],
            mode="lines+markers",
            name="Daily active sectors",
            line=dict(
                width=1.9,
                dash="dashdot",
            ),
            marker=dict(
                size=5,
                symbol="square",
            ),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "Active sectors: %{y}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    figure.update_yaxes(
        title_text="Anomaly (σ)",
        row=1,
        col=1,
    )

    figure.update_yaxes(
        title_text="G (0–1)",
        range=[0, 1.05],
        row=2,
        col=1,
        secondary_y=False,
    )

    figure.update_yaxes(
        title_text="Active sectors",
        row=2,
        col=1,
        secondary_y=True,
    )

    figure.add_vline(
        x=peak_date,
        line_dash="dot",
        line_width=2,
        annotation_text="Peak",
        annotation_position="top left",
    )

    figure.update_xaxes(
        title_text="Acquisition date",
        row=2,
        col=1,
    )

    figure.update_layout(
        template=PLOTLY_TEMPLATE,
        title={
            "text": (
                "Fire Harmonizer — "
                f"{event['event_id']} Event Lifecycle"
                f" | {int(event['spatial_footprint'])} sectors"
            ),
            "x": 0.5,
        },
        height=LIFECYCLE_SIZE[1],
        hovermode="x unified",
        margin=dict(
            l=70,
            r=80,
            t=105,
            b=60,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
    )

    write_png(
        figure,
        output_path,
        *LIFECYCLE_SIZE,
    )


# ============================================================
# SUMMARY REPORT
# ============================================================


def write_summary(
    science_path: Path,
    events_path: Path,
    output_paths: list[Path],
    elapsed_seconds: float,
) -> None:
    """Write a compact deterministic M10 generation report."""
    report_path = (
        CHART_DIR
        / "m10_chart_generation_summary.txt"
    )

    lines = [
        "FIRE HARMONIZER - MILESTONE 10",
        "================================",
        "",
        f"M7 input: {science_path}",
        f"M8 input: {events_path}",
        "",
        "Generated charts:",
        *[
            f"  {path.name}"
            for path in output_paths
        ],
        "",
        f"Generation time: {elapsed_seconds:.2f} seconds",
        "",
        "Rendering:",
        f"- Plotly template: {PLOTLY_TEMPLATE}",
        f"- PNG export scale: {EXPORT_SCALE}×",
        "- Plotly static export backend: Kaleido",
        "",
        "Method notes:",
        "- Figure 01 uses daily basin-wide means of M7 C/F/G.",
        "- Figure 02 reconstructs legacy M4 single-sector runs using "
        "anomaly_z when available, otherwise m7_C.",
        "- Figure 03 uses M7 C/F/G sector-day evidence.",
        "- Figure 04 reconstructs the selected M8 event from M7 nodes.",
        "- No external API calls or machine-learning libraries were used.",
    ]

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    start = time.perf_counter()

    print("=" * 70)
    print("FIRE HARMONIZER - MILESTONE 10")
    print("OFFLINE STATIC CHART GENERATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check static PNG export support.
    # --------------------------------------------------------
    require_kaleido()

    # --------------------------------------------------------
    # Prepare output directory.
    # --------------------------------------------------------
    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Resolve inputs.
    # --------------------------------------------------------
    science_path = require_path(
        SCIENCE_CANDIDATES,
        "the M7 science dataset",
    )

    events_path = require_path(
        EVENT_CANDIDATES,
        "the M8 reconstructed event dataset",
    )

    print(
        f"\nM7 science:\n  {science_path}"
    )

    print(
        f"\nM8 events:\n  {events_path}"
    )

    # --------------------------------------------------------
    # Load data.
    # --------------------------------------------------------
    print(
        "\n[1/6] Loading M7 science..."
    )

    science = load_science(
        science_path
    )

    print(
        f"  Rows: {len(science):,}"
    )

    print(
        "\n[2/6] Loading M8 events..."
    )

    events = load_events(
        events_path
    )

    print(
        f"  Events: {len(events):,}"
    )

    # --------------------------------------------------------
    # Output paths.
    # --------------------------------------------------------
    output_paths = [
        CHART_DIR / "01_longitudinal_evidence.png",
        CHART_DIR / "02_m4_vs_m8_paradigm_shift.png",
        CHART_DIR / "03_concordance_breakdown.png",
        CHART_DIR / "04_mega_event_lifecycle.png",
    ]

    # --------------------------------------------------------
    # Figure 01
    # --------------------------------------------------------
    print(
        "\n[3/6] Generating 01_longitudinal_evidence.png..."
    )

    chart_01_longitudinal_evidence(
        science,
        output_paths[0],
    )

    # --------------------------------------------------------
    # Figure 02
    # --------------------------------------------------------
    print(
        "[4/6] Generating 02_m4_vs_m8_paradigm_shift.png..."
    )

    chart_02_paradigm_shift(
        science,
        events,
        output_paths[1],
    )

    # --------------------------------------------------------
    # Figure 03
    # --------------------------------------------------------
    print(
        "[5/6] Generating 03_concordance_breakdown.png..."
    )

    chart_03_concordance_breakdown(
        science,
        output_paths[2],
    )

    # --------------------------------------------------------
    # Figure 04
    # --------------------------------------------------------
    print(
        "[6/6] Generating 04_mega_event_lifecycle.png..."
    )

    chart_04_mega_event_lifecycle(
        science,
        events,
        output_paths[3],
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    write_summary(
        science_path,
        events_path,
        output_paths,
        elapsed,
    )

    # --------------------------------------------------------
    # Final console summary.
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("MILESTONE 10 COMPLETE")
    print("=" * 70)

    for output_path in output_paths:
        print(
            f"\nSaved:\n  {output_path}"
        )

    print(
        f"\nGeneration time: {elapsed:.2f} seconds"
    )

    if elapsed <= 30:
        print(
            "Performance target: PASS (under 30 seconds)"
        )
    else:
        print(
            "Performance target: WARNING (over 30 seconds)"
        )

    print(
        "\nRendering engine: Plotly + Kaleido"
    )

    print(
        f"\nSummary report:\n  "
        f"{CHART_DIR / 'm10_chart_generation_summary.txt'}"
    )


if __name__ == "__main__":
    main()
