"""
FIRE HARMONIZER — MILESTONE 6
================================

FINAL AUTHORITATIVE STREAMLIT APPLICATION

Purpose:
    Present the precomputed M1–M5 scientific outputs in one
    interactive application.

Architecture:
    NASA FIRMS
        ↓
    M1 Preprocessing
        ↓
    M2 Validation
        ↓
    M3 Cross-Sensor Harmonization
        ↓
    M4 Critical-Period Detection
        ↓
    M5 Scientific Visualization
        ↓
    M6 Final Interactive Application

Design principles:
    - M6 is the final authoritative application.
    - No NASA downloading.
    - No maps, GIS, globe, 3D, database, authentication,
      React, or Flask.
    - M3 harmonized activity remains the primary metric.
    - FRP remains a separate metric.
    - Missing observation dates remain missing.
    - One global analysis-scope selector controls the app.
    - All-years views appear once; there is no duplicate
      All Years tab.
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PATHS / SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

ALL_YEARS_FILE = OUTPUT_DIR / "harmonized_science_all_years.csv"
CRITICAL_PERIODS_FILE = OUTPUT_DIR / "critical_periods.csv"

GRID_SIZE = 5
ROLLING_WINDOW_DAYS = 30

EXPECTED_SECTORS = [
    f"C{i:02d}" for i in range(1, GRID_SIZE * GRID_SIZE + 1)
]

SCIENCE_COLUMNS = [
    "acq_date",
    "sector",
    "anomaly_z",
    "modis_z",
    "viirs_z",
    "harmonized_activity",
    "total_frp",
    "sensor_discrepancy",
]

METRIC_OPTIONS = {
    "harmonized_activity": "Harmonized Activity",
    "anomaly_z": "Anomaly Z",
    "modis_z": "MODIS Z",
    "viirs_z": "VIIRS Z",
    "sensor_discrepancy": "Sensor Discrepancy",
    "total_frp": "Total FRP",
}


# ============================================================
# PAGE CONFIG / CSS
# ============================================================

st.set_page_config(
    page_title="Fire Harmonizer",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        div[data-testid="stMetric"] {
            padding: 0.3rem 0;
        }

        .section-note {
            color: #666666;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_science_data():
    """
    Load the complete M3 science dataset.

    Preferred:
        harmonized_science_all_years.csv

    Fallback:
        harmonized_science_YYYY.csv files
    """
    frames = []

    if ALL_YEARS_FILE.exists():
        df = pd.read_csv(ALL_YEARS_FILE)
        if df.empty:
            raise RuntimeError(
                "harmonized_science_all_years.csv is empty."
            )
        source = "harmonized_science_all_years.csv"
        frames.append(df)
    else:
        prefix = "harmonized_science_"
        yearly_files = []

        for path in OUTPUT_DIR.glob(f"{prefix}*.csv"):
            year_text = path.stem[len(prefix):]
            if year_text.isdigit():
                yearly_files.append((int(year_text), path))

        yearly_files.sort(key=lambda item: item[0])

        if not yearly_files:
            raise FileNotFoundError(
                "No harmonized science files were found in output/."
            )

        for _, path in yearly_files:
            frame = pd.read_csv(path)
            if not frame.empty:
                frames.append(frame)

        source = "yearly harmonized_science_YYYY.csv files"

    if not frames:
        raise RuntimeError("No usable harmonized science data was found.")

    df = pd.concat(frames, ignore_index=True)

    missing = [c for c in SCIENCE_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Science data is missing required columns: "
            + ", ".join(missing)
        )

    df = df[SCIENCE_COLUMNS].copy()

    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.normalize()

    df["sector"] = (
        df["sector"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    numeric_columns = [
        "anomaly_z",
        "modis_z",
        "viirs_z",
        "harmonized_activity",
        "total_frp",
        "sensor_discrepancy",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=SCIENCE_COLUMNS
    ).copy()

    if df.empty:
        raise RuntimeError("No valid science rows remain.")

    unexpected = set(df["sector"].unique()).difference(EXPECTED_SECTORS)
    if unexpected:
        raise RuntimeError(
            "Unexpected sector code(s): "
            + ", ".join(sorted(unexpected))
        )

    df["year"] = df["acq_date"].dt.year

    return (
        df.sort_values(["acq_date", "sector"]).reset_index(drop=True),
        source,
    )


@st.cache_data
def load_critical_periods():
    """
    Load M4 critical periods.
    """
    if not CRITICAL_PERIODS_FILE.exists():
        return pd.DataFrame()

    periods = pd.read_csv(CRITICAL_PERIODS_FILE)

    if periods.empty:
        return periods

    required = [
        "sector",
        "start_date",
        "end_date",
        "duration_days",
        "peak_anomaly_z",
    ]

    missing = [c for c in required if c not in periods.columns]
    if missing:
        raise RuntimeError(
            "critical_periods.csv is missing columns: "
            + ", ".join(missing)
        )

    for column in ["start_date", "end_date", "peak_date"]:
        if column in periods.columns:
            periods[column] = pd.to_datetime(
                periods[column],
                errors="coerce",
            )

    for column in [
        "duration_days",
        "peak_anomaly_z",
        "mean_anomaly_z",
        "mean_harmonized_activity",
        "peak_frp",
        "mean_frp",
        "mean_sensor_discrepancy",
        "mean_modis_z",
        "mean_viirs_z",
        "extreme_days",
    ]:
        if column in periods.columns:
            periods[column] = pd.to_numeric(
                periods[column],
                errors="coerce",
            )

    periods["sector"] = (
        periods["sector"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    periods = periods.dropna(
        subset=required
    ).copy()

    periods = periods[
        periods["end_date"] >= periods["start_date"]
    ].copy()

    return periods.sort_values(
        ["start_date", "sector"]
    ).reset_index(drop=True)


# ============================================================
# COMMON DATA HELPERS
# ============================================================

def filter_scope(data, all_years, year):
    """Return the data represented by the current global scope."""
    if all_years:
        return data
    return data[data["year"] == year].copy()


def filter_period_scope(periods, all_years, year):
    """Return M4 periods represented by the current scope."""
    if periods.empty or all_years:
        return periods.copy()

    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year}-12-31")

    return periods[
        (periods["end_date"] >= year_start)
        & (periods["start_date"] <= year_end)
    ].copy()


def add_period_severity(periods):
    """Assign the single displayed M4 severity category."""
    result = periods.copy()
    result["severity"] = np.where(
        result["peak_anomaly_z"] >= 3.0,
        "Extreme peak (Z ≥ 3)",
        "Peak anomaly (2 ≤ Z < 3)",
    )
    return result


def create_daily_data(data):
    """
    Aggregate observed sector records to one row per observed date.

    Missing calendar dates remain absent rather than being replaced
    with zero.
    """
    daily = (
        data.groupby("acq_date", as_index=False)
        .agg(
            mean_harmonized_activity=("harmonized_activity", "mean"),
            mean_anomaly_z=("anomaly_z", "mean"),
            mean_modis_z=("modis_z", "mean"),
            mean_viirs_z=("viirs_z", "mean"),
            mean_sensor_discrepancy=("sensor_discrepancy", "mean"),
            mean_frp=("total_frp", "mean"),
            total_frp=("total_frp", "sum"),
            sectors_observed=("sector", "nunique"),
        )
        .sort_values("acq_date")
        .reset_index(drop=True)
    )

    indexed = daily.set_index("acq_date")

    window = f"{ROLLING_WINDOW_DAYS}D"

    indexed["harmonized_30d"] = (
        indexed["mean_harmonized_activity"]
        .rolling(window, min_periods=1)
        .mean()
    )

    indexed["anomaly_30d"] = (
        indexed["mean_anomaly_z"]
        .rolling(window, min_periods=1)
        .mean()
    )

    indexed["discrepancy_30d"] = (
        indexed["mean_sensor_discrepancy"]
        .rolling(window, min_periods=1)
        .mean()
    )

    return indexed.reset_index()


def create_year_calendar(data, year, metric_column):
    """
    Build Jan 1–Dec 31 for one selected year.
    Unobserved dates remain NaN.
    """
    dates = pd.date_range(
        f"{year}-01-01",
        f"{year}-12-31",
        freq="D",
    )

    observed = data.groupby("acq_date")[metric_column].mean()

    return pd.DataFrame(
        {
            "date": dates,
            "value": observed.reindex(dates).values,
        }
    )


def create_all_year_calendar(data, metric_column):
    """
    Build year × day-of-year matrix for the full available period.
    Missing observation dates remain NaN.
    """
    work = data[["acq_date", "year", metric_column]].copy()
    work["day_of_year"] = work["acq_date"].dt.dayofyear

    grouped = (
        work.groupby(["year", "day_of_year"])[metric_column]
        .mean()
    )

    years = list(range(int(work["year"].min()), int(work["year"].max()) + 1))

    return (
        grouped.unstack("day_of_year")
        .reindex(
            index=years,
            columns=range(1, 367),
        )
    )


# ============================================================
# CHART BUILDERS
# ============================================================

def build_calendar_chart(calendar_df, year, metric_label):
    """Selected-year one-row calendar."""
    dates = calendar_df["date"].tolist()
    values = calendar_df["value"].tolist()

    hover = [
        (
            f"{date:%d %b %Y}"
            if pd.isna(value)
            else f"{date:%d %b %Y}<br>{metric_label}: {value:.3f}"
        )
        for date, value in zip(dates, values)
    ]

    fig = go.Figure(
        go.Heatmap(
            x=dates,
            y=[metric_label],
            z=[values],
            customdata=[hover],
            hovertemplate="%{customdata}<extra></extra>",
            colorscale="Viridis",
            connectgaps=False,
            colorbar=dict(title=metric_label),
        )
    )

    for month in pd.date_range(
        f"{year}-01-01",
        f"{year}-12-01",
        freq="MS",
    ):
        fig.add_vline(
            x=month,
            line_width=0.7,
            line_color="rgba(0,0,0,0.25)",
        )

    fig.update_layout(
        title=f"{year} — {metric_label} Calendar",
        height=190,
        margin=dict(l=30, r=30, t=60, b=50),
        xaxis=dict(
            title="Date",
            tickformat="%b",
            dtick="M1",
            showgrid=False,
        ),
        yaxis=dict(
            title="",
            showticklabels=False,
        ),
    )

    return fig


def build_all_year_calendar_chart(data, metric_column, metric_label):
    """All-years year × day-of-year calendar."""
    matrix = create_all_year_calendar(data, metric_column)
    z = matrix.to_numpy(dtype=float)

    if metric_column in {
        "anomaly_z",
        "modis_z",
        "viirs_z",
    }:
        finite = z[np.isfinite(z)]
        limit = (
            float(np.nanpercentile(np.abs(finite), 99))
            if finite.size
            else 1.0
        )
        limit = max(limit, 1.0)
        colorscale = "RdBu_r"
        zmin, zmax = -limit, limit
    else:
        finite = z[np.isfinite(z)]
        zmax = (
            float(np.nanpercentile(finite, 99))
            if finite.size
            else 1.0
        )
        zmax = max(zmax, 1.0)
        colorscale = "Viridis"
        zmin = None

    fig = go.Figure(
        go.Heatmap(
            x=list(range(1, 367)),
            y=matrix.index.tolist(),
            z=z,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            connectgaps=False,
            hovertemplate=(
                "Year: %{y}<br>"
                "Day of year: %{x}<br>"
                f"{metric_label}: "
                "%{z:.3f}<extra></extra>"
            ),
            colorbar=dict(title=metric_label),
        )
    )

    fig.update_layout(
        title=f"All Years — {metric_label} Calendar",
        height=max(430, 120 + 28 * len(matrix.index)),
        margin=dict(l=60, r=30, t=60, b=55),
        xaxis=dict(
            title="Day of year",
            tickmode="array",
            tickvals=[
                1, 32, 60, 91, 121, 152,
                182, 213, 244, 274, 305, 335,
            ],
            ticktext=[
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            ],
            showgrid=False,
            range=[1, 366],
        ),
        yaxis=dict(
            title="Year",
            autorange="reversed",
            dtick=1,
        ),
    )

    return fig


def build_sector_matrix_chart(data, metric_column, metric_label, scope_label):
    """M3/M5 5×5 sector geometry."""
    values = data.groupby("sector")[metric_column].mean()

    matrix = np.full((GRID_SIZE, GRID_SIZE), np.nan)
    labels = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)

    for number in range(1, GRID_SIZE * GRID_SIZE + 1):
        sector = f"C{number:02d}"

        original_row = (number - 1) // GRID_SIZE
        column = (number - 1) % GRID_SIZE
        display_row = GRID_SIZE - 1 - original_row

        matrix[display_row, column] = values.get(sector, np.nan)
        labels[display_row, column] = sector

    text = np.empty(matrix.shape, dtype=object)

    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            sector = labels[row, column]
            value = matrix[row, column]
            text[row, column] = (
                f"{sector}<br>NA"
                if pd.isna(value)
                else f"{sector}<br>{value:.2f}"
            )

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=["Column 1", "Column 2", "Column 3", "Column 4", "Column 5"],
            y=["Row 5", "Row 4", "Row 3", "Row 2", "Row 1"],
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{text}<extra></extra>",
            colorscale="Viridis",
            colorbar=dict(title=metric_label),
            zsmooth=False,
        )
    )

    fig.update_layout(
        title=f"{scope_label} — {metric_label} by Sector",
        height=500,
        xaxis_title="Longitude sector",
        yaxis_title="Latitude sector",
        yaxis=dict(autorange="reversed"),
    )

    return fig


def build_sensor_comparison(data, scope_label):
    """MODIS, VIIRS and harmonized activity."""
    daily = create_daily_data(data)

    fig = go.Figure()

    for column, name, opacity, width in [
        ("mean_modis_z", "MODIS Z", 0.55, 1),
        ("mean_viirs_z", "VIIRS Z", 0.55, 1),
        ("mean_harmonized_activity", "Harmonized Activity", 0.45, 1),
        (
            "harmonized_30d",
            f"Harmonized Activity — {ROLLING_WINDOW_DAYS}-day mean",
            1.0,
            3,
        ),
    ]:
        fig.add_trace(
            go.Scatter(
                x=daily["acq_date"],
                y=daily[column],
                mode="lines",
                name=name,
                opacity=opacity,
                line=dict(width=width),
            )
        )

    fig.update_layout(
        title=f"{scope_label} — MODIS, VIIRS and Harmonized Activity",
        xaxis_title="Date",
        yaxis_title="Z / Harmonized Activity",
        hovermode="x unified",
        height=500,
    )

    return fig


def build_discrepancy_chart(data, scope_label):
    """MODIS–VIIRS sensor discrepancy."""
    daily = create_daily_data(data)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=daily["acq_date"],
            y=daily["mean_sensor_discrepancy"],
            mode="lines",
            name="Daily mean discrepancy",
            opacity=0.55,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=daily["acq_date"],
            y=daily["discrepancy_30d"],
            mode="lines",
            name=f"{ROLLING_WINDOW_DAYS}-day mean discrepancy",
            line=dict(width=3),
        )
    )

    fig.update_layout(
        title=f"{scope_label} — MODIS–VIIRS Sensor Discrepancy",
        xaxis_title="Date",
        yaxis_title="|MODIS Z − VIIRS Z|",
        hovermode="x unified",
        height=420,
    )

    return fig


def build_critical_timeline(periods, scope_label):
    """M4 critical-period peak timeline."""
    if periods.empty:
        return None, periods

    work = add_period_severity(periods)

    if "peak_date" in work.columns:
        work["plot_date"] = work["peak_date"].fillna(work["start_date"])
    else:
        work["plot_date"] = work["start_date"]

    fig = go.Figure()

    for severity, color, size, name in [
        (
            "Peak anomaly (2 ≤ Z < 3)",
            "orange",
            9,
            "Peak anomaly: 2 ≤ Z < 3",
        ),
        (
            "Extreme peak (Z ≥ 3)",
            "red",
            11,
            "Extreme peak: Z ≥ 3",
        ),
    ]:
        subset = work[work["severity"] == severity]

        if subset.empty:
            continue

        leading = (
            subset["leading_component"]
            if "leading_component" in subset.columns
            else pd.Series("—", index=subset.index)
        )

        customdata = [
            [
                start,
                end,
                float(duration),
                float(peak_z),
                lead,
            ]
            for start, end, duration, peak_z, lead in zip(
                subset["start_date"].dt.strftime("%Y-%m-%d"),
                subset["end_date"].dt.strftime("%Y-%m-%d"),
                subset["duration_days"],
                subset["peak_anomaly_z"],
                leading,
            )
        ]

        fig.add_trace(
            go.Scatter(
                x=subset["plot_date"],
                y=subset["sector"],
                mode="markers",
                name=name,
                marker=dict(
                    color=color,
                    size=size,
                    line=dict(color="white", width=1),
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Peak: %{x|%d %b %Y}<br>"
                    "Start: %{customdata[0]}<br>"
                    "End: %{customdata[1]}<br>"
                    "Duration: %{customdata[2]} days<br>"
                    "Peak Z: %{customdata[3]:.2f}<br>"
                    "Leading component: %{customdata[4]}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"{scope_label} — Milestone 4 Anomaly Peaks",
        xaxis_title="Peak date",
        yaxis_title="Sector",
        hovermode="closest",
        height=max(450, 250 + 20 * work["sector"].nunique()),
        legend_title="Severity",
        xaxis=dict(
            tickformat="%b %Y",
            dtick="M6" if scope_label == "All Years" else "M1",
        ),
    )

    return fig, work


def build_critical_periods_by_year_chart(periods):
    """M4 critical-period count by peak year and severity."""
    work = add_period_severity(periods)

    if "peak_date" in work.columns:
        work["peak_year"] = work["peak_date"].dt.year
    else:
        work["peak_year"] = work["start_date"].dt.year

    counts = (
        work.groupby(["peak_year", "severity"])
        .size()
        .reset_index(name="critical_periods")
    )

    fig = px.bar(
        counts,
        x="peak_year",
        y="critical_periods",
        color="severity",
        barmode="stack",
        title="Critical Periods by Peak Year",
        labels={
            "peak_year": "Peak year",
            "critical_periods": "Critical periods",
            "severity": "Severity",
        },
    )

    fig.update_layout(
        height=420,
        hovermode="x unified",
        xaxis=dict(dtick=1),
    )

    return fig


# ============================================================
# OUTPUT EXPLORER
# ============================================================

OUTPUT_ARTIFACTS = [
    ("critical_periods", "Critical periods", "M4 event records"),
    (
        "harmonized_daily_sector",
        "Harmonized daily sector",
        "Daily sector-level harmonized science output",
    ),
    (
        "harmonized_science_2026",
        "Harmonized science — 2026",
        "Latest-year harmonized science output",
    ),
    (
        "harmonized_science_all_years",
        "Harmonized science — all years",
        "Complete M3 science dataset",
    ),
    (
        "m5_daily_aggregate",
        "M5 daily aggregate",
        "Daily aggregate used by M5 visualizations",
    ),
    (
        "m5_sector_summary",
        "M5 sector summary",
        "Sector-level M5 summary",
    ),
    (
        "milestone_1_summary",
        "Milestone 1 summary",
        "M1 acquisition and preprocessing summary",
    ),
    (
        "milestone_2_validation",
        "Milestone 2 validation",
        "M2 validation output",
    ),
    (
        "milestone_3_summary",
        "Milestone 3 summary",
        "M3 harmonization summary",
    ),
    (
        "milestone_4_summary",
        "Milestone 4 summary",
        "M4 critical-period summary",
    ),
    (
        "milestone_5_summary",
        "Milestone 5 summary",
        "M5 visualization summary",
    ),
]


def locate_output_artifact(stem):
    """Locate an output artifact without assuming its extension."""
    for extension in [
        ".csv",
        ".txt",
        ".md",
        ".json",
        ".parquet",
    ]:
        candidate = OUTPUT_DIR / f"{stem}{extension}"
        if candidate.exists():
            return candidate

    matches = sorted(OUTPUT_DIR.glob(f"{stem}.*"))
    return matches[0] if matches else None


@st.cache_data
def load_output_csv(path_string):
    return pd.read_csv(path_string)


@st.cache_data
def load_output_text(path_string):
    return Path(path_string).read_text(
        encoding="utf-8",
        errors="replace",
    )


def render_output_explorer(data, periods):
    """
    Optional inspection/download section for the existing
    M1–M5 artifacts. It does not recalculate science.
    """
    st.header("Project Output Files")
    st.caption(
        "Inspect or download the precomputed M1–M5 artifacts."
    )

    for stem, label, description in OUTPUT_ARTIFACTS:
        path = locate_output_artifact(stem)

        with st.expander(label):
            st.caption(description)

            if path is None:
                st.info("File not found in output/.")
                continue

            st.caption(f"Path: output/{path.name}")

            if path.suffix.lower() == ".csv":
                if stem == "harmonized_science_all_years":
                    preview = data.head(250).copy()
                elif stem == "harmonized_science_2026":
                    preview = data[data["year"] == 2026].head(250).copy()
                elif stem == "critical_periods":
                    preview = periods.head(250).copy()
                else:
                    preview = load_output_csv(str(path)).head(250)

                st.dataframe(
                    preview,
                    use_container_width=True,
                    hide_index=True,
                )

            elif path.suffix.lower() in {".txt", ".md", ".json"}:
                content = load_output_text(str(path))
                st.code(content[:30000], language="text")
                if len(content) > 30000:
                    st.caption("Preview truncated to 30,000 characters.")

            else:
                st.info("Preview is not available for this file type.")

            try:
                st.download_button(
                    label=f"Download {path.name}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=(
                        "text/csv"
                        if path.suffix.lower() == ".csv"
                        else "text/plain"
                    ),
                    key=f"download_{stem}",
                )
            except OSError:
                st.warning("The file could not be read for download.")


# ============================================================
# APPLICATION LOAD
# ============================================================

try:
    data, data_source = load_science_data()
    critical_periods = load_critical_periods()
except Exception as error:
    st.error(str(error))
    st.stop()


# ============================================================
# SIDEBAR — SINGLE SOURCE OF UI STATE
# ============================================================

st.sidebar.title("🔥 Fire Harmonizer")
st.sidebar.caption("Cross-sensor burning activity analysis")
st.sidebar.divider()

available_years = sorted(data["year"].unique())

analysis_scope = st.sidebar.radio(
    "Analysis scope",
    ["Selected year", "All years"],
    index=0,
)

all_years_view = analysis_scope == "All years"

if all_years_view:
    selected_year = None
    scope_label = "All Years"
    st.sidebar.caption(
        "Using the complete available M3 science record."
    )
else:
    selected_year = st.sidebar.selectbox(
        "Analysis year",
        available_years,
        index=len(available_years) - 1,
    )
    scope_label = str(selected_year)

analysis_data = filter_scope(
    data,
    all_years_view,
    selected_year,
)

sector_options = ["All sectors"] + sorted(
    analysis_data["sector"].unique(),
    key=lambda value: int(value[1:]),
)

selected_sector = st.sidebar.selectbox(
    "Sector",
    sector_options,
)

selected_metric = st.sidebar.selectbox(
    "Calendar metric",
    list(METRIC_OPTIONS.keys()),
    format_func=lambda key: METRIC_OPTIONS[key],
)

st.sidebar.divider()
st.sidebar.caption(f"Science source: {data_source}")
st.sidebar.caption(
    f"Coverage: {data['acq_date'].min():%d %b %Y}"
    f" → {data['acq_date'].max():%d %b %Y}"
)


# ============================================================
# HEADER
# ============================================================

st.title("Fire Harmonizer")
st.subheader("A cross-sensor view of Earth's burning activity")
st.caption(f"Analysis scope: {scope_label}")
st.caption("MODIS + VIIRS → common temporal representation")

st.markdown(
    """
    <div class="section-note">
    Precomputed NASA FIRMS-derived observations are transformed
    into a common scientific representation using the M3
    harmonization framework.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GLOBAL KPIs
# ============================================================

mean_h = float(analysis_data["harmonized_activity"].mean())
max_anomaly = float(analysis_data["anomaly_z"].max())

active_sectors = int(
    analysis_data.loc[
        analysis_data["total_frp"] > 0,
        "sector",
    ].nunique()
)

scope_periods = filter_period_scope(
    critical_periods,
    all_years_view,
    selected_year,
)

if selected_sector != "All sectors":
    scope_periods = scope_periods[
        scope_periods["sector"] == selected_sector
    ].copy()

k1, k2, k3, k4 = st.columns(4)

k1.metric("Mean Harmonized Activity", f"{mean_h:.2f}")
k2.metric("Maximum Anomaly", f"{max_anomaly:.2f}σ")
k3.metric("Active Sectors", active_sectors)
k4.metric("Critical Periods", len(scope_periods))


# ============================================================
# MAIN TABS
# ============================================================

tab_overview, tab_sector, tab_critical, tab_method = st.tabs(
    [
        "Overview",
        "Sector Analysis",
        "Critical Periods",
        "Methodology & Data",
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab_overview:
    if all_years_view:
        st.header("All-Years Scientific Overview")
        st.caption(
            "All-years views are shown here once; the sidebar scope "
            "controls the entire application."
        )

        st.subheader("Harmonized Activity Calendar")
        st.plotly_chart(
            build_all_year_calendar_chart(
                data,
                "harmonized_activity",
                "Harmonized Activity",
            ),
            use_container_width=True,
            key="overview_all_harmonized_calendar",
        )

        st.subheader("Anomaly Calendar")
        st.plotly_chart(
            build_all_year_calendar_chart(
                data,
                "anomaly_z",
                "Anomaly Z",
            ),
            use_container_width=True,
            key="overview_all_anomaly_calendar",
        )

        st.subheader("AOI Sector Matrices")

        left, right = st.columns(2)

        with left:
            st.plotly_chart(
                build_sector_matrix_chart(
                    data,
                    "harmonized_activity",
                    "Mean Harmonized Activity",
                    "All Years",
                ),
                use_container_width=True,
                key="overview_all_harmonized_matrix",
            )

        with right:
            st.plotly_chart(
                build_sector_matrix_chart(
                    data,
                    "anomaly_z",
                    "Mean Anomaly Z",
                    "All Years",
                ),
                use_container_width=True,
                key="overview_all_anomaly_matrix",
            )

        st.subheader("MODIS vs VIIRS")
        st.plotly_chart(
            build_sensor_comparison(data, "All Years"),
            use_container_width=True,
            key="overview_all_sensor",
        )

        st.subheader("Sensor Discrepancy")
        st.plotly_chart(
            build_discrepancy_chart(data, "All Years"),
            use_container_width=True,
            key="overview_all_discrepancy",
        )

    else:
        st.header("Burning Activity Overview")
        st.caption(
            "Daily values are means across observed sectors. "
            "Missing observation dates remain blank."
        )

        st.plotly_chart(
            build_calendar_chart(
                create_year_calendar(
                    analysis_data,
                    selected_year,
                    selected_metric,
                ),
                selected_year,
                METRIC_OPTIONS[selected_metric],
            ),
            use_container_width=True,
            key="overview_selected_calendar",
        )

        st.subheader("AOI Sector Matrices")

        left, right = st.columns(2)

        with left:
            st.plotly_chart(
                build_sector_matrix_chart(
                    analysis_data,
                    "harmonized_activity",
                    "Mean Harmonized Activity",
                    scope_label,
                ),
                use_container_width=True,
                key="overview_selected_harmonized_matrix",
            )

        with right:
            st.plotly_chart(
                build_sector_matrix_chart(
                    analysis_data,
                    "anomaly_z",
                    "Mean Anomaly Z",
                    scope_label,
                ),
                use_container_width=True,
                key="overview_selected_anomaly_matrix",
            )

        st.subheader("MODIS vs VIIRS")
        st.plotly_chart(
            build_sensor_comparison(
                analysis_data,
                scope_label,
            ),
            use_container_width=True,
            key="overview_selected_sensor",
        )

        st.subheader("Sensor Discrepancy")
        st.plotly_chart(
            build_discrepancy_chart(
                analysis_data,
                scope_label,
            ),
            use_container_width=True,
            key="overview_selected_discrepancy",
        )

    st.subheader("Scope Summary")

    summary = st.columns(5)
    summary[0].metric(
        "Years",
        f"{data['year'].min()} → {data['year'].max()}",
    )
    summary[1].metric("Science Rows", f"{len(analysis_data):,}")
    summary[2].metric(
        "Unique Dates",
        f"{analysis_data['acq_date'].nunique():,}",
    )
    summary[3].metric(
        "Sectors",
        analysis_data["sector"].nunique(),
    )
    summary[4].metric(
        "Critical Periods",
        len(scope_periods),
    )


# ============================================================
# TAB 2 — SECTOR ANALYSIS
# ============================================================

with tab_sector:
    st.header("Sector Analysis")

    if selected_sector == "All sectors":
        st.caption(
            f"Sector summary for {scope_label.lower()}."
        )

        sector_summary = (
            analysis_data.groupby("sector", as_index=False)
            .agg(
                observations=("sector", "size"),
                mean_harmonized_activity=(
                    "harmonized_activity",
                    "mean",
                ),
                mean_anomaly_z=("anomaly_z", "mean"),
                max_anomaly_z=("anomaly_z", "max"),
                mean_modis_z=("modis_z", "mean"),
                mean_viirs_z=("viirs_z", "mean"),
                mean_sensor_discrepancy=(
                    "sensor_discrepancy",
                    "mean",
                ),
                total_frp=("total_frp", "sum"),
            )
            .sort_values(
                "mean_harmonized_activity",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        st.dataframe(
            sector_summary,
            use_container_width=True,
            hide_index=True,
        )

    else:
        sector_data = analysis_data[
            analysis_data["sector"] == selected_sector
        ].copy()

        st.subheader(f"{selected_sector} — {scope_label}")

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "Mean Harmonized Activity",
            f"{sector_data['harmonized_activity'].mean():.2f}",
        )
        s2.metric(
            "Mean Anomaly",
            f"{sector_data['anomaly_z'].mean():.2f}σ",
        )
        s3.metric(
            "Maximum Anomaly",
            f"{sector_data['anomaly_z'].max():.2f}σ",
        )
        s4.metric(
            "Total FRP",
            f"{sector_data['total_frp'].sum():.1f} MW",
        )

        st.plotly_chart(
            build_sensor_comparison(
                sector_data,
                f"{selected_sector} — {scope_label}",
            ),
            use_container_width=True,
            key="sector_sensor_chart",
        )

        st.subheader("Sector Observations")

        st.dataframe(
            sector_data[
                [
                    "acq_date",
                    "modis_z",
                    "viirs_z",
                    "harmonized_activity",
                    "anomaly_z",
                    "sensor_discrepancy",
                    "total_frp",
                ]
            ]
            .sort_values("acq_date")
            .reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# TAB 3 — CRITICAL PERIODS
# ============================================================

with tab_critical:
    st.header("Critical Periods")
    st.caption(
        "Milestone 4 periods are shown by their peak anomaly date. "
        "Severity is represented by the single peak marker."
    )

    if scope_periods.empty:
        st.info(
            "No Milestone 4 critical periods match the current scope "
            "and sector selection."
        )
    else:
        timeline_chart, timeline_table = build_critical_timeline(
            scope_periods,
            scope_label,
        )

        if timeline_chart is not None:
            st.plotly_chart(
                timeline_chart,
                use_container_width=True,
                key="critical_timeline",
            )

        timeline_table = add_period_severity(
            timeline_table
        )

        extreme_count = int(
            (
                timeline_table["severity"]
                == "Extreme peak (Z ≥ 3)"
            ).sum()
        )
        anomaly_count = int(
            (
                timeline_table["severity"]
                == "Peak anomaly (2 ≤ Z < 3)"
            ).sum()
        )

        p1, p2, p3 = st.columns(3)
        p1.metric("Critical Periods", len(timeline_table))
        p2.metric("Extreme Peaks", extreme_count)
        p3.metric("2 ≤ Z < 3 Peaks", anomaly_count)

        if all_years_view:
            st.plotly_chart(
                build_critical_periods_by_year_chart(
                    timeline_table
                ),
                use_container_width=True,
                key="critical_by_year",
            )

        st.subheader("Critical-Period Records")

        display_columns = [
            "sector",
            "start_date",
            "end_date",
            "duration_days",
            "peak_date",
            "peak_anomaly_z",
            "severity",
        ]

        optional_columns = [
            "peak_frp",
            "extreme_days",
            "leading_component",
            "mean_modis_z",
            "mean_viirs_z",
            "mean_sensor_discrepancy",
        ]

        display_columns += [
            column
            for column in optional_columns
            if column in timeline_table.columns
        ]

        st.dataframe(
            timeline_table[display_columns],
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# TAB 4 — METHODOLOGY & DATA
# ============================================================

with tab_method:
    st.header("Methodology")

    st.markdown(
        """
        ### 1. Sensor normalization

        MODIS and VIIRS are standardized independently
        against their historical baselines.

        ### 2. Harmonized activity

        The M3 metric is:

        **H = 0.5 × ZMODIS + 0.5 × ZVIIRS**

        ### 3. Sensor discrepancy

        Sensor disagreement is retained independently as:

        **D = |ZMODIS − ZVIIRS|**

        It is not multiplied into harmonized activity.

        ### 4. Historical anomaly

        Harmonized activity is evaluated against the
        historical sector/season baseline.

        ### 5. Critical periods

        M4 defines a critical period as:

        **anomaly Z ≥ 2 for at least 3 consecutive calendar days**

        Extreme individual days are identified by:

        **anomaly Z ≥ 3**

        ### 6. FRP

        Fire Radiative Power remains a separate physical metric.
        It is not multiplied directly into the primary harmonized
        activity score.

        ### 7. Sector structure

        The AOI uses the M3/M5 5×5 sector grid:

        **C01–C25**

        with the established orientation retained.

        ### 8. Missing data

        Missing observation dates are not interpreted as zero
        fire activity. They remain missing in the application.
        """
    )

    st.header("Data Coverage")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Years",
            f"{data['year'].min()} → {data['year'].max()}",
        )
        st.metric("Science Rows", f"{len(data):,}")
        st.metric(
            "Unique Dates",
            f"{data['acq_date'].nunique():,}",
        )

    with c2:
        st.metric("Sectors", data["sector"].nunique())
        st.metric(
            "First Observation",
            f"{data['acq_date'].min():%d %b %Y}",
        )
        st.metric(
            "Last Observation",
            f"{data['acq_date'].max():%d %b %Y}",
        )

    st.header("Milestone 4 Integration")

    if critical_periods.empty:
        st.info("critical_periods.csv is not present.")
    else:
        m4_1, m4_2, m4_3 = st.columns(3)

        m4_1.metric(
            "All-years M4 Periods",
            len(critical_periods),
        )
        m4_2.metric(
            "M4 Sectors",
            critical_periods["sector"].nunique(),
        )
        m4_3.metric(
            "Extreme Peaks",
            int(
                (
                    critical_periods["peak_anomaly_z"] >= 3.0
                ).sum()
            ),
        )

    st.header("Data Download")

    download_df = analysis_data.copy()

    if selected_sector != "All sectors":
        download_df = download_df[
            download_df["sector"] == selected_sector
        ].copy()

    st.download_button(
        label="Download analyzed data",
        data=download_df.to_csv(index=False).encode("utf-8"),
        file_name=(
            f"fire_harmonizer_"
            f"{'all_years' if all_years_view else selected_year}"
            f"{'_' + selected_sector if selected_sector != 'All sectors' else ''}"
            ".csv"
        ),
        mime="text/csv",
    )

    render_output_explorer(
        data,
        critical_periods,
    )

    st.header("Interpretation Notes")

    st.markdown(
        """
        The application presents standardized and harmonized
        fire-activity indicators derived from active-fire
        observations.

        These values should not be interpreted as a direct
        measurement of wildfire severity, burned area, or
        total fire energy.

        Sensor discrepancy is shown separately because MODIS
        and VIIRS can respond differently to the same burning
        conditions.

        FRP is retained separately from the harmonized
        activity score.
        """
    )


# ============================================================
# DAY INSPECTOR
# ============================================================

st.divider()
st.header(f"Day Inspector — {scope_label}")

available_dates = sorted(
    analysis_data["acq_date"].dt.date.unique()
)

if available_dates:
    selected_date = st.selectbox(
        "Select a date",
        available_dates,
        index=len(available_dates) - 1,
        key="day_inspector_date",
    )

    selected = analysis_data[
        analysis_data["acq_date"].dt.date == selected_date
    ].copy()

    if selected_sector != "All sectors":
        selected = selected[
            selected["sector"] == selected_sector
        ].copy()

    if selected.empty:
        st.info("No observations match the selected date and sector.")
    else:
        d1, d2, d3 = st.columns(3)
        d1.metric(
            "Harmonized",
            f"{selected['harmonized_activity'].mean():.2f}",
        )
        d2.metric(
            "Anomaly",
            f"{selected['anomaly_z'].mean():.2f}σ",
        )
        d3.metric(
            "FRP",
            f"{selected['total_frp'].sum():.1f} MW",
        )

        d4, d5, d6 = st.columns(3)
        d4.metric(
            "MODIS",
            f"{selected['modis_z'].mean():.2f}σ",
        )
        d5.metric(
            "VIIRS",
            f"{selected['viirs_z'].mean():.2f}σ",
        )
        d6.metric(
            "Sensor Discrepancy",
            f"{selected['sensor_discrepancy'].mean():.2f}",
        )

        st.subheader(
            f"Sector observations — {selected_date:%d %b %Y}"
        )

        st.dataframe(
            selected[
                [
                    "sector",
                    "modis_z",
                    "viirs_z",
                    "harmonized_activity",
                    "anomaly_z",
                    "sensor_discrepancy",
                    "total_frp",
                ]
            ]
            .sort_values(
                "anomaly_z",
                ascending=False,
            )
            .reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Fire Harmonizer — Milestone 6 | Final authoritative interactive "
    "application built from precomputed M1–M5 outputs."
)
