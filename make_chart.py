"""
FIRE HARMONIZER - MILESTONE 5
================================

TEMPORAL VISUALIZATION & SCIENTIFIC CHARTS — ALL YEARS

Purpose:

    Convert the multi-year harmonized science results into
    clear temporal and sector-based visualizations.

Inputs:

    Preferred:
        output/harmonized_science_all_years.csv

    Fallback:
        output/harmonized_science_YYYY.csv

    Milestone 4:
        output/critical_periods.csv

Outputs:

    output/charts/harmonized_calendar_all_years.png
    output/charts/anomaly_calendar_all_years.png
    output/charts/sector_matrix_harmonized_all_years.png
    output/charts/sector_matrix_anomaly_all_years.png
    output/charts/sensor_comparison_all_years.png
    output/charts/sensor_discrepancy_all_years.png
    output/charts/critical_period_timeline_all_years.png

    output/m5_daily_aggregate.csv
    output/m5_sector_summary.csv
    output/milestone_5_summary.txt

Important:

    - No NASA downloading.
    - All available years are used by default.
    - M4 critical periods are read from critical_periods.csv.
    - M4 critical_periods.csv is expected to represent the same
      all-years science dataset.
    - Cross-year critical periods are preserved.
    - No maps or GIS are used.
    - Missing days are NOT converted to zero.
    - FRP remains a separate metric.
    - Harmonized activity remains the M3 metric.
    - The 5×5 sector structure remains C01-C25.
"""


from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output"

CHART_DIR = OUTPUT_DIR / "charts"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHART_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

GRID_SIZE = 5

ROLLING_WINDOW_DAYS = 30

EXPECTED_SECTORS = [
    f"C{i:02d}"
    for i in range(
        1,
        GRID_SIZE * GRID_SIZE + 1
    )
]


# ============================================================
# REQUIRED SCIENCE COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "acq_date",
    "sector",
    "anomaly_z",
    "modis_z",
    "viirs_z",
    "harmonized_activity",
    "total_frp",
    "sensor_discrepancy",
]


# ============================================================
# M4 CORE COLUMNS
# ============================================================

M4_REQUIRED_COLUMNS = [
    "sector",
    "start_date",
    "end_date",
    "duration_days",
    "peak_anomaly_z",
]

M4_OPTIONAL_COLUMNS = [
    "peak_date",
    "mean_anomaly_z",
    "mean_harmonized_activity",
    "peak_frp",
    "mean_frp",
    "mean_sensor_discrepancy",
    "mean_modis_z",
    "mean_viirs_z",
    "leading_component",
    "extreme_days",
]


# ============================================================
# OUTPUT FILES
# ============================================================

DAILY_AGGREGATE_FILE = (
    OUTPUT_DIR
    / "m5_daily_aggregate.csv"
)

SECTOR_SUMMARY_FILE = (
    OUTPUT_DIR
    / "m5_sector_summary.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "milestone_5_summary.txt"
)

M4_CRITICAL_PERIODS_FILE = (
    OUTPUT_DIR
    / "critical_periods.csv"
)


# ============================================================
# LOAD SCIENCE DATA
# ============================================================

def load_science_data(
    start_year=None,
    end_year=None,
):
    """
    Load the multi-year harmonized science dataset.

    Preferred source:
        harmonized_science_all_years.csv

    Fallback:
        harmonized_science_YYYY.csv
    """

    all_years_file = (
        OUTPUT_DIR
        / "harmonized_science_all_years.csv"
    )

    frames = []

    # ========================================================
    # OPTION 1 — ALL-YEARS FILE
    # ========================================================

    if all_years_file.exists():

        print()
        print(
            "Using all-years harmonized science file:"
        )

        print(
            f"  {all_years_file}"
        )

        try:
            df = pd.read_csv(
                all_years_file
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not read "
                "harmonized_science_all_years.csv: "
                f"{exc}"
            )

        if df.empty:
            raise RuntimeError(
                "harmonized_science_all_years.csv is empty."
            )

        frames.append(
            df
        )

    # ========================================================
    # OPTION 2 — YEARLY FILES
    # ========================================================

    else:

        print()
        print(
            "All-years file not found."
        )

        print(
            "Discovering yearly science files..."
        )

        prefix = "harmonized_science_"

        paths = sorted(
            OUTPUT_DIR.glob(
                f"{prefix}*.csv"
            )
        )

        selected = []

        for path in paths:

            stem = path.stem

            if not stem.startswith(prefix):
                continue

            year_text = stem[
                len(prefix):
            ]

            if not year_text.isdigit():
                continue

            year = int(
                year_text
            )

            if (
                start_year is not None
                and year < start_year
            ):
                continue

            if (
                end_year is not None
                and year > end_year
            ):
                continue

            selected.append(
                (
                    year,
                    path
                )
            )

        if not selected:
            raise FileNotFoundError(
                "No yearly harmonized science files were found."
            )

        selected.sort(
            key=lambda x: x[0]
        )

        print()
        print(
            "Yearly science files selected:"
        )

        for year, path in selected:

            print(
                f"  {year}: {path.name}"
            )

        for year, path in selected:

            print()
            print(
                f"Loading {path.name} ..."
            )

            try:
                year_df = pd.read_csv(
                    path
                )
            except Exception as exc:

                print(
                    f"  ⚠ Could not read file: {exc}"
                )

                continue

            if year_df.empty:

                print(
                    "  ⚠ Empty file — skipped."
                )

                continue

            year_df["source_year"] = year

            frames.append(
                year_df
            )

    # ========================================================
    # CHECK
    # ========================================================

    if not frames:
        raise RuntimeError(
            "No usable harmonized science data was found."
        )

    # ========================================================
    # COMBINE
    # ========================================================

    df = pd.concat(
        frames,
        ignore_index=True
    )

    # ========================================================
    # COLUMN CHECK
    # ========================================================

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Science data is missing required columns:\n"
            + ", ".join(missing)
        )

    # ========================================================
    # KEEP REQUIRED SCIENCE COLUMNS
    # ========================================================

    df = df[
        REQUIRED_COLUMNS
    ].copy()

    # ========================================================
    # DATE
    # ========================================================

    df["acq_date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce"
    )

    df["acq_date"] = (
        df["acq_date"]
        .dt.normalize()
    )

    # ========================================================
    # SECTOR
    # ========================================================

    df["sector"] = (
        df["sector"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ========================================================
    # NUMERICAL COLUMNS
    # ========================================================

    numeric_columns = [
        "anomaly_z",
        "modis_z",
        "viirs_z",
        "harmonized_activity",
        "total_frp",
        "sensor_discrepancy",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # ========================================================
    # VALID ROWS
    # ========================================================

    valid = (
        df["acq_date"].notna()
        &
        df["sector"].notna()
        &
        df["anomaly_z"].notna()
        &
        df["modis_z"].notna()
        &
        df["viirs_z"].notna()
        &
        df["harmonized_activity"].notna()
        &
        df["total_frp"].notna()
        &
        df["sensor_discrepancy"].notna()
    )

    removed = int(
        (~valid).sum()
    )

    if removed > 0:

        print(
            f"⚠ Removed {removed:,} invalid row(s)."
        )

    df = df.loc[
        valid
    ].copy()

    if df.empty:
        raise RuntimeError(
            "No valid science rows remain."
        )

    # ========================================================
    # YEAR
    # ========================================================

    df["year"] = (
        df["acq_date"]
        .dt.year
    )

    # ========================================================
    # DAY OF YEAR
    # ========================================================

    df["day_of_year"] = (
        df["acq_date"]
        .dt.dayofyear
    )

    # ========================================================
    # OPTIONAL YEAR FILTER
    # ========================================================

    if start_year is not None:

        df = df[
            df["year"] >= start_year
        ]

    if end_year is not None:

        df = df[
            df["year"] <= end_year
        ]

    if df.empty:

        raise RuntimeError(
            "No science data remains after year filtering."
        )

    # ========================================================
    # SORT
    # ========================================================

    df = df.sort_values(
        [
            "acq_date",
            "sector",
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # SECTOR VALIDATION
    # ========================================================

    found_sectors = set(
        df["sector"].unique()
    )

    unexpected = found_sectors.difference(
        EXPECTED_SECTORS
    )

    if unexpected:

        raise RuntimeError(
            "Unexpected sector code(s) found: "
            + ", ".join(
                sorted(unexpected)
            )
        )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "M5 SCIENCE DATA"
    )

    print(
        "=" * 70
    )

    print(
        f"Years: "
        f"{df['year'].min()} -> "
        f"{df['year'].max()}"
    )

    print(
        f"Rows: "
        f"{len(df):,}"
    )

    print(
        f"Unique dates: "
        f"{df['acq_date'].nunique():,}"
    )

    print(
        f"Sectors: "
        f"{df['sector'].nunique():,}"
    )

    print(
        f"Date range: "
        f"{df['acq_date'].min().date()}"
        f" -> "
        f"{df['acq_date'].max().date()}"
    )

    print(
        "=" * 70
    )

    return df


# ============================================================
# DAILY BASIN AGGREGATION
# ============================================================

def create_daily_aggregate(df):
    """
    Create one row per observed day.

    Daily values use the mean across observed sectors.

    FRP remains a separate metric.

    30-day values use a calendar-time rolling window,
    rather than simply the previous 30 rows.
    """

    daily = (
        df
        .groupby(
            "acq_date",
            as_index=False
        )
        .agg(
            mean_harmonized_activity=(
                "harmonized_activity",
                "mean"
            ),

            mean_anomaly_z=(
                "anomaly_z",
                "mean"
            ),

            mean_modis_z=(
                "modis_z",
                "mean"
            ),

            mean_viirs_z=(
                "viirs_z",
                "mean"
            ),

            mean_sensor_discrepancy=(
                "sensor_discrepancy",
                "mean"
            ),

            mean_frp=(
                "total_frp",
                "mean"
            ),

            total_frp=(
                "total_frp",
                "sum"
            ),

            sectors_observed=(
                "sector",
                "nunique"
            ),
        )
    )

    # ========================================================
    # SORT
    # ========================================================

    daily = daily.sort_values(
        "acq_date"
    ).reset_index(
        drop=True
    )

    # ========================================================
    # CALENDAR-TIME 30-DAY ROLLING MEANS
    # ========================================================

    daily = daily.set_index(
        "acq_date"
    )

    daily["harmonized_30d"] = (
        daily[
            "mean_harmonized_activity"
        ]
        .rolling(
            f"{ROLLING_WINDOW_DAYS}D",
            min_periods=1
        )
        .mean()
    )

    daily["anomaly_30d"] = (
        daily[
            "mean_anomaly_z"
        ]
        .rolling(
            f"{ROLLING_WINDOW_DAYS}D",
            min_periods=1
        )
        .mean()
    )

    daily["discrepancy_30d"] = (
        daily[
            "mean_sensor_discrepancy"
        ]
        .rolling(
            f"{ROLLING_WINDOW_DAYS}D",
            min_periods=1
        )
        .mean()
    )

    daily = daily.reset_index()

    return daily


# ============================================================
# SECTOR SUMMARY
# ============================================================

def create_sector_summary(df):
    """
    Create all-year statistics for each sector.
    """

    summary = (
        df
        .groupby(
            "sector",
            as_index=False
        )
        .agg(
            observations=(
                "sector",
                "size"
            ),

            days_observed=(
                "acq_date",
                "nunique"
            ),

            mean_harmonized_activity=(
                "harmonized_activity",
                "mean"
            ),

            mean_anomaly_z=(
                "anomaly_z",
                "mean"
            ),

            max_anomaly_z=(
                "anomaly_z",
                "max"
            ),

            mean_modis_z=(
                "modis_z",
                "mean"
            ),

            mean_viirs_z=(
                "viirs_z",
                "mean"
            ),

            mean_sensor_discrepancy=(
                "sensor_discrepancy",
                "mean"
            ),

            mean_frp=(
                "total_frp",
                "mean"
            ),

            total_frp=(
                "total_frp",
                "sum"
            ),
        )
    )

    return summary


# ============================================================
# CALENDAR MONTH POSITIONS
# ============================================================

def month_positions():
    """
    Return approximate month-start positions.

    A non-leap reference year is used.
    """

    reference_year = 2021

    dates = pd.date_range(
        start=f"{reference_year}-01-01",
        end=f"{reference_year}-12-01",
        freq="MS"
    )

    positions = [
        int(date.dayofyear)
        for date in dates
    ]

    labels = [
        date.strftime("%b")
        for date in dates
    ]

    return positions, labels


# ============================================================
# CREATE YEAR × DAY-OF-YEAR MATRIX
# ============================================================

def create_calendar_matrix(
    df,
    value_column,
):
    """
    Create:

        rows    = years
        columns = day of year 1..366

    Missing dates remain NaN.

    They are NOT changed to zero.
    """

    grouped = (
        df
        .groupby(
            [
                "year",
                "day_of_year",
            ],
            as_index=False
        )[value_column]
        .mean()
    )

    matrix = (
        grouped
        .pivot(
            index="year",
            columns="day_of_year",
            values=value_column
        )
    )

    matrix = matrix.reindex(
        columns=range(1, 367)
    )

    matrix = matrix.sort_index()

    return matrix


# ============================================================
# SAVE CALENDAR HEATMAP
# ============================================================

def save_calendar_heatmap(
    df,
    value_column,
    title,
    colorbar_label,
    output_name,
):
    """
    Create a multi-year year × day-of-year heatmap.

    - 99th percentile robust clipping
    - Symmetric scale around zero
    - Missing dates explicitly shown in gray
    - Outliers cannot dominate the scale
    """

    matrix = create_calendar_matrix(
        df,
        value_column
    )

    if matrix.empty:
        return

    values = matrix.values.astype(
        float
    )

    valid_values = values[
        np.isfinite(values)
    ]

    if len(valid_values) == 0:
        return

    # ========================================================
    # ROBUST SCALE
    # ========================================================

    percentile_limit = float(
        np.nanpercentile(
            np.abs(valid_values),
            99
        )
    )

    if (
        not np.isfinite(percentile_limit)
        or percentile_limit <= 0
    ):

        percentile_limit = 1.0

    # ========================================================
    # MASK MISSING VALUES
    # ========================================================

    masked_values = np.ma.masked_invalid(
        values
    )

    # ========================================================
    # COLORMAP
    # ========================================================

    cmap = plt.get_cmap(
        "viridis"
    ).copy()

    cmap.set_bad(
        "#e0e0e0"
    )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    norm = plt.Normalize(
        vmin=-percentile_limit,
        vmax=percentile_limit
    )

    # ========================================================
    # FIGURE
    # ========================================================

    fig_width = 16

    fig_height = max(
        4,
        0.55 * len(matrix)
    )

    fig, ax = plt.subplots(
        figsize=(
            fig_width,
            fig_height
        )
    )

    # ========================================================
    # HEATMAP
    # ========================================================

    image = ax.imshow(
        masked_values,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap=cmap,
        norm=norm
    )

    # ========================================================
    # X AXIS
    # ========================================================

    month_pos, month_labels = (
        month_positions()
    )

    month_ticks = [
        position - 1
        for position in month_pos
    ]

    ax.set_xticks(
        month_ticks
    )

    ax.set_xticklabels(
        month_labels
    )

    ax.set_xlabel(
        "Month"
    )

    # ========================================================
    # Y AXIS
    # ========================================================

    ax.set_yticks(
        range(
            len(matrix.index)
        )
    )

    ax.set_yticklabels(
        [
            str(year)
            for year in matrix.index
        ]
    )

    ax.set_ylabel(
        "Year"
    )

    # ========================================================
    # TITLE
    # ========================================================

    ax.set_title(
        title
        + "\n"
        + "Color scale clipped at the 99th percentile"
    )

    # ========================================================
    # COLORBAR
    # ========================================================

    colorbar = fig.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        colorbar_label
    )

    # ========================================================
    # MISSING DATA BACKGROUND
    # ========================================================

    ax.set_facecolor(
        "#e0e0e0"
    )

    # ========================================================
    # SAVE
    # ========================================================

    plt.tight_layout()

    output_path = (
        CHART_DIR
        / output_name
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"  ✅ {output_path}"
    )


# ============================================================
# CREATE 5 × 5 SECTOR MATRIX
# ============================================================

def create_sector_matrix(
    sector_summary,
    value_column,
):
    """
    Construct the 5×5 scientific sector matrix.

    Milestone 3 numbering:

        C01 C02 C03 C04 C05
        C06 C07 C08 C09 C10
        C11 C12 C13 C14 C15
        C16 C17 C18 C19 C20
        C21 C22 C23 C24 C25

    C21-C25 are displayed at the northern/top side.
    """

    values = np.full(
        (
            GRID_SIZE,
            GRID_SIZE
        ),
        np.nan
    )

    lookup = (
        sector_summary
        .set_index("sector")
    )

    for number in range(
        1,
        GRID_SIZE * GRID_SIZE + 1
    ):

        sector = (
            f"C{number:02d}"
        )

        original_row = (
            number - 1
        ) // GRID_SIZE

        column = (
            number - 1
        ) % GRID_SIZE

        if sector in lookup.index:

            values[
                original_row,
                column
            ] = lookup.loc[
                sector,
                value_column
            ]

    values = np.flipud(
        values
    )

    return values


# ============================================================
# SAVE 5 × 5 SECTOR MATRIX
# ============================================================

def save_sector_matrix(
    sector_summary,
    value_column,
    title,
    colorbar_label,
    output_name,
):
    """
    Save an annotated 5×5 sector matrix.
    """

    matrix = create_sector_matrix(
        sector_summary,
        value_column
    )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    image = ax.imshow(
        matrix,
        interpolation="nearest"
    )

    # ========================================================
    # SECTOR LABELS
    # ========================================================

    labels = []

    for row in range(
        GRID_SIZE
    ):

        row_labels = []

        for column in range(
            GRID_SIZE
        ):

            sector_number = (
                (GRID_SIZE - 1 - row)
                * GRID_SIZE
                + column
                + 1
            )

            sector_label = (
                f"C{sector_number:02d}"
            )

            row_labels.append(
                sector_label
            )

        labels.append(
            row_labels
        )

    # ========================================================
    # WRITE LABEL + VALUE
    # ========================================================

    for row in range(
        GRID_SIZE
    ):

        for column in range(
            GRID_SIZE
        ):

            value = matrix[
                row,
                column
            ]

            sector_label = labels[
                row
            ][
                column
            ]

            if np.isnan(value):

                display_text = (
                    f"{sector_label}\n"
                    "NA"
                )

            else:

                display_text = (
                    f"{sector_label}\n"
                    f"{value:.2f}"
                )

            ax.text(
                column,
                row,
                display_text,
                ha="center",
                va="center"
            )

    # ========================================================
    # AXES
    # ========================================================

    ax.set_xticks(
        range(GRID_SIZE)
    )

    ax.set_xticklabels(
        [
            "Column 1",
            "Column 2",
            "Column 3",
            "Column 4",
            "Column 5",
        ]
    )

    ax.set_yticks(
        range(GRID_SIZE)
    )

    ax.set_yticklabels(
        [
            "Row 5",
            "Row 4",
            "Row 3",
            "Row 2",
            "Row 1",
        ]
    )

    ax.set_xlabel(
        "Longitude sector"
    )

    ax.set_ylabel(
        "Latitude sector"
    )

    ax.set_title(
        title
    )

    # ========================================================
    # COLORBAR
    # ========================================================

    colorbar = fig.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        colorbar_label
    )

    # ========================================================
    # SAVE
    # ========================================================

    plt.tight_layout()

    output_path = (
        CHART_DIR
        / output_name
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"  ✅ {output_path}"
    )


# ============================================================
# SENSOR COMPARISON
# ============================================================

def save_sensor_comparison(
    daily
):
    """
    Compare:

        MODIS Z
        VIIRS Z
        Harmonized Activity

    The 30-day rolling harmonized value is also shown.
    """

    fig, ax = plt.subplots(
        figsize=(16, 6)
    )

    # ========================================================
    # DAILY MODIS
    # ========================================================

    ax.plot(
        daily["acq_date"],
        daily["mean_modis_z"],
        label="MODIS Z",
        alpha=0.45
    )

    # ========================================================
    # DAILY VIIRS
    # ========================================================

    ax.plot(
        daily["acq_date"],
        daily["mean_viirs_z"],
        label="VIIRS Z",
        alpha=0.45
    )

    # ========================================================
    # DAILY HARMONIZED
    # ========================================================

    ax.plot(
        daily["acq_date"],
        daily["mean_harmonized_activity"],
        label="Harmonized Activity",
        alpha=0.35
    )

    # ========================================================
    # 30-DAY HARMONIZED
    # ========================================================

    ax.plot(
        daily["acq_date"],
        daily["harmonized_30d"],
        label="Harmonized Activity — 30-day mean",
        linewidth=2
    )

    # ========================================================
    # TITLE
    # ========================================================

    ax.set_title(
        "MODIS, VIIRS and Harmonized Activity — All Years"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Z / Harmonized Activity"
    )

    # ========================================================
    # X AXIS
    # ========================================================

    ax.xaxis.set_major_locator(
        mdates.YearLocator()
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )

    # ========================================================
    # GRID
    # ========================================================

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend()

    plt.tight_layout()

    output_path = (
        CHART_DIR
        / "sensor_comparison_all_years.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"  ✅ {output_path}"
    )


# ============================================================
# SENSOR DISCREPANCY
# ============================================================

def save_sensor_discrepancy(
    daily
):
    """
    Show MODIS–VIIRS discrepancy through time.

    Discrepancy:

        |MODIS Z - VIIRS Z|
    """

    fig, ax = plt.subplots(
        figsize=(16, 5)
    )

    # ========================================================
    # DAILY
    # ========================================================

    ax.plot(
        daily["acq_date"],
        daily["mean_sensor_discrepancy"],
        label="Daily mean discrepancy",
        alpha=0.45
    )

    # ========================================================
    # ROLLING
    # ========================================================

    ax.plot(
        daily["acq_date"],
        daily["discrepancy_30d"],
        label="30-day mean discrepancy",
        linewidth=2
    )

    # ========================================================
    # TITLE
    # ========================================================

    ax.set_title(
        "MODIS–VIIRS Sensor Discrepancy — All Years"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "|MODIS Z − VIIRS Z|"
    )

    # ========================================================
    # X AXIS
    # ========================================================

    ax.xaxis.set_major_locator(
        mdates.YearLocator()
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )

    # ========================================================
    # GRID
    # ========================================================

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend()

    plt.tight_layout()

    output_path = (
        CHART_DIR
        / "sensor_discrepancy_all_years.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"  ✅ {output_path}"
    )


# ============================================================
# LOAD M4 CRITICAL PERIODS
# ============================================================

def load_critical_periods(
    start_year=None,
    end_year=None,
):
    """
    Load Milestone 4 critical periods.

    M4 is now an all-years continuation of M3, so this file
    should contain critical periods detected from:

        output/harmonized_science_all_years.csv

    Year filtering is performed by DATE OVERLAP rather than
    simply checking start_date.year.

    This preserves cross-year periods such as:

        2018-12-30 -> 2019-01-02
    """

    path = M4_CRITICAL_PERIODS_FILE

    if not path.exists():

        print()
        print(
            "⚠ critical_periods.csv not found."
        )

        return pd.DataFrame()

    print()
    print(
        f"Using Milestone 4 critical-period file:"
    )

    print(
        f"  {path}"
    )

    try:

        periods = pd.read_csv(
            path
        )

    except Exception as exc:

        raise RuntimeError(
            "Could not read critical_periods.csv: "
            f"{exc}"
        )

    if periods.empty:

        print(
            "  ⚠ critical_periods.csv is empty."
        )

        return periods

    # ========================================================
    # REQUIRED COLUMN CHECK
    # ========================================================

    missing = [
        column
        for column in M4_REQUIRED_COLUMNS
        if column not in periods.columns
    ]

    if missing:

        raise RuntimeError(
            "critical_periods.csv is missing columns:\n"
            + ", ".join(missing)
        )

    # ========================================================
    # DATE CONVERSION
    # ========================================================

    periods["start_date"] = pd.to_datetime(
        periods["start_date"],
        errors="coerce"
    )

    periods["end_date"] = pd.to_datetime(
        periods["end_date"],
        errors="coerce"
    )

    if "peak_date" in periods.columns:

        periods["peak_date"] = pd.to_datetime(
            periods["peak_date"],
            errors="coerce"
        )

    # ========================================================
    # NUMERICAL CONVERSION
    # ========================================================

    numerical_columns = [
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
    ]

    for column in numerical_columns:

        if column in periods.columns:

            periods[column] = pd.to_numeric(
                periods[column],
                errors="coerce"
            )

    # ========================================================
    # SECTOR
    # ========================================================

    periods["sector"] = (
        periods["sector"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ========================================================
    # DROP INVALID CORE ROWS
    # ========================================================

    periods = periods.dropna(
        subset=[
            "start_date",
            "end_date",
            "duration_days",
            "peak_anomaly_z",
            "sector",
        ]
    ).copy()

    if periods.empty:

        print(
            "  ⚠ No valid M4 critical periods remain."
        )

        return periods

    # ========================================================
    # SORT
    # ========================================================

    periods = periods.sort_values(
        [
            "start_date",
            "sector",
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # VALIDATE SECTORS
    # ========================================================

    found_sectors = set(
        periods["sector"].unique()
    )

    unexpected = found_sectors.difference(
        EXPECTED_SECTORS
    )

    if unexpected:

        raise RuntimeError(
            "M4 critical_periods.csv contains "
            "unexpected sector code(s): "
            + ", ".join(
                sorted(unexpected)
            )
        )

    # ========================================================
    # VALIDATE DATE ORDER
    # ========================================================

    invalid_order = (
        periods["end_date"]
        < periods["start_date"]
    )

    if invalid_order.any():

        count = int(
            invalid_order.sum()
        )

        raise RuntimeError(
            f"M4 contains {count} period(s) where "
            "end_date occurs before start_date."
        )

    # ========================================================
    # VALIDATE DURATION
    # ========================================================

    calculated_duration = (
        (
            periods["end_date"]
            - periods["start_date"]
        ).dt.days
        + 1
    )

    duration_mismatch = (
        calculated_duration
        != periods["duration_days"]
    )

    if duration_mismatch.any():

        count = int(
            duration_mismatch.sum()
        )

        print(
            f"⚠ {count} M4 period(s) have a duration mismatch."
        )

        periods.loc[
            duration_mismatch,
            "duration_days"
        ] = calculated_duration[
            duration_mismatch
        ]

    # ========================================================
    # VALIDATE PEAK DATE
    # ========================================================

    if "peak_date" in periods.columns:

        invalid_peak = (
            periods["peak_date"].notna()
            &
            (
                (periods["peak_date"] < periods["start_date"])
                |
                (periods["peak_date"] > periods["end_date"])
            )
        )

        if invalid_peak.any():

            count = int(
                invalid_peak.sum()
            )

            raise RuntimeError(
                f"M4 contains {count} period(s) with "
                "peak_date outside the critical-period range."
            )

    # ========================================================
    # M4 SCIENTIFIC THRESHOLD VALIDATION
    # ========================================================

    threshold_violations = (
        periods["peak_anomaly_z"]
        < 2.0
    )

    if threshold_violations.any():

        count = int(
            threshold_violations.sum()
        )

        print(
            f"⚠ {count} M4 period(s) have "
            "peak_anomaly_z < 2."
        )

    # ========================================================
    # DATE-OVERLAP YEAR FILTER
    # ========================================================

    if (
        start_year is not None
        or end_year is not None
    ):

        filter_start = (
            pd.Timestamp(
                f"{start_year}-01-01"
            )
            if start_year is not None
            else periods["start_date"].min()
        )

        filter_end = (
            pd.Timestamp(
                f"{end_year}-12-31"
            )
            if end_year is not None
            else periods["end_date"].max()
        )

        # Keep periods that overlap the requested interval.
        overlap = (
            periods["end_date"] >= filter_start
        ) & (
            periods["start_date"] <= filter_end
        )

        periods = periods.loc[
            overlap
        ].copy()

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "M4 CRITICAL PERIODS"
    )

    print(
        "=" * 70
    )

    if periods.empty:

        print(
            "No critical periods remain after filtering."
        )

        print(
            "=" * 70
        )

        return periods

    print(
        f"Periods: "
        f"{len(periods):,}"
    )

    print(
        f"Period start: "
        f"{periods['start_date'].min().date()}"
    )

    print(
        f"Period end: "
        f"{periods['end_date'].max().date()}"
    )

    print(
        f"Sectors represented: "
        f"{periods['sector'].nunique()}"
    )

    years = sorted(
        set(
            periods["start_date"].dt.year
        )
        |
        set(
            periods["end_date"].dt.year
        )
    )

    print(
        "Years touched: "
        + ", ".join(
            str(year)
            for year in years
        )
    )

    print(
        "=" * 70
    )

    return periods

# ============================================================
# CRITICAL PERIOD TIMELINE — CLEAN ALL-YEARS VERSION
# ============================================================
# ============================================================
# CRITICAL PERIOD TIMELINE — PEAK SEVERITY ONLY
# ============================================================

def save_critical_period_timeline(
    periods
):
    """
    Display Milestone 4 anomaly peaks across all years.

    Visual encoding:

        Orange:
            Peak anomaly, 2 <= Z < 3

        Red:
            Extreme peak, Z >= 3

    No critical-period spans are drawn.
    No blue category is used.
    Each event is represented by exactly one marker.

    Features:

        - Orange and red severity markers only.
        - No per-period text annotations.
        - Adaptive date axis.
        - Supports multiple years.
        - Preserves cross-year events.
        - Year separators for long ranges.
        - Legend matches plotted colors.
    """

    output_path = (
        CHART_DIR
        / "critical_period_timeline_all_years.png"
    )

    # ========================================================
    # NO PERIODS
    # ========================================================

    if periods.empty:

        fig, ax = plt.subplots(
            figsize=(14, 5)
        )

        ax.text(
            0.5,
            0.5,
            "No critical periods detected",
            ha="center",
            va="center",
            transform=ax.transAxes
        )

        ax.set_axis_off()

        ax.set_title(
            "Milestone 4 Anomaly Peaks — All Years"
        )

        plt.tight_layout()

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            f"  ✅ {output_path}"
        )

        return

    # ========================================================
    # COPY + CLEAN
    # ========================================================

    periods = periods.copy()

    periods["start_date"] = pd.to_datetime(
        periods["start_date"],
        errors="coerce"
    )

    periods["end_date"] = pd.to_datetime(
        periods["end_date"],
        errors="coerce"
    )

    periods["peak_anomaly_z"] = pd.to_numeric(
        periods["peak_anomaly_z"],
        errors="coerce"
    )

    if "peak_date" in periods.columns:

        periods["peak_date"] = pd.to_datetime(
            periods["peak_date"],
            errors="coerce"
        )

    periods = periods.dropna(
        subset=[
            "start_date",
            "end_date",
            "sector",
            "peak_anomaly_z",
        ]
    ).copy()

    if periods.empty:
        return

    # ========================================================
    # SECTOR ORDER
    # ========================================================

    sector_order = sorted(
        periods["sector"].unique(),
        key=lambda x: (
            int(x[1:])
            if (
                x.startswith("C")
                and x[1:].isdigit()
            )
            else 999
        )
    )

    sector_to_y = {
        sector: index
        for index, sector in enumerate(
            sector_order
        )
    }

    # ========================================================
    # FIGURE
    # ========================================================

    fig_height = max(
        6,
        len(sector_order) * 0.55
    )

    fig, ax = plt.subplots(
        figsize=(
            16,
            fig_height
        )
    )

    # ========================================================
    # DRAW SEVERITY MARKERS ONLY
    # ========================================================

    for _, row in periods.iterrows():

        sector = row["sector"]

        y = sector_to_y[
            sector
        ]

        # ----------------------------------------------------
        # Peak date
        # ----------------------------------------------------

        if (
            "peak_date" in row.index
            and pd.notna(
                row["peak_date"]
            )
        ):

            peak_date = row[
                "peak_date"
            ]

        else:

            peak_date = row[
                "start_date"
            ]

        # ----------------------------------------------------
        # Peak anomaly
        # ----------------------------------------------------

        peak_anomaly = row[
            "peak_anomaly_z"
        ]

        # ----------------------------------------------------
        # Exclusive severity mapping
        # ----------------------------------------------------

        if peak_anomaly >= 3.0:

            marker_color = "red"
            marker_size = 55

        elif peak_anomaly >= 2.0:

            marker_color = "orange"
            marker_size = 45

        else:

            # M4 should normally never produce this category,
            # because critical periods require anomaly_z >= 2.
            continue

        # ----------------------------------------------------
        # Plot exactly ONE marker
        # ----------------------------------------------------

        ax.scatter(
            peak_date,
            y,
            s=marker_size,
            color=marker_color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3
        )

    # ========================================================
    # LEGEND
    # ========================================================

    from matplotlib.lines import Line2D

    legend_handles = [

        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor="red",
            markeredgecolor="white",
            markersize=8,
            label="Extreme peak: Z ≥ 3"
        ),

        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor="orange",
            markeredgecolor="white",
            markersize=8,
            label="Peak anomaly: 2 ≤ Z < 3"
        ),

    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fontsize=9
    )

    # ========================================================
    # Y AXIS
    # ========================================================

    ax.set_yticks(
        range(
            len(sector_order)
        )
    )

    ax.set_yticklabels(
        sector_order
    )

    ax.set_ylabel(
        "Sector"
    )

    # ========================================================
    # DATE RANGE
    # ========================================================

    min_date = periods[
        "start_date"
    ].min()

    max_date = periods[
        "end_date"
    ].max()

    total_days = (
        max_date - min_date
    ).days

    # ========================================================
    # ADAPTIVE PADDING
    # ========================================================

    if total_days <= 30:

        padding_days = 5

    elif total_days <= 120:

        padding_days = 15

    else:

        padding_days = 30

    axis_start = (
        min_date
        - pd.Timedelta(
            days=padding_days
        )
    )

    axis_end = (
        max_date
        + pd.Timedelta(
            days=padding_days
        )
    )

    ax.set_xlim(
        axis_start,
        axis_end
    )

    # ========================================================
    # YEAR SEPARATORS
    # ========================================================

    if total_days > 365:

        first_year = min_date.year
        last_year = max_date.year

        for year in range(
            first_year + 1,
            last_year + 1
        ):

            january_first = pd.Timestamp(
                f"{year}-01-01"
            )

            if (
                axis_start
                <= january_first
                <= axis_end
            ):

                ax.axvline(
                    january_first,
                    linewidth=0.8,
                    alpha=0.25,
                    zorder=1
                )

    # ========================================================
    # ADAPTIVE DATE LOCATOR
    # ========================================================

    if total_days > 730:

        locator = mdates.YearLocator(
            base=1
        )

        formatter = mdates.DateFormatter(
            "%Y"
        )

    elif total_days > 120:

        locator = mdates.MonthLocator(
            interval=1
        )

        formatter = mdates.DateFormatter(
            "%b\n%Y"
        )

    elif total_days > 30:

        locator = mdates.WeekdayLocator(
            byweekday=mdates.MO,
            interval=1
        )

        formatter = mdates.DateFormatter(
            "%d %b"
        )

    else:

        locator = mdates.DayLocator(
            interval=max(
                1,
                total_days // 8
            )
        )

        formatter = mdates.DateFormatter(
            "%d %b"
        )

    ax.xaxis.set_major_locator(
        locator
    )

    ax.xaxis.set_major_formatter(
        formatter
    )

    # ========================================================
    # X AXIS STYLE
    # ========================================================

    ax.tick_params(
        axis="x",
        which="major",
        labelbottom=True,
        rotation=45,
        labelsize=9
    )

    ax.set_xlabel(
        "Date"
    )

    # ========================================================
    # TITLE
    # ========================================================

    ax.set_title(
        "Milestone 4 Anomaly Peaks — All Years"
    )

    # ========================================================
    # GRID
    # ========================================================

    ax.grid(
        True,
        axis="x",
        alpha=0.25,
        zorder=0
    )

    # ========================================================
    # LAYOUT
    # ========================================================

    plt.tight_layout()

    # ========================================================
    # SAVE
    # ========================================================

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"  ✅ {output_path}"
    )
# ============================================================
# CREATE M5 SUMMARY
# ============================================================

def create_milestone_5_summary(
    df,
    daily,
    sector_summary,
    periods,
    start_year,
    end_year,
):
    """
    Create:

        output/milestone_5_summary.txt
    """

    # ========================================================
    # YEARS
    # ========================================================

    years = sorted(
        df["year"].unique()
    )

    # ========================================================
    # BASIC COUNTS
    # ========================================================

    science_rows = len(
        df
    )

    unique_dates = (
        df["acq_date"]
        .nunique()
    )

    sector_count = (
        df["sector"]
        .nunique()
    )

    # ========================================================
    # CRITICAL PERIODS
    # ========================================================

    if periods is None or periods.empty:

        critical_period_count = 0
        extreme_day_count = 0
        critical_years = []

    else:

        critical_period_count = len(
            periods
        )

        if "extreme_days" in periods.columns:

            extreme_day_count = int(
                pd.to_numeric(
                    periods[
                        "extreme_days"
                    ],
                    errors="coerce"
                )
                .fillna(0)
                .sum()
            )

        else:

            extreme_day_count = 0

        critical_years = sorted(
            set(
                periods[
                    "start_date"
                ]
                .dt.year
                .tolist()
            )
            |
            set(
                periods[
                    "end_date"
                ]
                .dt.year
                .tolist()
            )
        )

    # ========================================================
    # SCIENTIFIC STATISTICS
    # ========================================================

    mean_modis = float(
        df["modis_z"].mean()
    )

    mean_viirs = float(
        df["viirs_z"].mean()
    )

    mean_harmonized = float(
        df[
            "harmonized_activity"
        ].mean()
    )

    max_harmonized = float(
        df[
            "harmonized_activity"
        ].max()
    )

    mean_anomaly = float(
        df["anomaly_z"].mean()
    )

    max_anomaly = float(
        df["anomaly_z"].max()
    )

    mean_discrepancy = float(
        df[
            "sensor_discrepancy"
        ].mean()
    )

    max_discrepancy = float(
        df[
            "sensor_discrepancy"
        ].max()
    )

    mean_frp = float(
        df["total_frp"].mean()
    )

    max_frp = float(
        df["total_frp"].max()
    )

    total_frp = float(
        df["total_frp"].sum()
    )

    # ========================================================
    # DAILY COVERAGE
    # ========================================================

    mean_sectors_observed = float(
        daily[
            "sectors_observed"
        ].mean()
    )

    min_sectors_observed = int(
        daily[
            "sectors_observed"
        ].min()
    )

    max_sectors_observed = int(
        daily[
            "sectors_observed"
        ].max()
    )

    # ========================================================
    # SECTOR LIST
    # ========================================================

    sectors = sorted(
        sector_summary[
            "sector"
        ]
        .astype(str)
        .unique()
    )

    # ========================================================
    # SUMMARY TEXT
    # ========================================================

    lines = []

    lines.append(
        "FIRE HARMONIZER - MILESTONE 5"
    )

    lines.append(
        "TEMPORAL VISUALIZATION & "
        "SCIENTIFIC ANALYSIS SUMMARY"
    )

    lines.append(
        "=" * 70
    )

    lines.append("")

    # ========================================================
    # PURPOSE
    # ========================================================

    lines.append(
        "Purpose"
    )

    lines.append(
        "-------"
    )

    lines.append(
        "Milestone 5 converts the all-years "
        "harmonized science dataset into temporal "
        "and sector-based scientific visualizations."
    )

    lines.append(
        "No NASA downloading is performed."
    )

    lines.append(
        "No geographic map or GIS visualization is used."
    )

    lines.append("")

    # ========================================================
    # DATA COVERAGE
    # ========================================================

    lines.append(
        "Data Coverage"
    )

    lines.append(
        "-------------"
    )

    lines.append(
        f"Years processed      : "
        f"{df['year'].min()} -> "
        f"{df['year'].max()}"
    )

    lines.append(
        "Year list            : "
        + ", ".join(
            str(int(year))
            for year in years
        )
    )

    lines.append(
        f"Science rows         : "
        f"{science_rows:,}"
    )

    lines.append(
        f"Unique dates         : "
        f"{unique_dates:,}"
    )

    lines.append(
        f"Sectors represented : "
        f"{sector_count}"
    )

    lines.append(
        f"Date range           : "
        f"{df['acq_date'].min().date()}"
        f" -> "
        f"{df['acq_date'].max().date()}"
    )

    if (
        start_year is not None
        or end_year is not None
    ):

        requested_start = (
            str(start_year)
            if start_year is not None
            else "earliest"
        )

        requested_end = (
            str(end_year)
            if end_year is not None
            else "latest"
        )

        lines.append(
            f"Requested range     : "
            f"{requested_start} -> "
            f"{requested_end}"
        )

    lines.append("")

    # ========================================================
    # DAILY AGGREGATION
    # ========================================================

    lines.append(
        "Daily Temporal Aggregation"
    )

    lines.append(
        "--------------------------"
    )

    lines.append(
        "Daily temporal values use means across "
        "observed sectors."
    )

    lines.append(
        "Missing dates are not converted to zero."
    )

    lines.append(
        "30-day rolling metrics use a calendar-time "
        "30-day window."
    )

    lines.append(
        f"Mean sectors observed/day:"
        f" {mean_sectors_observed:.2f}"
    )

    lines.append(
        f"Minimum sectors observed:"
        f" {min_sectors_observed}"
    )

    lines.append(
        f"Maximum sectors observed:"
        f" {max_sectors_observed}"
    )

    lines.append("")

    # ========================================================
    # SCIENTIFIC METRICS
    # ========================================================

    lines.append(
        "Scientific Metrics"
    )

    lines.append(
        "------------------"
    )

    lines.append(
        f"Mean MODIS Z             : "
        f"{mean_modis:.4f}"
    )

    lines.append(
        f"Mean VIIRS Z             : "
        f"{mean_viirs:.4f}"
    )

    lines.append(
        f"Mean Harmonized Activity : "
        f"{mean_harmonized:.4f}"
    )

    lines.append(
        f"Maximum Harmonized       : "
        f"{max_harmonized:.4f}"
    )

    lines.append(
        f"Mean Anomaly Z           : "
        f"{mean_anomaly:.4f}"
    )

    lines.append(
        f"Maximum Anomaly Z        : "
        f"{max_anomaly:.4f}"
    )

    lines.append(
        f"Mean Sensor Discrepancy  : "
        f"{mean_discrepancy:.4f}"
    )

    lines.append(
        f"Maximum Sensor Discrepancy:"
        f" {max_discrepancy:.4f}"
    )

    lines.append(
        f"Mean FRP                 : "
        f"{mean_frp:.4f}"
    )

    lines.append(
        f"Maximum FRP              : "
        f"{max_frp:.4f}"
    )

    lines.append(
        f"Total FRP                : "
        f"{total_frp:.4f}"
    )

    lines.append("")

    # ========================================================
    # SECTOR STRUCTURE
    # ========================================================

    lines.append(
        "Sector Structure"
    )

    lines.append(
        "----------------"
    )

    lines.append(
        "Grid: 5 × 5"
    )

    lines.append(
        "Expected sectors: C01-C25"
    )

    lines.append(
        "Sectors found:"
    )

    lines.append(
        "  "
        + ", ".join(
            sectors
        )
    )

    lines.append("")

    # ========================================================
    # M4 INTEGRATION
    # ========================================================

    lines.append(
        "Milestone 4 Integration"
    )

    lines.append(
        "-----------------------"
    )

    lines.append(
        "Source: output/critical_periods.csv"
    )

    lines.append(
        "M4 source is expected to be the all-years "
        "harmonized science dataset."
    )

    lines.append(
        "Critical-period definition:"
    )

    lines.append(
        "anomaly_z >= 2 for at least "
        "3 consecutive calendar days."
    )

    lines.append(
        f"Critical periods: "
        f"{critical_period_count:,}"
    )

    lines.append(
        f"Extreme days inside periods: "
        f"{extreme_day_count:,}"
    )

    if critical_years:

        lines.append(
            "Years touched by critical periods:"
        )

        lines.append(
            "  "
            + ", ".join(
                str(year)
                for year in critical_years
            )
        )

        if len(critical_years) > 1:

            lines.append(
                "Critical-period date range:"
            )

            lines.append(
                "  "
                f"{periods['start_date'].min().date()}"
                " -> "
                f"{periods['end_date'].max().date()}"
            )

    lines.append("")

    # ========================================================
    # GENERATED CHARTS
    # ========================================================

    lines.append(
        "Generated Charts"
    )

    lines.append(
        "----------------"
    )

    chart_files = [
        "harmonized_calendar_all_years.png",
        "anomaly_calendar_all_years.png",
        "sector_matrix_harmonized_all_years.png",
        "sector_matrix_anomaly_all_years.png",
        "sensor_comparison_all_years.png",
        "sensor_discrepancy_all_years.png",
        "critical_period_timeline_all_years.png",
    ]

    for filename in chart_files:

        lines.append(
            f"  output/charts/{filename}"
        )

    lines.append("")

    # ========================================================
    # GENERATED DATA
    # ========================================================

    lines.append(
        "Generated Data Files"
    )

    lines.append(
        "--------------------"
    )

    lines.append(
        f"  {DAILY_AGGREGATE_FILE}"
    )

    lines.append(
        f"  {SECTOR_SUMMARY_FILE}"
    )

    lines.append("")

    # ========================================================
    # VISUALIZATION NOTES
    # ========================================================

    lines.append(
        "Visualization Notes"
    )

    lines.append(
        "--------------------"
    )

    lines.append(
        "Calendar heatmaps use robust "
        "99th-percentile color scaling."
    )

    lines.append(
        "Missing calendar observations are shown "
        "as a neutral gray background."
    )

    lines.append(
        "The critical-period timeline uses the "
        "actual M4 dates and supports multiple years."
    )

    lines.append(
        "Cross-year critical periods are preserved."
    )

    lines.append(
        "FRP is visualized and summarized separately "
        "from the harmonized metric."
    )

    lines.append("")

    # ========================================================
    # STATUS
    # ========================================================

    lines.append(
        "Milestone 5 Status"
    )

    lines.append(
        "------------------"
    )

    lines.append(
        "✅ All-years temporal analysis completed."
    )

    lines.append(
        "✅ Harmonized activity calendar created."
    )

    lines.append(
        "✅ Anomaly calendar created."
    )

    lines.append(
        "✅ 5×5 harmonized sector matrix created."
    )

    lines.append(
        "✅ 5×5 anomaly sector matrix created."
    )

    lines.append(
        "✅ MODIS–VIIRS comparison created."
    )

    lines.append(
        "✅ Sensor discrepancy visualization created."
    )

    lines.append(
        "✅ Milestone 4 all-years critical-period "
        "timeline integrated."
    )

    # ========================================================
    # WRITE
    # ========================================================

    summary_text = "\n".join(
        lines
    )

    SUMMARY_FILE.write_text(
        summary_text,
        encoding="utf-8"
    )

    print(
        f"  ✅ {SUMMARY_FILE}"
    )

    return SUMMARY_FILE


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Fire Harmonizer Milestone 5 — "
            "all-years temporal visualization."
        )
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help=(
            "Optional first year to include."
        ),
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help=(
            "Optional last year to include."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # HEADER
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "FIRE HARMONIZER - MILESTONE 5"
    )

    print(
        "TEMPORAL VISUALIZATION & SCIENTIFIC CHARTS — ALL YEARS"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # LOAD SCIENCE
    # ========================================================

    df = load_science_data(
        start_year=args.start_year,
        end_year=args.end_year
    )

    # ========================================================
    # DAILY AGGREGATE
    # ========================================================

    print()
    print(
        "Creating daily aggregate..."
    )

    daily = create_daily_aggregate(
        df
    )

    daily.to_csv(
        DAILY_AGGREGATE_FILE,
        index=False
    )

    print(
        f"  ✅ Saved {DAILY_AGGREGATE_FILE}"
    )

    # ========================================================
    # SECTOR SUMMARY
    # ========================================================

    print()
    print(
        "Creating all-year sector summary..."
    )

    sector_summary = create_sector_summary(
        df
    )

    sector_summary.to_csv(
        SECTOR_SUMMARY_FILE,
        index=False
    )

    print(
        f"  ✅ Saved {SECTOR_SUMMARY_FILE}"
    )

    # ========================================================
    # CHART 1 — HARMONIZED CALENDAR
    # ========================================================

    print()
    print(
        "Creating harmonized activity calendar..."
    )

    save_calendar_heatmap(
        df=df,
        value_column="harmonized_activity",
        title=(
            "Harmonized Activity Calendar — All Years"
        ),
        colorbar_label=(
            "Mean Harmonized Activity"
        ),
        output_name=(
            "harmonized_calendar_all_years.png"
        )
    )

    # ========================================================
    # CHART 2 — ANOMALY CALENDAR
    # ========================================================

    print()
    print(
        "Creating anomaly calendar..."
    )

    save_calendar_heatmap(
        df=df,
        value_column="anomaly_z",
        title=(
            "Anomaly Z Calendar — All Years"
        ),
        colorbar_label=(
            "Mean Anomaly Z"
        ),
        output_name=(
            "anomaly_calendar_all_years.png"
        )
    )

    # ========================================================
    # CHART 3 — HARMONIZED SECTOR MATRIX
    # ========================================================

    print()
    print(
        "Creating harmonized 5×5 sector matrix..."
    )

    save_sector_matrix(
        sector_summary=sector_summary,
        value_column="mean_harmonized_activity",
        title=(
            "5×5 Sector Matrix — "
            "Mean Harmonized Activity"
        ),
        colorbar_label=(
            "Mean Harmonized Activity"
        ),
        output_name=(
            "sector_matrix_harmonized_all_years.png"
        )
    )

    # ========================================================
    # CHART 4 — ANOMALY SECTOR MATRIX
    # ========================================================

    print()
    print(
        "Creating anomaly 5×5 sector matrix..."
    )

    save_sector_matrix(
        sector_summary=sector_summary,
        value_column="mean_anomaly_z",
        title=(
            "5×5 Sector Matrix — "
            "Mean Anomaly Z"
        ),
        colorbar_label=(
            "Mean Anomaly Z"
        ),
        output_name=(
            "sector_matrix_anomaly_all_years.png"
        )
    )

    # ========================================================
    # CHART 5 — SENSOR COMPARISON
    # ========================================================

    print()
    print(
        "Creating MODIS–VIIRS comparison..."
    )

    save_sensor_comparison(
        daily
    )

    # ========================================================
    # CHART 6 — SENSOR DISCREPANCY
    # ========================================================

    print()
    print(
        "Creating sensor discrepancy chart..."
    )

    save_sensor_discrepancy(
        daily
    )

    # ========================================================
    # LOAD M4
    # ========================================================

    print()
    print(
        "Loading Milestone 4 all-years critical periods..."
    )

    periods = load_critical_periods(
        start_year=args.start_year,
        end_year=args.end_year
    )

    if periods.empty:

        print(
            "  ⚠ No critical periods available."
        )

    else:

        print(
            f"  ✅ Loaded "
            f"{len(periods):,} critical periods."
        )

    # ========================================================
    # CHART 7 — CRITICAL PERIOD TIMELINE
    # ========================================================

    print()
    print(
        "Creating critical-period timeline..."
    )

    save_critical_period_timeline(
        periods
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "Creating Milestone 5 summary..."
    )

    summary_file = create_milestone_5_summary(
        df=df,
        daily=daily,
        sector_summary=sector_summary,
        periods=periods,
        start_year=args.start_year,
        end_year=args.end_year
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "MILESTONE 5 COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Charts:"
    )

    print(
        f"  {CHART_DIR}"
    )

    print()

    print(
        "Generated data:"
    )

    print(
        f"  {DAILY_AGGREGATE_FILE}"
    )

    print(
        f"  {SECTOR_SUMMARY_FILE}"
    )

    print()

    print(
        "Summary:"
    )

    print(
        f"  {summary_file}"
    )

    print()

    print(
        "M4 critical periods:"
    )

    print(
        f"  {M4_CRITICAL_PERIODS_FILE}"
    )

    print()

    print(
        "Generated charts:"
    )

    generated_charts = sorted(
        CHART_DIR.glob(
            "*.png"
        )
    )

    for path in generated_charts:

        print(
            f"  ✅ {path.name}"
        )

    print()

    print(
        "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()