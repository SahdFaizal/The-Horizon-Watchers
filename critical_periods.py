"""
FIRE HARMONIZER - MILESTONE 4
================================

CRITICAL PERIOD DETECTION — ALL YEARS

Definition:

    anomaly_z >= 2
    for at least 3 consecutive calendar days

PRIMARY INPUT:

    output/harmonized_science_all_years.csv

FALLBACK INPUT:

    output/harmonized_science_YYYY.csv

Outputs:

    output/critical_periods.csv
    output/milestone_4_summary.txt

Important:

    - No NASA downloading.
    - No --year argument is required.
    - The all-years M3 science file is used first.
    - If the all-years file is unavailable, yearly files
      are combined automatically.
    - Critical periods are detected continuously across
      all available years.
    - A period may cross December -> January if the dates
      are truly consecutive.
    - Duplicate date + sector rows are rejected.
"""


from pathlib import Path
import argparse

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    BASE_DIR
    / "output"
)


# ============================================================
# INPUT FILES
# ============================================================

ALL_YEARS_FILE = (
    OUTPUT_DIR
    / "harmonized_science_all_years.csv"
)


# ============================================================
# CRITICAL PERIOD SETTINGS
# ============================================================

MIN_ANOMALY = 2.0

MIN_DURATION = 3

EXTREME_ANOMALY = 3.0


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
# OUTPUT COLUMNS
# ============================================================

OUTPUT_COLUMNS = [
    "sector",
    "start_date",
    "end_date",
    "duration_days",
    "peak_date",
    "peak_anomaly_z",
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
# LOAD ONE SCIENCE FILE
# ============================================================

def load_one_science_file(
    path,
    source_year=None,
):
    """
    Load and validate one harmonized science CSV.
    """

    print()
    print(
        f"Loading {path.name} ..."
    )

    try:

        df = pd.read_csv(
            path
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not read {path}:\n{exc}"
        )

    if df.empty:

        raise RuntimeError(
            f"Science file is empty:\n{path}"
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{path.name} is missing required columns:\n"
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Keep only M4-required columns
    # --------------------------------------------------------

    df = df[
        REQUIRED_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    df["acq_date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce"
    )

    df["acq_date"] = (
        df["acq_date"]
        .dt.normalize()
    )

    # --------------------------------------------------------
    # Sector
    #
    # StringDtype preserves missing values instead of turning
    # NaN into the literal string "NAN".
    # --------------------------------------------------------

    df["sector"] = (
        df["sector"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Numerical columns
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Validate rows
    # --------------------------------------------------------

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
            f"  ⚠ Removed {removed:,} invalid row(s)."
        )

    df = df.loc[
        valid
    ].copy()

    if df.empty:

        raise RuntimeError(
            f"No valid rows remain in {path.name}."
        )

    # --------------------------------------------------------
    # Store source year only when useful for diagnostics.
    #
    # For the all-years file, actual year is derived from date.
    # --------------------------------------------------------

    if source_year is not None:

        df["source_year"] = (
            source_year
        )

    else:

        df["source_year"] = (
            df["acq_date"]
            .dt.year
        )

    print(
        f"  ✅ Loaded {len(df):,} rows."
    )

    return df


# ============================================================
# LOAD ALL AVAILABLE SCIENCE DATA
# ============================================================

def load_all_science_data(
    start_year=None,
    end_year=None,
):
    """
    Load the complete historical science dataset.

    Priority:

        1. harmonized_science_all_years.csv
        2. harmonized_science_YYYY.csv files

    Optional:

        --start-year
        --end-year
    """

    # ========================================================
    # PRIMARY SOURCE — ALL YEARS
    # ========================================================

    if ALL_YEARS_FILE.exists():

        print()
        print(
            "=" * 70
        )

        print(
            "PRIMARY M4 INPUT"
        )

        print(
            "=" * 70
        )

        print(
            "Using:"
        )

        print(
            f"  {ALL_YEARS_FILE}"
        )

        df = load_one_science_file(
            ALL_YEARS_FILE
        )

        input_mode = (
            "harmonized_science_all_years.csv"
        )

        # ----------------------------------------------------
        # Apply optional year filtering
        # ----------------------------------------------------

        if start_year is not None:

            df = df[
                df["acq_date"].dt.year
                >= start_year
            ].copy()

        if end_year is not None:

            df = df[
                df["acq_date"].dt.year
                <= end_year
            ].copy()

        if df.empty:

            raise RuntimeError(
                "No science data remains after "
                "the requested year filtering."
            )

    # ========================================================
    # FALLBACK — YEARLY FILES
    # ========================================================

    else:

        print()
        print(
            "=" * 70
        )

        print(
            "ALL-YEARS FILE NOT FOUND"
        )

        print(
            "=" * 70
        )

        print(
            "Falling back to yearly harmonized science files."
        )

        prefix = (
            "harmonized_science_"
        )

        paths = sorted(
            OUTPUT_DIR.glob(
                f"{prefix}*.csv"
            )
        )

        selected = []

        # ----------------------------------------------------
        # Discover numeric-year filenames
        # ----------------------------------------------------

        for path in paths:

            stem = path.stem

            if not stem.startswith(prefix):
                continue

            year_text = (
                stem[
                    len(prefix):
                ]
            )

            # Only accept:
            #
            # harmonized_science_2012.csv
            # harmonized_science_2013.csv
            # ...
            #
            # Ignore:
            #
            # harmonized_science_all_years.csv

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
                "No harmonized science data was found.\n\n"
                "Expected:\n"
                f"  {ALL_YEARS_FILE}\n"
                "or yearly files such as:\n"
                "  harmonized_science_2025.csv"
            )

        selected.sort(
            key=lambda item: item[0]
        )

        print()
        print(
            "Yearly science files selected:"
        )

        for year, path in selected:

            print(
                f"  {year}: {path.name}"
            )

        frames = []

        for year, path in selected:

            frame = load_one_science_file(
                path,
                source_year=year
            )

            frames.append(
                frame
            )

        if not frames:

            raise RuntimeError(
                "No usable yearly harmonized science "
                "data was loaded."
            )

        df = pd.concat(
            frames,
            ignore_index=True
        )

        input_mode = (
            "yearly harmonized_science_YYYY.csv files"
        )

    # ========================================================
    # GLOBAL SORT
    # ========================================================

    df = df.sort_values(
        [
            "sector",
            "acq_date",
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # DUPLICATE DATE + SECTOR CHECK
    # ========================================================

    duplicate_mask = df.duplicated(
        subset=[
            "acq_date",
            "sector",
        ],
        keep=False
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    if duplicate_count > 0:

        duplicate_rows = (
            df.loc[
                duplicate_mask,
                [
                    "acq_date",
                    "sector",
                    "source_year",
                ]
            ]
            .sort_values(
                [
                    "sector",
                    "acq_date",
                ]
            )
        )

        print()
        print(
            "❌ DUPLICATE DATE/SECTOR ROWS DETECTED"
        )

        print(
            f"Duplicate rows: "
            f"{duplicate_count:,}"
        )

        print()
        print(
            "Examples:"
        )

        print(
            duplicate_rows
            .head(20)
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Milestone 4 stopped because duplicate "
            "date + sector rows were found."
        )

    # ========================================================
    # SECTOR CHECK
    # ========================================================

    sectors = sorted(
        df["sector"]
        .dropna()
        .unique()
    )

    expected_sectors = [
        f"C{i:02d}"
        for i in range(
            1,
            26
        )
    ]

    missing_sectors = [
        sector
        for sector in expected_sectors
        if sector not in sectors
    ]

    if missing_sectors:

        print()
        print(
            "⚠ Missing expected sector(s): "
            + ", ".join(
                missing_sectors
            )
        )

    # ========================================================
    # COMBINED DATA SUMMARY
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "COMBINED SCIENCE DATA"
    )

    print(
        "=" * 70
    )

    print(
        f"Input source: "
        f"{input_mode}"
    )

    print(
        f"Years loaded: "
        f"{df['source_year'].nunique():,}"
    )

    print(
        "Year range: "
        f"{df['acq_date'].dt.year.min()}"
        f" -> "
        f"{df['acq_date'].dt.year.max()}"
    )

    print(
        f"Rows: "
        f"{len(df):,}"
    )

    print(
        f"Unique sectors: "
        f"{df['sector'].nunique():,}"
    )

    print(
        f"Unique dates: "
        f"{df['acq_date'].nunique():,}"
    )

    print(
        "Date range: "
        f"{df['acq_date'].min().date()}"
        f" -> "
        f"{df['acq_date'].max().date()}"
    )

    print(
        "=" * 70
    )

    return df, input_mode


# ============================================================
# DETECT CRITICAL PERIODS FOR ONE SECTOR
# ============================================================

def detect_for_sector(
    df: pd.DataFrame,
    sector: str,
):
    """
    Detect periods where:

        anomaly_z >= MIN_ANOMALY

    for at least:

        MIN_DURATION

    exact consecutive calendar days.
    """

    sector_df = (
        df[
            df["sector"] == sector
        ]
        .sort_values(
            "acq_date"
        )
        .copy()
    )

    if sector_df.empty:

        return []

    # --------------------------------------------------------
    # Duplicate safety check
    # --------------------------------------------------------

    duplicate_dates = (
        sector_df
        .duplicated(
            subset=["acq_date"],
            keep=False
        )
    )

    if duplicate_dates.any():

        raise RuntimeError(
            f"Duplicate dates found inside sector "
            f"{sector}."
        )

    # --------------------------------------------------------
    # Threshold flag
    # --------------------------------------------------------

    sector_df["high"] = (
        sector_df["anomaly_z"]
        >= MIN_ANOMALY
    )

    results = []

    i = 0

    # ========================================================
    # SCAN CHRONOLOGICALLY
    # ========================================================

    while i < len(sector_df):

        current = (
            sector_df.iloc[i]
        )

        # ----------------------------------------------------
        # Current row below threshold
        # ----------------------------------------------------

        if not current["high"]:

            i += 1

            continue

        # ----------------------------------------------------
        # Start candidate
        # ----------------------------------------------------

        start = i

        end = i

        # ----------------------------------------------------
        # Extend candidate
        # ----------------------------------------------------

        while (
            end + 1
            < len(sector_df)
        ):

            current_date = (
                sector_df.iloc[
                    end
                ]["acq_date"]
            )

            next_date = (
                sector_df.iloc[
                    end + 1
                ]["acq_date"]
            )

            days_difference = (
                next_date
                - current_date
            ).days

            next_high = (
                sector_df.iloc[
                    end + 1
                ]["high"]
            )

            # Both requirements must hold:
            #
            # 1. exactly one calendar day later
            # 2. next day is also >= 2

            if (
                days_difference == 1
                and next_high
            ):

                end += 1

            else:

                break

        # ----------------------------------------------------
        # Extract candidate period
        # ----------------------------------------------------

        period = (
            sector_df.iloc[
                start:end + 1
            ]
            .copy()
        )

        duration = len(
            period
        )

        # ====================================================
        # RETAIN >= 3 DAYS ONLY
        # ====================================================

        if duration >= MIN_DURATION:

            # ------------------------------------------------
            # Peak anomaly
            # ------------------------------------------------

            peak_index = (
                period[
                    "anomaly_z"
                ]
                .idxmax()
            )

            peak = period.loc[
                peak_index
            ]

            # ------------------------------------------------
            # Sensor means
            # ------------------------------------------------

            mean_modis = (
                period[
                    "modis_z"
                ]
                .mean()
            )

            mean_viirs = (
                period[
                    "viirs_z"
                ]
                .mean()
            )

            # ------------------------------------------------
            # Leading component
            # ------------------------------------------------

            sensor_difference = abs(
                mean_modis
                - mean_viirs
            )

            if sensor_difference < 0.25:

                leading_component = (
                    "Balanced"
                )

            elif (
                abs(mean_modis)
                >
                abs(mean_viirs)
            ):

                leading_component = (
                    "MODIS"
                )

            else:

                leading_component = (
                    "VIIRS"
                )

            # ------------------------------------------------
            # Extreme days
            # ------------------------------------------------

            extreme_days = int(
                (
                    period[
                        "anomaly_z"
                    ]
                    >= EXTREME_ANOMALY
                )
                .sum()
            )

            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            results.append(
                {
                    "sector":
                        sector,

                    "start_date":
                        period[
                            "acq_date"
                        ].min(),

                    "end_date":
                        period[
                            "acq_date"
                        ].max(),

                    "duration_days":
                        duration,

                    "peak_date":
                        peak[
                            "acq_date"
                        ],

                    "peak_anomaly_z":
                        peak[
                            "anomaly_z"
                        ],

                    "mean_anomaly_z":
                        period[
                            "anomaly_z"
                        ].mean(),

                    "mean_harmonized_activity":
                        period[
                            "harmonized_activity"
                        ].mean(),

                    "peak_frp":
                        period[
                            "total_frp"
                        ].max(),

                    "mean_frp":
                        period[
                            "total_frp"
                        ].mean(),

                    "mean_sensor_discrepancy":
                        period[
                            "sensor_discrepancy"
                        ].mean(),

                    "mean_modis_z":
                        mean_modis,

                    "mean_viirs_z":
                        mean_viirs,

                    "leading_component":
                        leading_component,

                    "extreme_days":
                        extreme_days,
                }
            )

        # ----------------------------------------------------
        # Continue after this candidate
        # ----------------------------------------------------

        i = (
            end + 1
        )

    return results


# ============================================================
# VALIDATE DETECTED PERIODS
# ============================================================

def validate_periods(
    periods
):
    """
    Perform final sanity checks.
    """

    if periods.empty:

        return

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    if (
        periods[
            "duration_days"
        ]
        < MIN_DURATION
    ).any():

        raise RuntimeError(
            "Internal error: a critical period "
            "has an invalid duration."
        )

    # --------------------------------------------------------
    # Peak anomaly
    # --------------------------------------------------------

    if (
        periods[
            "peak_anomaly_z"
        ]
        < MIN_ANOMALY
    ).any():

        raise RuntimeError(
            "Internal error: a critical period "
            "has peak anomaly below the threshold."
        )

    # --------------------------------------------------------
    # Peak must lie inside its period
    # --------------------------------------------------------

    invalid_peak_location = (
        (
            periods["peak_date"]
            < periods["start_date"]
        )
        |
        (
            periods["peak_date"]
            > periods["end_date"]
        )
    )

    if invalid_peak_location.any():

        raise RuntimeError(
            "Internal error: peak date lies outside "
            "its critical-period date range."
        )

    # --------------------------------------------------------
    # Date ordering
    # --------------------------------------------------------

    invalid_order = (
        periods["end_date"]
        <
        periods["start_date"]
    )

    if invalid_order.any():

        raise RuntimeError(
            "Internal error: critical-period end date "
            "precedes start date."
        )

    # --------------------------------------------------------
    # Check duration against actual date range
    # --------------------------------------------------------

    expected_duration = (
        periods["end_date"]
        - periods["start_date"]
    ).dt.days + 1

    invalid_duration = (
        expected_duration
        !=
        periods["duration_days"]
    )

    if invalid_duration.any():

        raise RuntimeError(
            "Internal error: reported duration does "
            "not match the calendar date range."
        )


# ============================================================
# CREATE MILESTONE 4 SUMMARY
# ============================================================

def create_summary(
    df,
    periods,
    input_mode,
    selected_start_year,
    selected_end_year,
):
    """
    Create:

        output/milestone_4_summary.txt
    """

    summary_file = (
        OUTPUT_DIR
        / "milestone_4_summary.txt"
    )

    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    years = sorted(
        df["source_year"]
        .unique()
    )

    sectors = sorted(
        df["sector"]
        .dropna()
        .unique()
    )

    science_rows = len(
        df
    )

    unique_dates = (
        df["acq_date"]
        .nunique()
    )

    # ========================================================
    # CRITICAL PERIOD STATISTICS
    # ========================================================

    total_periods = len(
        periods
    )

    if (
        not periods.empty
    ):

        total_extreme_days = int(
            periods[
                "extreme_days"
            ]
            .sum()
        )

        longest_duration = int(
            periods[
                "duration_days"
            ]
            .max()
        )

        mean_duration = float(
            periods[
                "duration_days"
            ]
            .mean()
        )

        highest_peak = float(
            periods[
                "peak_anomaly_z"
            ]
            .max()
        )

        max_frp = float(
            periods[
                "peak_frp"
            ]
            .max()
        )

        mean_discrepancy = float(
            periods[
                "mean_sensor_discrepancy"
            ]
            .mean()
        )

        critical_start = (
            periods[
                "start_date"
            ]
            .min()
            .date()
        )

        critical_end = (
            periods[
                "end_date"
            ]
            .max()
            .date()
        )

        # ----------------------------------------------------
        # Period count by sector
        # ----------------------------------------------------

        periods_by_sector = (
            periods[
                "sector"
            ]
            .value_counts()
            .sort_index()
        )

        # ----------------------------------------------------
        # Leading components
        # ----------------------------------------------------

        leading_counts = (
            periods[
                "leading_component"
            ]
            .value_counts()
        )

    else:

        total_extreme_days = 0

        longest_duration = 0

        mean_duration = 0.0

        highest_peak = 0.0

        max_frp = 0.0

        mean_discrepancy = 0.0

        critical_start = (
            "None"
        )

        critical_end = (
            "None"
        )

        periods_by_sector = (
            pd.Series(
                dtype="int64"
            )
        )

        leading_counts = (
            pd.Series(
                dtype="int64"
            )
        )

    # ========================================================
    # SUMMARY TEXT
    # ========================================================

    lines = []

    lines.append(
        "FIRE HARMONIZER - MILESTONE 4"
    )

    lines.append(
        "CRITICAL PERIOD DETECTION SUMMARY"
    )

    lines.append(
        "=" * 70
    )

    lines.append("")

    # ========================================================
    # DEFINITION
    # ========================================================

    lines.append(
        "Definition"
    )

    lines.append(
        "----------"
    )

    lines.append(
        f"Critical period threshold : "
        f"anomaly_z >= {MIN_ANOMALY:.1f}"
    )

    lines.append(
        f"Minimum duration          : "
        f"{MIN_DURATION} consecutive calendar days"
    )

    lines.append(
        f"Extreme day threshold     : "
        f"anomaly_z >= {EXTREME_ANOMALY:.1f}"
    )

    lines.append(
        "Continuity requirement    : "
        "exactly consecutive calendar dates"
    )

    lines.append("")

    # ========================================================
    # INPUT
    # ========================================================

    lines.append(
        "Input Data"
    )

    lines.append(
        "----------"
    )

    lines.append(
        f"Input source : "
        f"{input_mode}"
    )

    lines.append(
        f"Years processed : "
        f"{len(years)}"
    )

    lines.append(
        "Year list       : "
        + ", ".join(
            str(int(year))
            for year in years
        )
    )

    lines.append(
        f"Sectors analyzed: "
        f"{len(sectors)}"
    )

    lines.append(
        f"Science rows    : "
        f"{science_rows:,}"
    )

    lines.append(
        f"Unique dates    : "
        f"{unique_dates:,}"
    )

    lines.append(
        f"Date range      : "
        f"{df['acq_date'].min().date()}"
        f" -> "
        f"{df['acq_date'].max().date()}"
    )

    if (
        selected_start_year is not None
        or selected_end_year is not None
    ):

        start_text = (
            str(selected_start_year)
            if selected_start_year is not None
            else "earliest"
        )

        end_text = (
            str(selected_end_year)
            if selected_end_year is not None
            else "latest"
        )

        lines.append(
            f"Requested range: "
            f"{start_text} -> {end_text}"
        )

    lines.append("")

    # ========================================================
    # RESULTS
    # ========================================================

    lines.append(
        "Critical Period Results"
    )

    lines.append(
        "-----------------------"
    )

    lines.append(
        f"Critical periods detected : "
        f"{total_periods:,}"
    )

    lines.append(
        f"Extreme days inside periods: "
        f"{total_extreme_days:,}"
    )

    lines.append(
        f"Longest critical period    : "
        f"{longest_duration} days"
    )

    lines.append(
        f"Mean critical-period length: "
        f"{mean_duration:.2f} days"
    )

    lines.append(
        f"Highest peak anomaly Z     : "
        f"{highest_peak:.3f}"
    )

    lines.append(
        f"Maximum peak FRP           : "
        f"{max_frp:.3f}"
    )

    lines.append(
        f"Mean sensor discrepancy    : "
        f"{mean_discrepancy:.3f}"
    )

    lines.append(
        f"Critical-period date range : "
        f"{critical_start} -> "
        f"{critical_end}"
    )

    lines.append("")

    # ========================================================
    # PERIODS BY SECTOR
    # ========================================================

    lines.append(
        "Critical Periods by Sector"
    )

    lines.append(
        "--------------------------"
    )

    if periods_by_sector.empty:

        lines.append(
            "None"
        )

    else:

        for sector, count in (
            periods_by_sector.items()
        ):

            lines.append(
                f"  {sector}: {int(count):,}"
            )

    lines.append("")

    # ========================================================
    # LEADING SENSOR COMPONENT
    # ========================================================

    lines.append(
        "Leading Sensor Component"
    )

    lines.append(
        "------------------------"
    )

    if leading_counts.empty:

        lines.append(
            "None"
        )

    else:

        for component, count in (
            leading_counts.items()
        ):

            lines.append(
                f"  {component}: {int(count):,}"
            )

    lines.append("")

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    lines.append(
        "Output Files"
    )

    lines.append(
        "------------"
    )

    lines.append(
        f"Critical periods CSV:"
    )

    lines.append(
        f"  {OUTPUT_DIR / 'critical_periods.csv'}"
    )

    lines.append(
        f"Milestone 4 summary:"
    )

    lines.append(
        f"  {summary_file}"
    )

    lines.append("")

    # ========================================================
    # VALIDATION
    # ========================================================

    lines.append(
        "Validation"
    )

    lines.append(
        "----------"
    )

    lines.append(
        "Duplicate date + sector rows were checked "
        "before detection."
    )

    lines.append(
        "Critical periods require anomaly_z >= 2."
    )

    lines.append(
        "Critical periods require at least 3 "
        "consecutive calendar days."
    )

    lines.append(
        "Dates must differ by exactly one calendar day."
    )

    lines.append(
        "Periods may cross December -> January "
        "when dates are truly consecutive."
    )

    lines.append(
        "Extreme days are anomaly_z >= 3."
    )

    # ========================================================
    # WRITE
    # ========================================================

    summary_text = "\n".join(
        lines
    )

    summary_file.write_text(
        summary_text,
        encoding="utf-8"
    )

    return summary_file


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Fire Harmonizer Milestone 4 — "
            "critical-period detection across all years."
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
        "FIRE HARMONIZER - MILESTONE 4"
    )

    print(
        "CRITICAL PERIOD DETECTION — ALL YEARS"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Critical threshold: "
        f"anomaly_z >= {MIN_ANOMALY}"
    )

    print(
        f"Minimum duration: "
        f"{MIN_DURATION} consecutive days"
    )

    print(
        f"Extreme threshold: "
        f"anomaly_z >= {EXTREME_ANOMALY}"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    df, input_mode = (
        load_all_science_data(
            start_year=args.start_year,
            end_year=args.end_year,
        )
    )

    # ========================================================
    # DETECT
    # ========================================================

    all_results = []

    sectors = sorted(
        df["sector"]
        .dropna()
        .unique()
    )

    print()

    print(
        f"Detecting critical periods across "
        f"{len(sectors):,} sectors..."
    )

    for sector in sectors:

        results = detect_for_sector(
            df,
            sector
        )

        all_results.extend(
            results
        )

        if results:

            print(
                f"  {sector}: "
                f"{len(results):,} "
                f"critical period(s)"
            )

    # ========================================================
    # RESULT DATAFRAME
    # ========================================================

    periods = pd.DataFrame(
        all_results,
        columns=OUTPUT_COLUMNS
    )

    # ========================================================
    # NO PERIODS
    # ========================================================

    if periods.empty:

        print()
        print(
            "No critical periods detected."
        )

        periods = pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    else:

        # ----------------------------------------------------
        # Normalize dates
        # ----------------------------------------------------

        periods["start_date"] = pd.to_datetime(
            periods["start_date"],
            errors="coerce"
        )

        periods["end_date"] = pd.to_datetime(
            periods["end_date"],
            errors="coerce"
        )

        periods["peak_date"] = pd.to_datetime(
            periods["peak_date"],
            errors="coerce"
        )

        # ----------------------------------------------------
        # Sort chronologically
        # ----------------------------------------------------

        periods = (
            periods
            .sort_values(
                [
                    "start_date",
                    "sector",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    validate_periods(
        periods
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SAVE CRITICAL PERIODS
    # ========================================================

    output_file = (
        OUTPUT_DIR
        / "critical_periods.csv"
    )

    periods.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # CREATE SUMMARY
    # ========================================================

    summary_file = create_summary(
        df=df,
        periods=periods,
        input_mode=input_mode,
        selected_start_year=args.start_year,
        selected_end_year=args.end_year,
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "MILESTONE 4 COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Input:"
    )

    print(
        f"  {input_mode}"
    )

    print()

    print(
        "Saved:"
    )

    print(
        f"  {output_file}"
    )

    print(
        f"  {summary_file}"
    )

    print()

    print(
        "Critical periods:",
        len(periods)
    )

    # ========================================================
    # TOP PERIODS
    # ========================================================

    if not periods.empty:

        print()

        print(
            "Top periods:"
        )

        display_columns = [
            "sector",
            "start_date",
            "end_date",
            "duration_days",
            "peak_anomaly_z",
            "peak_date",
            "extreme_days",
        ]

        print(
            periods[
                display_columns
            ]
            .head(15)
            .to_string(
                index=False
            )
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