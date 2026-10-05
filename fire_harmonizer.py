"""
FIRE HARMONIZER - MILESTONE 1
================================

OFFLINE VERSION

Purpose:
    Read existing MODIS and VIIRS FIRMS CSV files,
    convert both sensors into the same 5x5 AOI sector system,
    aggregate observations by day and sector,
    normalize each sensor independently,
    calculate a first Harmonized Activity Index,
    calculate sensor discrepancy,
    and save every important intermediate result.

IMPORTANT:
    This version DOES NOT download anything.
    It does NOT use the internet.
    It does NOT require a FIRMS API key.

Pipeline:

    RAW MODIS CSV
          |
          v
    Clean + AOI filter
          |
          v
    5x5 sectors
          |
          v
    Processed MODIS
          |
          v
    Daily MODIS matrix
          |
          |
          +--------------------+
                               |
    RAW VIIRS CSV              |
          |                    |
          v                    |
    Clean + AOI filter         |
          |                    |
          v                    |
    5x5 sectors                |
          |                    |
          v                    |
    Processed VIIRS            |
          |                    |
          v                    |
    Daily VIIRS matrix         |
          |                    |
          +---------+----------+
                    |
                    v
          Common DATE x SECTOR
                    |
                    v
          Sensor Z-score
                    |
                    v
          Harmonized Activity
                    |
          +---------+---------+
          |                   |
          v                   v
      Discrepancy            FRP
          |
          v
     FINAL OUTPUT
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# File locations
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "output"


# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

MODIS_FILE = RAW_DIR / "modis.csv"
VIIRS_FILE = RAW_DIR / "viirs.csv"


# ------------------------------------------------------------
# AOI
# ------------------------------------------------------------
#
# Format:
# west, south, east, north
#
# Example: Amazon Basin
#

AOI = (
    -75.0,   # west
    -15.0,   # south
    -50.0,   # east
    5.0      # north
)


# ------------------------------------------------------------
# Spatial grid
# ------------------------------------------------------------

GRID_SIZE = 5


# ------------------------------------------------------------
# Harmonization weights
# ------------------------------------------------------------

MODIS_WEIGHT = 0.5
VIIRS_WEIGHT = 0.5


# ============================================================
# 2. CREATE REQUIRED DIRECTORIES
# ============================================================

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. UTILITY FUNCTIONS
# ============================================================

def print_section(title: str) -> None:
    """Print a readable section header."""

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_subsection(title: str) -> None:
    """Print a smaller section header."""

    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


# ============================================================
# 4. CHECK INPUT FILES
# ============================================================

def check_input_files() -> None:
    """
    Make sure the two existing CSV files are present.

    No downloading occurs.
    """

    missing_files = []

    if not MODIS_FILE.exists():
        missing_files.append(str(MODIS_FILE))

    if not VIIRS_FILE.exists():
        missing_files.append(str(VIIRS_FILE))

    if missing_files:
        print("\nERROR: Required input files are missing.\n")

        for file in missing_files:
            print(f"  Missing: {file}")

        print("\nPlace your existing FIRMS CSV files here:")

        print(f"  {MODIS_FILE}")
        print(f"  {VIIRS_FILE}")

        raise FileNotFoundError(
            "MODIS and/or VIIRS input files were not found."
        )


# ============================================================
# 5. LOAD EXISTING DATA
# ============================================================

def load_raw_data():
    """
    Read the existing MODIS and VIIRS CSV files.

    Returns:
        modis_df
        viirs_df
    """

    print_subsection("LOADING EXISTING RAW DATA")

    print(f"MODIS file: {MODIS_FILE}")
    print(f"VIIRS file: {VIIRS_FILE}")

    modis = pd.read_csv(MODIS_FILE)
    viirs = pd.read_csv(VIIRS_FILE)

    print(f"\nMODIS raw rows: {len(modis):,}")
    print(f"VIIRS raw rows: {len(viirs):,}")

    return modis, viirs


# ============================================================
# 6. STANDARDIZE COLUMN NAMES
# ============================================================

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names so that FIRMS files can be
    handled consistently.

    Example:
        'Latitude' -> 'latitude'
        ' FRP '     -> 'frp'
    """

    df = df.copy()

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    return df


# ============================================================
# 7. PREPARE ONE SENSOR
# ============================================================

def prepare_sensor(
    df: pd.DataFrame,
    sensor_name: str,
    bounds: tuple,
    grid_size: int = 5,
) -> pd.DataFrame:
    """
    Clean one sensor dataset and assign observations
    to the common 5x5 AOI sector system.
    """

    print_subsection(
        f"PROCESSING {sensor_name}"
    )

    df = standardize_columns(df)

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "latitude",
        "longitude",
        "acq_date",
        "frp",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{sensor_name} is missing required columns: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # Convert data types
    # --------------------------------------------------------

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    df["frp"] = pd.to_numeric(
        df["frp"],
        errors="coerce"
    ).fillna(0)

    df["acq_date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
            "acq_date"
        ]
    ).copy()

    after_invalid = len(df)

    print(
        f"Removed invalid rows: "
        f"{before - after_invalid:,}"
    )

    # --------------------------------------------------------
    # AOI filter
    # --------------------------------------------------------

    west, south, east, north = bounds

    before_aoi = len(df)

    df = df[
        (df["longitude"] >= west)
        & (df["longitude"] <= east)
        & (df["latitude"] >= south)
        & (df["latitude"] <= north)
    ].copy()

    after_aoi = len(df)

    print(
        f"Removed rows outside AOI: "
        f"{before_aoi - after_aoi:,}"
    )

    # --------------------------------------------------------
    # Sensor label
    # --------------------------------------------------------

    df["sensor"] = sensor_name

    # --------------------------------------------------------
    # Assign observations to 5x5 sectors
    # --------------------------------------------------------

    x = (
        (df["longitude"] - west)
        / (east - west)
    )

    y = (
        (df["latitude"] - south)
        / (north - south)
    )

    df["col"] = np.floor(
        x * grid_size
    ).astype(int)

    df["row"] = np.floor(
        y * grid_size
    ).astype(int)

    # Protect against boundary issues
    df["col"] = df["col"].clip(
        0,
        grid_size - 1
    )

    df["row"] = df["row"].clip(
        0,
        grid_size - 1
    )

    # --------------------------------------------------------
    # Convert row/column to C01 ... C25
    # --------------------------------------------------------

    sector_number = (
        df["row"] * grid_size
        + df["col"]
        + 1
    )

    df["sector"] = (
        "C"
        + sector_number
        .astype(int)
        .astype(str)
        .str.zfill(2)
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "acq_date",
            "sector"
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"{sensor_name} processed rows: "
        f"{len(df):,}"
    )

    print(
        f"{sensor_name} sectors used: "
        f"{df['sector'].nunique()}"
    )

    return df


# ============================================================
# 8. SAVE PROCESSED SENSOR DATA
# ============================================================

def save_processed_sensor(
    df: pd.DataFrame,
    sensor_name: str,
) -> Path:
    """
    Save cleaned + sectorized sensor observations.
    """

    filename = (
        f"{sensor_name.lower()}_processed.csv"
    )

    path = PROCESSED_DIR / filename

    df.to_csv(
        path,
        index=False
    )

    print(f"Saved: {path}")

    return path


# ============================================================
# 9. AGGREGATE SENSOR BY DATE + SECTOR
# ============================================================

def aggregate_sensor(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert individual hotspot detections into:

        DATE × SECTOR

    Metrics:
        detection_count
        total_frp
        mean_frp
    """

    result = (
        df.groupby(
            [
                "acq_date",
                "sector"
            ],
            as_index=False
        )
        .agg(
            detection_count=(
                "sector",
                "size"
            ),
            total_frp=(
                "frp",
                "sum"
            ),
            mean_frp=(
                "frp",
                "mean"
            )
        )
    )

    return result


# ============================================================
# 10. SAVE DAILY SENSOR DATA
# ============================================================

def save_daily_sensor(
    df: pd.DataFrame,
    sensor_name: str,
) -> Path:

    filename = (
        f"{sensor_name.lower()}_daily.csv"
    )

    path = PROCESSED_DIR / filename

    df.to_csv(
        path,
        index=False
    )

    print(f"Saved: {path}")

    return path


# ============================================================
# 11. CREATE COMPLETE DATE × SECTOR MATRIX
# ============================================================

def create_common_matrix(
    modis_daily: pd.DataFrame,
    viirs_daily: pd.DataFrame,
    bounds: tuple,
    grid_size: int,
) -> pd.DataFrame:
    """
    Create a complete Date × Sector matrix.

    Every date and all 25 sectors exist in the final matrix,
    including combinations where no hotspot was detected.
    """

    start_date = min(
        modis_daily["acq_date"].min(),
        viirs_daily["acq_date"].min()
    )

    end_date = max(
        modis_daily["acq_date"].max(),
        viirs_daily["acq_date"].max()
    )

    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    sectors = [
        f"C{i:02d}"
        for i in range(
            1,
            grid_size * grid_size + 1
        )
    ]

    index = pd.MultiIndex.from_product(
        [
            dates,
            sectors
        ],
        names=[
            "acq_date",
            "sector"
        ]
    )

    result = pd.DataFrame(
        index=index
    ).reset_index()

    # --------------------------------------------------------
    # MODIS
    # --------------------------------------------------------

    result = result.merge(
        modis_daily,
        on=[
            "acq_date",
            "sector"
        ],
        how="left"
    )

    result = result.rename(
        columns={
            "detection_count":
                "modis_count",

            "total_frp":
                "modis_frp",

            "mean_frp":
                "modis_mean_frp"
        }
    )

    # --------------------------------------------------------
    # VIIRS
    # --------------------------------------------------------

    result = result.merge(
        viirs_daily,
        on=[
            "acq_date",
            "sector"
        ],
        how="left"
    )

    result = result.rename(
        columns={
            "detection_count":
                "viirs_count",

            "total_frp":
                "viirs_frp",

            "mean_frp":
                "viirs_mean_frp"
        }
    )

    # --------------------------------------------------------
    # Missing observations = 0 detections
    # --------------------------------------------------------

    numeric_columns = [
        "modis_count",
        "viirs_count",
        "modis_frp",
        "viirs_frp",
        "modis_mean_frp",
        "viirs_mean_frp",
    ]

    for column in numeric_columns:
        if column in result.columns:
            result[column] = (
                result[column]
                .fillna(0)
            )

    return result


# ============================================================
# 12. Z-SCORE NORMALIZATION
# ============================================================

def zscore(
    series: pd.Series,
) -> pd.Series:
    """
    First-pass standardization.

    IMPORTANT:
        This is Milestone 1 normalization only.

        Later we will replace this with a proper historical
        seasonal baseline.
    """

    mean = series.mean()

    std = series.std()

    if pd.isna(std) or std == 0:
        std = 1.0

    return (
        series - mean
    ) / std


# ============================================================
# 13. CORE HARMONIZATION FUNCTION
# ============================================================

def calculate_harmonization(
    df: pd.DataFrame,
    modis_weight: float = 0.5,
    viirs_weight: float = 0.5,
) -> pd.DataFrame:
    """
    CORE SCIENTIFIC FUNCTION OF MILESTONE 1.

    MODIS count
          ↓
       MODIS Z

    VIIRS count
          ↓
       VIIRS Z

          ↓

    H = wM * MODIS_Z
        +
        wV * VIIRS_Z

          ↓

    Sensor discrepancy
    Total FRP
    """

    if not np.isclose(
        modis_weight + viirs_weight,
        1.0
    ):
        raise ValueError(
            "MODIS_WEIGHT + VIIRS_WEIGHT must equal 1.0"
        )

    result = df.copy()

    # --------------------------------------------------------
    # Independent normalization
    # --------------------------------------------------------

    result["modis_z"] = zscore(
        result["modis_count"]
    )

    result["viirs_z"] = zscore(
        result["viirs_count"]
    )

    # --------------------------------------------------------
    # Harmonized Activity
    # --------------------------------------------------------

    result["harmonized_activity"] = (
        modis_weight
        * result["modis_z"]
        +
        viirs_weight
        * result["viirs_z"]
    )

    # --------------------------------------------------------
    # Sensor discrepancy
    # --------------------------------------------------------

    result["sensor_discrepancy"] = (
        result["modis_z"]
        -
        result["viirs_z"]
    ).abs()

    # --------------------------------------------------------
    # FRP remains separate
    # --------------------------------------------------------

    result["total_frp"] = (
        result["modis_frp"]
        +
        result["viirs_frp"]
    )

    # --------------------------------------------------------
    # Order columns
    # --------------------------------------------------------

    preferred_columns = [
        "acq_date",
        "sector",

        "modis_count",
        "viirs_count",

        "modis_z",
        "viirs_z",

        "harmonized_activity",

        "sensor_discrepancy",

        "modis_frp",
        "viirs_frp",
        "total_frp",

        "modis_mean_frp",
        "viirs_mean_frp",
    ]

    existing_columns = [
        column
        for column in preferred_columns
        if column in result.columns
    ]

    other_columns = [
        column
        for column in result.columns
        if column not in existing_columns
    ]

    result = result[
        existing_columns + other_columns
    ]

    return result


# ============================================================
# 14. SAVE FINAL RESULT
# ============================================================

def save_final_output(
    result: pd.DataFrame,
) -> Path:

    output_file = (
        OUTPUT_DIR
        / "harmonized_daily_sector.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(f"\nSaved final output: {output_file}")

    return output_file


# ============================================================
# 15. WRITE MILESTONE SUMMARY
# ============================================================

def write_summary(
    modis_raw: pd.DataFrame,
    viirs_raw: pd.DataFrame,
    modis_processed: pd.DataFrame,
    viirs_processed: pd.DataFrame,
    modis_daily: pd.DataFrame,
    viirs_daily: pd.DataFrame,
    result: pd.DataFrame,
) -> Path:
    """
    Write a small text summary so that every run leaves
    behind a record of what happened.
    """

    summary_file = (
        OUTPUT_DIR
        / "milestone_1_summary.txt"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "FIRE HARMONIZER - MILESTONE 1\n"
        )

        file.write(
            "====================================\n\n"
        )

        file.write(
            "MODE: OFFLINE\n"
        )

        file.write(
            "No data was downloaded by this script.\n\n"
        )

        file.write(
            "AOI:\n"
        )

        file.write(
            f"  {AOI}\n\n"
        )

        file.write(
            "GRID:\n"
        )

        file.write(
            f"  {GRID_SIZE} x {GRID_SIZE}\n\n"
        )

        file.write(
            "WEIGHTS:\n"
        )

        file.write(
            f"  MODIS = {MODIS_WEIGHT}\n"
        )

        file.write(
            f"  VIIRS = {VIIRS_WEIGHT}\n\n"
        )

        file.write(
            "RAW ROWS:\n"
        )

        file.write(
            f"  MODIS = {len(modis_raw):,}\n"
        )

        file.write(
            f"  VIIRS = {len(viirs_raw):,}\n\n"
        )

        file.write(
            "PROCESSED ROWS:\n"
        )

        file.write(
            f"  MODIS = {len(modis_processed):,}\n"
        )

        file.write(
            f"  VIIRS = {len(viirs_processed):,}\n\n"
        )

        file.write(
            "DAILY SECTOR ROWS:\n"
        )

        file.write(
            f"  MODIS = {len(modis_daily):,}\n"
        )

        file.write(
            f"  VIIRS = {len(viirs_daily):,}\n\n"
        )

        file.write(
            "FINAL COMMON MATRIX:\n"
        )

        file.write(
            f"  Rows = {len(result):,}\n"
        )

        file.write(
            f"  Sectors = {result['sector'].nunique()}\n"
        )

        file.write(
            f"  Dates = {result['acq_date'].nunique()}\n\n"
        )

        file.write(
            "OUTPUT COLUMNS:\n"
        )

        for column in result.columns:
            file.write(
                f"  - {column}\n"
            )

    print(
        f"Saved summary: {summary_file}"
    )

    return summary_file


# ============================================================
# 16. DISPLAY FINAL SAMPLE
# ============================================================

def display_result_sample(
    result: pd.DataFrame,
) -> None:

    print_subsection(
        "FINAL HARMONIZED DATA - FIRST 20 ROWS"
    )

    columns = [
        "acq_date",
        "sector",
        "modis_count",
        "viirs_count",
        "modis_z",
        "viirs_z",
        "harmonized_activity",
        "sensor_discrepancy",
        "total_frp",
    ]

    print(
        result[columns]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# 17. MAIN PIPELINE
# ============================================================

def main():

    print_section(
        "FIRE HARMONIZER - MILESTONE 1"
    )

    print(
        "\nOFFLINE MODE"
    )

    print(
        "Existing CSV files will be processed."
    )

    print(
        "No data will be downloaded."
    )

    print(
        f"\nAOI: {AOI}"
    )

    print(
        f"Grid: {GRID_SIZE} x {GRID_SIZE}"
    )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    check_input_files()

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    modis_raw, viirs_raw = (
        load_raw_data()
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    modis_processed = prepare_sensor(
        modis_raw,
        sensor_name="MODIS",
        bounds=AOI,
        grid_size=GRID_SIZE,
    )

    viirs_processed = prepare_sensor(
        viirs_raw,
        sensor_name="VIIRS",
        bounds=AOI,
        grid_size=GRID_SIZE,
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    print_subsection(
        "SAVING PROCESSED SENSOR DATA"
    )

    save_processed_sensor(
        modis_processed,
        "MODIS"
    )

    save_processed_sensor(
        viirs_processed,
        "VIIRS"
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    print_subsection(
        "AGGREGATING DAILY SENSOR DATA"
    )

    modis_daily = aggregate_sensor(
        modis_processed
    )

    viirs_daily = aggregate_sensor(
        viirs_processed
    )

    print(
        f"MODIS daily-sector rows: "
        f"{len(modis_daily):,}"
    )

    print(
        f"VIIRS daily-sector rows: "
        f"{len(viirs_daily):,}"
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    save_daily_sensor(
        modis_daily,
        "MODIS"
    )

    save_daily_sensor(
        viirs_daily,
        "VIIRS"
    )

    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    print_subsection(
        "CREATING COMMON DATE × SECTOR MATRIX"
    )

    common_matrix = create_common_matrix(
        modis_daily,
        viirs_daily,
        bounds=AOI,
        grid_size=GRID_SIZE,
    )

    common_matrix_file = (
        PROCESSED_DIR
        / "common_daily_matrix.csv"
    )

    common_matrix.to_csv(
        common_matrix_file,
        index=False
    )

    print(
        f"Common matrix rows: "
        f"{len(common_matrix):,}"
    )

    print(
        f"Saved: {common_matrix_file}"
    )

    # --------------------------------------------------------
    # STEP 8
    # --------------------------------------------------------

    print_subsection(
        "CALCULATING HARMONIZED ACTIVITY"
    )

    result = calculate_harmonization(
        common_matrix,
        modis_weight=MODIS_WEIGHT,
        viirs_weight=VIIRS_WEIGHT,
    )

    # --------------------------------------------------------
    # STEP 9
    # --------------------------------------------------------

    final_file = save_final_output(
        result
    )

    # --------------------------------------------------------
    # STEP 10
    # --------------------------------------------------------

    write_summary(
        modis_raw=modis_raw,
        viirs_raw=viirs_raw,
        modis_processed=modis_processed,
        viirs_processed=viirs_processed,
        modis_daily=modis_daily,
        viirs_daily=viirs_daily,
        result=result,
    )

    # --------------------------------------------------------
    # STEP 11
    # --------------------------------------------------------

    display_result_sample(
        result
    )

    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MILESTONE 1 COMPLETE")
    print("=" * 70)

    print("\nFiles created:")

    print(
        f"  RAW:"
        f"\n    {MODIS_FILE}"
        f"\n    {VIIRS_FILE}"
    )

    print(
        "\n  PROCESSED:"
    )

    print(
        f"    {PROCESSED_DIR / 'modis_processed.csv'}"
    )

    print(
        f"    {PROCESSED_DIR / 'viirs_processed.csv'}"
    )

    print(
        f"    {PROCESSED_DIR / 'modis_daily.csv'}"
    )

    print(
        f"    {PROCESSED_DIR / 'viirs_daily.csv'}"
    )

    print(
        f"    {PROCESSED_DIR / 'common_daily_matrix.csv'}"
    )

    print(
        "\n  OUTPUT:"
    )

    print(
        f"    {final_file}"
    )

    print(
        f"    {OUTPUT_DIR / 'milestone_1_summary.txt'}"
    )

    print("\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()