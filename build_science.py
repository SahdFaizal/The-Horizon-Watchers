"""
FIRE HARMONIZER - MILESTONE 3
==============================================

FULL AMAZON BASIN HISTORICAL SENSOR ENGINE

This version combines:

1. NASA FIRMS downloading
2. Amazon Basin data acquisition
3. MODIS + VIIRS S-NPP
4. Raw-data storage
5. Chunk-by-chunk processing
6. 5x5 sectorization
7. Daily aggregation
8. Common MODIS/VIIRS matrix
9. Historical sensor baselines
10. Harmonized Activity Index
11. Sensor discrepancy
12. Historical anomaly score
13. Severity classification

IMPORTANT:

- This script DOES download data.
- It does NOT download the whole world.
- It downloads ONLY the Amazon Basin study bounding box.
- It processes data in small chunks.
- It does NOT keep all raw satellite observations in RAM.
- It automatically detects the latest common standard-processing
  date available for MODIS and VIIRS S-NPP.
- It can resume from previously downloaded chunks.
- Standard Processing products are used for scientific analysis.

CORE DATA FLOW:

NASA FIRMS
     |
     +---- MODIS_SP
     |
     +---- VIIRS_SNPP_SP
              |
              v
       Amazon Basin BBOX
              |
              v
       5-day API chunks
              |
              v
          RAW CSV
              |
              v
       Clean + Sectorize
              |
              v
       Daily aggregation
              |
              v
      Common Date x Sector
              |
              v
      Historical baseline
              |
              v
       MODIS Z / VIIRS Z
              |
              v
       Harmonized Activity
              |
         +----+----+
         |         |
         v         v
    Anomaly      FRP
         |
         v
      Severity

NO:
    Flask
    Streamlit
    Lovable
    JavaScript
    Machine Learning
    Maps
    3D Earth
    Database

Those come later.
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
from io import StringIO
from datetime import datetime, timedelta
import json
import os
import time

import numpy as np
import pandas as pd
import requests


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"

PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = BASE_DIR / "output"

CHUNK_PROCESSED_DIR = (
    PROCESSED_DIR / "chunks"
)


# Create folders
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

CHUNK_PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. NASA FIRMS SETTINGS
# ============================================================

MAP_KEY = os.getenv(
    "FIRMS_MAP_KEY"
)

if not MAP_KEY:

    raise RuntimeError(
        "\nFIRMS_MAP_KEY is not set.\n\n"
        "Windows PowerShell:\n"
        '    $env:FIRMS_MAP_KEY="YOUR_KEY"\n\n'
        "Then run the script again."
    )


FIRMS_BASE_URL = (
    "https://firms.modaps.eosdis.nasa.gov"
)


# ============================================================
# 3. AMAZON BASIN STUDY REGION
# ============================================================

"""
IMPORTANT:

This is an Amazon Basin STUDY BOUNDING BOX.

It intentionally encloses the Amazon Basin but also contains
some surrounding land because a rectangular bounding box cannot
follow the exact hydrographic basin boundary.

Exact polygon filtering can be introduced later.

Format:

    west, south, east, north
"""

AMAZON_BASIN_BBOX = (
    -82.0,   # west
    -20.0,   # south
    -44.0,   # east
    10.0     # north
)


# ============================================================
# 4. HISTORICAL PERIOD
# ============================================================

"""
VIIRS S-NPP standard processing begins:

    2012-01-20

Therefore the common MODIS + VIIRS period begins there.

END_DATE = None means:

    automatically use the latest date for which BOTH
    standard-processing products are available.
"""

START_DATE = "2012-01-20"

END_DATE = None


# ============================================================
# 5. SENSOR SOURCES
# ============================================================

MODIS_SOURCE = (
    "MODIS_SP"
)

VIIRS_SOURCE = (
    "VIIRS_SNPP_SP"
)


# ============================================================
# 6. DOWNLOAD SETTINGS
# ============================================================

"""
NASA FIRMS Area API allows 1-5 days per request.

Use the maximum 5-day chunk to minimize requests.
"""

CHUNK_DAYS = 5

REQUEST_TIMEOUT = 180

MAX_RETRIES = 4

RETRY_DELAY_SECONDS = 5

REQUEST_DELAY_SECONDS = 0.15


# ============================================================
# 7. GRID SETTINGS
# ============================================================

GRID_SIZE = 5


# ============================================================
# 8. HARMONIZATION SETTINGS
# ============================================================

MODIS_WEIGHT = 0.5

VIIRS_WEIGHT = 0.5

MIN_BASELINE_SAMPLES = 20


# ============================================================
# 9. CHECKPOINT
# ============================================================

MANIFEST_FILE = (
    PROCESSED_DIR
    / "download_manifest.json"
)


def load_manifest():
    """
    Load completed chunk information.
    """

    if not MANIFEST_FILE.exists():

        return {}

    try:

        with open(
            MANIFEST_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError
    ):

        return {}


def save_manifest(
    manifest
):

    with open(
        MANIFEST_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2
        )


# ============================================================
# 10. PRINT HELPERS
# ============================================================

def section(title):

    print("\n")
    print("=" * 75)
    print(title)
    print("=" * 75)


def subsection(title):

    print("\n")
    print("-" * 75)
    print(title)
    print("-" * 75)


# ============================================================
# 11. GET FIRMS DATA AVAILABILITY
# ============================================================

def get_data_availability():

    """
    Ask FIRMS which date ranges are currently available
    for our two standard-processing products.
    """

    url = (
        FIRMS_BASE_URL
        + "/api/data_availability/csv/"
        + MAP_KEY
        + "/ALL"
    )

    print(
        "\nChecking NASA FIRMS data availability..."
    )

    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    availability = pd.read_csv(
        StringIO(
            response.text
        )
    )

    required_columns = [
        "data_id",
        "min_date",
        "max_date"
    ]

    missing = [
        column
        for column in required_columns
        if column not in availability.columns
    ]

    if missing:

        raise ValueError(
            "FIRMS availability response is missing: "
            + str(missing)
        )

    return availability


# ============================================================
# 12. DETERMINE COMMON DATE RANGE
# ============================================================

def determine_date_range():

    availability = (
        get_data_availability()
    )

    wanted_sources = [
        MODIS_SOURCE,
        VIIRS_SOURCE
    ]

    rows = availability[
        availability["data_id"]
        .isin(wanted_sources)
    ].copy()

    if len(rows) != 2:

        raise RuntimeError(
            "Could not find both required FIRMS "
            "standard-processing products."
        )

    rows["min_date"] = pd.to_datetime(
        rows["min_date"]
    )

    rows["max_date"] = pd.to_datetime(
        rows["max_date"]
    )

    common_start = rows[
        "min_date"
    ].max()

    common_end = rows[
        "max_date"
    ].min()

    requested_start = pd.Timestamp(
        START_DATE
    )

    start = max(
        requested_start,
        common_start
    )

    if END_DATE is None:

        end = common_end

    else:

        requested_end = pd.Timestamp(
            END_DATE
        )

        end = min(
            requested_end,
            common_end
        )

    if start > end:

        raise RuntimeError(
            "No overlapping MODIS + VIIRS date range."
        )

    print(
        "\nCommon scientific date range:"
    )

    print(
        f"  Start: {start.date()}"
    )

    print(
        f"  End:   {end.date()}"
    )

    print(
        "\nMODIS available:"
    )

    modis_row = rows[
        rows["data_id"]
        == MODIS_SOURCE
    ].iloc[0]

    print(
        f"  {modis_row['min_date'].date()}"
        f" → "
        f"{modis_row['max_date'].date()}"
    )

    print(
        "\nVIIRS S-NPP available:"
    )

    viirs_row = rows[
        rows["data_id"]
        == VIIRS_SOURCE
    ].iloc[0]

    print(
        f"  {viirs_row['min_date'].date()}"
        f" → "
        f"{viirs_row['max_date'].date()}"
    )

    return (
        start.date(),
        end.date()
    )


# ============================================================
# 13. CHUNK GENERATOR
# ============================================================

def generate_chunks(
    start,
    end,
    chunk_days=5
):

    current = start

    while current <= end:

        remaining = (
            end - current
        ).days + 1

        days = min(
            chunk_days,
            remaining
        )

        yield (
            current,
            days
        )

        current += timedelta(
            days=days
        )


# ============================================================
# 14. RAW FILE PATH
# ============================================================

def raw_path(
    source,
    start,
    days
):

    year_dir = (
        RAW_DIR
        / source
        / str(start.year)
    )

    year_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return (
        year_dir
        /
        f"{source}_{start.isoformat()}_{days}d.csv"
    )


# ============================================================
# 15. PROCESSED CHUNK PATH
# ============================================================

def processed_chunk_path(
    source,
    start,
    days
):

    year_dir = (
        CHUNK_PROCESSED_DIR
        / source
        / str(start.year)
    )

    year_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return (
        year_dir
        /
        f"{source}_{start.isoformat()}_{days}d_daily.csv"
    )


# ============================================================
# 16. DOWNLOAD ONE FIRMS CHUNK
# ============================================================

def download_chunk(
    source,
    start,
    days
):

    west, south, east, north = (
        AMAZON_BASIN_BBOX
    )

    area = (
        f"{west},{south},"
        f"{east},{north}"
    )

    date_string = (
        start.isoformat()
    )

    url = (
        FIRMS_BASE_URL
        + "/api/area/csv/"
        + MAP_KEY
        + "/"
        + source
        + "/"
        + area
        + "/"
        + str(days)
        + "/"
        + date_string
    )

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            print(
                f"Downloading {source}"
            )

            print(
                f"  {date_string}"
                f" → "
                f"{(start + timedelta(days=days-1)).isoformat()}"
            )

            response = requests.get(
                url,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            text = response.text.strip()

            if not text:

                raise RuntimeError(
                    "FIRMS returned an empty response."
                )

            return pd.read_csv(
                StringIO(text)
            )

        except Exception as error:

            print(
                f"  Attempt {attempt}/"
                f"{MAX_RETRIES} failed:"
            )

            print(
                f"  {error}"
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY_SECONDS
                    * attempt
                )

            else:

                raise


# ============================================================
# 17. STANDARDIZE COLUMNS
# ============================================================

def standardize_columns(
    df
):

    df = df.copy()

    df.columns = [
        str(column)
        .strip()
        .lower()
        for column in df.columns
    ]

    return df


# ============================================================
# 18. PREPARE SENSOR
# ============================================================

def prepare_sensor(
    df,
    sensor_name,
):

    df = standardize_columns(
        df
    )

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
            f"{sensor_name} missing required columns: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # Convert types
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
    # Remove invalid
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
            "acq_date"
        ]
    ).copy()

    # --------------------------------------------------------
    # Add sensor
    # --------------------------------------------------------

    df["sensor"] = sensor_name

    # --------------------------------------------------------
    # AOI
    # --------------------------------------------------------

    west, south, east, north = (
        AMAZON_BASIN_BBOX
    )

    df = df[
        (df["longitude"] >= west)
        &
        (df["longitude"] <= east)
        &
        (df["latitude"] >= south)
        &
        (df["latitude"] <= north)
    ].copy()

    # --------------------------------------------------------
    # 5 x 5 sectors
    # --------------------------------------------------------

    x = (
        (df["longitude"] - west)
        /
        (east - west)
    )

    y = (
        (df["latitude"] - south)
        /
        (north - south)
    )

    df["col"] = np.floor(
        x * GRID_SIZE
    ).astype(int)

    df["row"] = np.floor(
        y * GRID_SIZE
    ).astype(int)

    # Boundary protection
    df["col"] = df[
        "col"
    ].clip(
        0,
        GRID_SIZE - 1
    )

    df["row"] = df[
        "row"
    ].clip(
        0,
        GRID_SIZE - 1
    )

    sector_number = (
        df["row"] * GRID_SIZE
        + df["col"]
        + 1
    )

    df["sector"] = (
        "C"
        +
        sector_number
        .astype(str)
        .str.zfill(2)
    )

    return df


# ============================================================
# 19. AGGREGATE CHUNK
# ============================================================

def aggregate_sensor(
    df
):

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
# 20. PROCESS ONE CHUNK
# ============================================================

def process_chunk(
    source,
    start,
    days,
    manifest
):

    key = (
        f"{source}|"
        f"{start.isoformat()}|"
        f"{days}"
    )

    raw_file = raw_path(
        source,
        start,
        days
    )

    processed_file = (
        processed_chunk_path(
            source,
            start,
            days
        )
    )

    # --------------------------------------------------------
    # Already completed?
    # --------------------------------------------------------

    if (
        key in manifest
        and processed_file.exists()
    ):

        print(
            f"\nSKIPPING COMPLETED:"
            f"\n  {key}"
        )

        return

    # --------------------------------------------------------
    # Reuse raw file if already downloaded
    # --------------------------------------------------------

    if raw_file.exists():

        print(
            f"\nRaw file already exists:"
            f"\n  {raw_file}"
        )

        raw = pd.read_csv(
            raw_file
        )

    else:

        raw = download_chunk(
            source,
            start,
            days
        )

        raw.to_csv(
            raw_file,
            index=False
        )

        print(
            f"Saved raw:"
            f"\n  {raw_file}"
        )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    prepared = prepare_sensor(
        raw,
        source
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    daily = aggregate_sensor(
        prepared
    )

    # --------------------------------------------------------
    # Save processed chunk
    # --------------------------------------------------------

    daily.to_csv(
        processed_file,
        index=False
    )

    print(
        f"Saved processed:"
        f"\n  {processed_file}"
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    manifest[key] = {
        "source": source,
        "start": start.isoformat(),
        "days": days,
        "raw_file": str(raw_file),
        "processed_file": str(
            processed_file
        ),
        "raw_rows": int(
            len(raw)
        ),
        "processed_rows": int(
            len(daily)
        )
    }

    save_manifest(
        manifest
    )


# ============================================================
# 21. DOWNLOAD + PROCESS ALL YEARS
# ============================================================

def acquire_all_data(
    start,
    end,
):

    section(
        "STEP 1 — NASA FIRMS DATA ACQUISITION"
    )

    print(
        f"\nStudy region:"
    )

    print(
        f"  Amazon Basin bounding box:"
        f" {AMAZON_BASIN_BBOX}"
    )

    print(
        f"\nDate range:"
    )

    print(
        f"  {start} → {end}"
    )

    print(
        "\nSensors:"
    )

    print(
        f"  MODIS: {MODIS_SOURCE}"
    )

    print(
        f"  VIIRS: {VIIRS_SOURCE}"
    )

    manifest = load_manifest()

    chunks = list(
        generate_chunks(
            start,
            end,
            CHUNK_DAYS
        )
    )

    print(
        f"\n5-day chunks per sensor:"
        f" {len(chunks):,}"
    )

    print(
        f"Total API data chunks:"
        f" {len(chunks) * 2:,}"
    )

    # --------------------------------------------------------
    # MODIS
    # --------------------------------------------------------

    subsection(
        "MODIS"
    )

    for index, (
        chunk_start,
        days
    ) in enumerate(
        chunks,
        start=1
    ):

        print(
            f"\nMODIS chunk "
            f"{index}/{len(chunks)}"
        )

        process_chunk(
            MODIS_SOURCE,
            chunk_start,
            days,
            manifest
        )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    # --------------------------------------------------------
    # VIIRS
    # --------------------------------------------------------

    subsection(
        "VIIRS S-NPP"
    )

    for index, (
        chunk_start,
        days
    ) in enumerate(
        chunks,
        start=1
    ):

        print(
            f"\nVIIRS chunk "
            f"{index}/{len(chunks)}"
        )

        process_chunk(
            VIIRS_SOURCE,
            chunk_start,
            days,
            manifest
        )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    print(
        "\nAcquisition and chunk processing complete."
    )


# ============================================================
# 22. COMBINE PROCESSED CHUNKS
# ============================================================

def combine_daily_chunks(
    source
):

    section(
        f"STEP 2 — COMBINE {source} DAILY DATA"
    )

    source_dir = (
        CHUNK_PROCESSED_DIR
        / source
    )

    files = sorted(
        source_dir.rglob(
            "*_daily.csv"
        )
    )

    if not files:

        raise RuntimeError(
            f"No processed files found for "
            f"{source}."
        )

    print(
        f"Processed chunk files:"
        f" {len(files):,}"
    )

    frames = []

    for file in files:

        df = pd.read_csv(
            file
        )

        if df.empty:
            continue

        frames.append(
            df
        )

    if not frames:

        raise RuntimeError(
            f"No usable processed data for "
            f"{source}."
        )

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    combined["acq_date"] = (
        pd.to_datetime(
            combined["acq_date"]
        )
    )

    # --------------------------------------------------------
    # Reaggregate to protect against duplicate chunks
    # --------------------------------------------------------

    combined = (
        combined.groupby(
            [
                "acq_date",
                "sector"
            ],
            as_index=False
        )
        .agg(
            detection_count=(
                "detection_count",
                "sum"
            ),

            total_frp=(
                "total_frp",
                "sum"
            )
        )
    )

    if source == MODIS_SOURCE:

        combined = combined.rename(
            columns={
                "detection_count":
                    "modis_count",

                "total_frp":
                    "modis_frp"
            }
        )

        output_file = (
            PROCESSED_DIR
            / "modis_daily.csv"
        )

    else:

        combined = combined.rename(
            columns={
                "detection_count":
                    "viirs_count",

                "total_frp":
                    "viirs_frp"
            }
        )

        output_file = (
            PROCESSED_DIR
            / "viirs_daily.csv"
        )

    combined = combined.sort_values(
        [
            "acq_date",
            "sector"
        ]
    )

    combined.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved:"
        f"\n  {output_file}"
    )

    print(
        f"Rows:"
        f" {len(combined):,}"
    )

    print(
        f"Dates:"
        f" {combined['acq_date'].nunique():,}"
    )

    print(
        f"Sectors:"
        f" {combined['sector'].nunique():,}"
    )

    return combined


# ============================================================
# 23. CREATE COMMON MATRIX
# ============================================================

def create_common_matrix(
    modis_daily,
    viirs_daily,
    start,
    end
):

    section(
        "STEP 3 — CREATE COMMON DATE × SECTOR MATRIX"
    )

    dates = pd.date_range(
        start,
        end,
        freq="D"
    )

    sectors = [
        f"C{i:02d}"
        for i in range(
            1,
            GRID_SIZE * GRID_SIZE + 1
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

    # --------------------------------------------------------
    # Fill missing detection/FRP values
    # --------------------------------------------------------

    numeric_columns = [
        "modis_count",
        "viirs_count",
        "modis_frp",
        "viirs_frp",
    ]

    for column in numeric_columns:

        if column in result.columns:

            result[column] = (
                result[column]
                .fillna(0)
            )

    result = result.sort_values(
        [
            "acq_date",
            "sector"
        ]
    ).reset_index(
        drop=True
    )

    output_file = (
        PROCESSED_DIR
        / "common_daily_matrix.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved:"
        f"\n  {output_file}"
    )

    print(
        f"Rows:"
        f" {len(result):,}"
    )

    print(
        f"Expected:"
        f" {len(dates) * len(sectors):,}"
    )

    print(
        f"Dates:"
        f" {len(dates):,}"
    )

    print(
        f"Sectors:"
        f" {len(sectors):,}"
    )

    return result


# ============================================================
# 24. LEAVE-ONE-YEAR-OUT STATISTICS
# ============================================================

def calculate_loo_stats(
    df,
    value_column,
    group_columns
):

    """
    Calculate baseline statistics using
    all years EXCEPT the current row's year.

    This is used for retrospective historical
    comparison.
    """

    work = df[
        group_columns
        +
        [
            "year",
            value_column
        ]
    ].copy()

    # --------------------------------------------------------
    # All years
    # --------------------------------------------------------

    overall = (
        work.groupby(
            group_columns,
            dropna=False
        )[value_column]
        .agg(
            total_sum="sum",

            total_count="count",

            total_sumsq=lambda x:
                np.square(x).sum()
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Per year
    # --------------------------------------------------------

    yearly = (
        work.groupby(
            group_columns
            +
            ["year"],
            dropna=False
        )[value_column]
        .agg(
            year_sum="sum",

            year_count="count",

            year_sumsq=lambda x:
                np.square(x).sum()
        )
        .reset_index()
    )

    stats = yearly.merge(
        overall,
        on=group_columns,
        how="left"
    )

    # --------------------------------------------------------
    # Remove current year
    # --------------------------------------------------------

    stats["base_sum"] = (
        stats["total_sum"]
        -
        stats["year_sum"]
    )

    stats["base_count"] = (
        stats["total_count"]
        -
        stats["year_count"]
    )

    stats["base_sumsq"] = (
        stats["total_sumsq"]
        -
        stats["year_sumsq"]
    )

    # --------------------------------------------------------
    # Mean
    # --------------------------------------------------------

    stats["baseline_mean"] = (
        stats["base_sum"]
        /
        stats["base_count"]
        .replace(
            0,
            np.nan
        )
    )

    # --------------------------------------------------------
    # Variance
    # --------------------------------------------------------

    variance = (
        stats["base_sumsq"]
        -
        (
            stats["base_sum"] ** 2
            /
            stats["base_count"]
            .replace(
                0,
                np.nan
            )
        )
    )

    denominator = (
        stats["base_count"] - 1
    )

    stats["baseline_variance"] = (
        variance
        /
        denominator.replace(
            0,
            np.nan
        )
    ).clip(
        lower=0
    )

    stats["baseline_std"] = np.sqrt(
        stats["baseline_variance"]
    )

    return stats[
        group_columns
        +
        [
            "year",
            "baseline_mean",
            "baseline_std",
            "base_count"
        ]
    ]


# ============================================================
# 25. ADD SENSOR BASELINE
# ============================================================

def add_sensor_baseline(
    df,
    value_column,
    prefix
):

    result = df.copy()

    # --------------------------------------------------------
    # Primary baseline:
    # sector + month + leave-one-year-out
    # --------------------------------------------------------

    stats = calculate_loo_stats(
        result,
        value_column,
        [
            "sector",
            "month"
        ]
    )

    stats = stats.rename(
        columns={
            "baseline_mean":
                f"{prefix}_baseline_mean",

            "baseline_std":
                f"{prefix}_baseline_std",

            "base_count":
                f"{prefix}_baseline_n"
        }
    )

    result = result.merge(
        stats[
            [
                "sector",
                "month",
                "year",

                f"{prefix}_baseline_mean",
                f"{prefix}_baseline_std",
                f"{prefix}_baseline_n",
            ]
        ],
        on=[
            "sector",
            "month",
            "year"
        ],
        how="left"
    )

    result[
        f"{prefix}_baseline_level"
    ] = "sector-month-historical"

    # --------------------------------------------------------
    # Fallback: sector historical baseline
    # --------------------------------------------------------

    missing = (
        result[
            f"{prefix}_baseline_mean"
        ].isna()
        |
        result[
            f"{prefix}_baseline_std"
        ].isna()
    )

    if missing.any():

        fallback = calculate_loo_stats(
            result,
            value_column,
            ["sector"]
        )

        fallback = fallback.rename(
            columns={
                "baseline_mean":
                    "_mean",

                "baseline_std":
                    "_std",

                "base_count":
                    "_n"
            }
        )

        result = result.merge(
            fallback[
                [
                    "sector",
                    "year",
                    "_mean",
                    "_std",
                    "_n"
                ]
            ],
            on=[
                "sector",
                "year"
            ],
            how="left"
        )

        result.loc[
            missing,
            f"{prefix}_baseline_mean"
        ] = result.loc[
            missing,
            "_mean"
        ]

        result.loc[
            missing,
            f"{prefix}_baseline_std"
        ] = result.loc[
            missing,
            "_std"
        ]

        result.loc[
            missing,
            f"{prefix}_baseline_n"
        ] = result.loc[
            missing,
            "_n"
        ]

        result.loc[
            missing,
            f"{prefix}_baseline_level"
        ] = "sector-historical"

        result = result.drop(
            columns=[
                "_mean",
                "_std",
                "_n"
            ]
        )

    # --------------------------------------------------------
    # Final global fallback
    # --------------------------------------------------------

    global_mean = (
        result[value_column]
        .mean()
    )

    global_std = (
        result[value_column]
        .std()
    )

    if (
        pd.isna(global_std)
        or global_std == 0
    ):

        global_std = 1.0

    missing = (
        result[
            f"{prefix}_baseline_mean"
        ].isna()
        |
        result[
            f"{prefix}_baseline_std"
        ].isna()
    )

    result.loc[
        missing,
        f"{prefix}_baseline_mean"
    ] = global_mean

    result.loc[
        missing,
        f"{prefix}_baseline_std"
    ] = global_std

    result.loc[
        missing,
        f"{prefix}_baseline_n"
    ] = 0

    result.loc[
        missing,
        f"{prefix}_baseline_level"
    ] = "global-fallback"

    # --------------------------------------------------------
    # Protect zero standard deviations
    # --------------------------------------------------------

    result[
        f"{prefix}_baseline_std"
    ] = (
        result[
            f"{prefix}_baseline_std"
        ]
        .replace(
            0,
            1.0
        )
        .fillna(1.0)
    )

    return result


# ============================================================
# 26. SENSOR Z-SCORE
# ============================================================

def calculate_sensor_z(
    df,
    value_column,
    prefix
):

    df = df.copy()

    df[
        f"{prefix}_z"
    ] = (
        df[value_column]
        -
        df[
            f"{prefix}_baseline_mean"
        ]
    ) / df[
        f"{prefix}_baseline_std"
    ]

    return df


# ============================================================
# 27. CORE HARMONIZATION
# ============================================================

def calculate_harmonization(
    df
):

    if not np.isclose(
        MODIS_WEIGHT
        +
        VIIRS_WEIGHT,
        1.0
    ):

        raise ValueError(
            "MODIS_WEIGHT + VIIRS_WEIGHT "
            "must equal 1.0"
        )

    result = df.copy()

    # --------------------------------------------------------
    # Normalize MODIS
    # --------------------------------------------------------

    result = add_sensor_baseline(
        result,
        "modis_count",
        "modis"
    )

    result = calculate_sensor_z(
        result,
        "modis_count",
        "modis"
    )

    # --------------------------------------------------------
    # Normalize VIIRS
    # --------------------------------------------------------

    result = add_sensor_baseline(
        result,
        "viirs_count",
        "viirs"
    )

    result = calculate_sensor_z(
        result,
        "viirs_count",
        "viirs"
    )

    # --------------------------------------------------------
    # MAIN HARMONIZATION
    # --------------------------------------------------------

    result[
        "harmonized_activity"
    ] = (
        MODIS_WEIGHT
        *
        result["modis_z"]
        +
        VIIRS_WEIGHT
        *
        result["viirs_z"]
    )

    # --------------------------------------------------------
    # Sensor discrepancy
    # --------------------------------------------------------

    result[
        "sensor_discrepancy"
    ] = (
        result["modis_z"]
        -
        result["viirs_z"]
    ).abs()

    # --------------------------------------------------------
    # FRP remains independent
    # --------------------------------------------------------

    result[
        "total_frp"
    ] = (
        result["modis_frp"]
        +
        result["viirs_frp"]
    )

    # Log FRP for visualization later
    result[
        "frp_log"
    ] = np.log10(
        1
        +
        result["total_frp"].clip(
            lower=0
        )
    )

    return result


# ============================================================
# 28. HARMONIZED HISTORICAL BASELINE
# ============================================================

def add_harmonized_baseline(
    df
):

    stats = calculate_loo_stats(
        df,
        "harmonized_activity",
        [
            "sector",
            "month"
        ]
    )

    stats = stats.rename(
        columns={
            "baseline_mean":
                "harmonized_baseline_mean",

            "baseline_std":
                "harmonized_baseline_std",

            "base_count":
                "harmonized_baseline_n"
        }
    )

    result = df.merge(
        stats[
            [
                "sector",
                "month",
                "year",
                "harmonized_baseline_mean",
                "harmonized_baseline_std",
                "harmonized_baseline_n"
            ]
        ],
        on=[
            "sector",
            "month",
            "year"
        ],
        how="left"
    )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    missing = (
        result[
            "harmonized_baseline_mean"
        ].isna()
        |
        result[
            "harmonized_baseline_std"
        ].isna()
    )

    if missing.any():

        mean = (
            result[
                "harmonized_activity"
            ].mean()
        )

        std = (
            result[
                "harmonized_activity"
            ].std()
        )

        if pd.isna(std) or std == 0:

            std = 1.0

        result.loc[
            missing,
            "harmonized_baseline_mean"
        ] = mean

        result.loc[
            missing,
            "harmonized_baseline_std"
        ] = std

        result.loc[
            missing,
            "harmonized_baseline_n"
        ] = 0

    result[
        "harmonized_baseline_std"
    ] = (
        result[
            "harmonized_baseline_std"
        ]
        .replace(
            0,
            1.0
        )
        .fillna(1.0)
    )

    return result


# ============================================================
# 29. ANOMALY
# ============================================================

def calculate_anomaly(
    df
):

    result = df.copy()

    result[
        "anomaly_z"
    ] = (
        result[
            "harmonized_activity"
        ]
        -
        result[
            "harmonized_baseline_mean"
        ]
    ) / result[
        "harmonized_baseline_std"
    ]

    conditions = [
        result["anomaly_z"] < 1.0,

        (
            result["anomaly_z"] >= 1.0
        )
        &
        (
            result["anomaly_z"] < 2.0
        ),

        (
            result["anomaly_z"] >= 2.0
        )
        &
        (
            result["anomaly_z"] < 3.0
        ),

        result["anomaly_z"] >= 3.0
    ]

    choices = [
        "Normal",
        "Elevated",
        "Unusual",
        "Extreme"
    ]

    result[
        "severity"
    ] = np.select(
        conditions,
        choices,
        default="Normal"
    )

    return result


# ============================================================
# 30. SAVE SCIENCE OUTPUT
# ============================================================

def save_science_output(
    result,
    start,
    end
):

    # Full historical result
    all_file = (
        OUTPUT_DIR
        /
        "harmonized_science_all_years.csv"
    )

    result.to_csv(
        all_file,
        index=False
    )

    print(
        f"\nSaved all-years science file:"
        f"\n  {all_file}"
    )

    # Current/latest target year
    latest_year = (
        pd.Timestamp(end)
        .year
    )

    target = result[
        result["year"]
        ==
        latest_year
    ].copy()

    target_file = (
        OUTPUT_DIR
        /
        f"harmonized_science_{latest_year}.csv"
    )

    target.to_csv(
        target_file,
        index=False
    )

    print(
        f"Saved latest-year science file:"
        f"\n  {target_file}"
    )

    return all_file, target_file


# ============================================================
# 31. WRITE PROJECT SUMMARY
# ============================================================

def write_project_summary(
    result,
    start,
    end
):

    summary_file = (
        OUTPUT_DIR
        /
        "milestone_3_summary.txt"
    )

    years = sorted(
        result["year"].unique()
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "FIRE HARMONIZER - MILESTONE 3\n"
        )

        file.write(
            "====================================\n\n"
        )

        file.write(
            "STUDY REGION\n"
        )

        file.write(
            f"Amazon Basin bounding box: "
            f"{AMAZON_BASIN_BBOX}\n\n"
        )

        file.write(
            "DATA SOURCES\n"
        )

        file.write(
            f"MODIS: {MODIS_SOURCE}\n"
        )

        file.write(
            f"VIIRS: {VIIRS_SOURCE}\n\n"
        )

        file.write(
            "PERIOD\n"
        )

        file.write(
            f"Start: {start}\n"
        )

        file.write(
            f"End: {end}\n"
        )

        file.write(
            f"Years: {years}\n\n"
        )

        file.write(
            "GRID\n"
        )

        file.write(
            f"{GRID_SIZE} x {GRID_SIZE}\n"
        )

        file.write(
            f"Sectors: "
            f"{GRID_SIZE * GRID_SIZE}\n\n"
        )

        file.write(
            "HARMONIZATION\n"
        )

        file.write(
            f"MODIS weight: "
            f"{MODIS_WEIGHT}\n"
        )

        file.write(
            f"VIIRS weight: "
            f"{VIIRS_WEIGHT}\n\n"
        )

        file.write(
            "DATASET SIZE\n"
        )

        file.write(
            f"Rows: "
            f"{len(result):,}\n"
        )

        file.write(
            f"Dates: "
            f"{result['acq_date'].nunique():,}\n"
        )

        file.write(
            f"Sectors: "
            f"{result['sector'].nunique():,}\n\n"
        )

        file.write(
            "SEVERITY COUNTS\n"
        )

        file.write(
            result[
                "severity"
            ]
            .value_counts()
            .to_string()
        )

        file.write(
            "\n\n"
        )

        file.write(
            "NOTE\n"
        )

        file.write(
            "The Amazon region is represented by a "
            "bounding box, not an exact hydrographic "
            "polygon. The resulting sectors therefore "
            "represent sectors of the study bounding "
            "box.\n"
        )

    print(
        f"Saved:"
        f"\n  {summary_file}"
    )


# ============================================================
# 32. MAIN
# ============================================================

def main():

    section(
        "FIRE HARMONIZER - MILESTONE 3"
    )

    print(
        "\nFULL AMAZON BASIN HISTORICAL ENGINE"
    )

    print(
        "\nThis run will:"
    )

    print(
        "  1. Query FIRMS availability"
    )

    print(
        "  2. Download MODIS + VIIRS"
    )

    print(
        "  3. Process each 5-day chunk"
    )

    print(
        "  4. Save raw data"
    )

    print(
        "  5. Save processed data"
    )

    print(
        "  6. Build common matrix"
    )

    print(
        "  7. Calculate historical baselines"
    )

    print(
        "  8. Calculate Harmonized Activity"
    )

    print(
        "  9. Calculate anomalies"
    )

    # --------------------------------------------------------
    # Determine common period
    # --------------------------------------------------------

    start, end = (
        determine_date_range()
    )

    # --------------------------------------------------------
    # Acquire data
    # --------------------------------------------------------

    acquire_all_data(
        start,
        end
    )

    # --------------------------------------------------------
    # Combine daily products
    # --------------------------------------------------------

    modis_daily = (
        combine_daily_chunks(
            MODIS_SOURCE
        )
    )

    viirs_daily = (
        combine_daily_chunks(
            VIIRS_SOURCE
        )
    )

    # --------------------------------------------------------
    # Common matrix
    # --------------------------------------------------------

    common = create_common_matrix(
        modis_daily,
        viirs_daily,
        start,
        end
    )

    # --------------------------------------------------------
    # Add temporal fields
    # --------------------------------------------------------

    common["year"] = (
        common["acq_date"]
        .dt.year
    )

    common["month"] = (
        common["acq_date"]
        .dt.month
    )

    common["day_of_year"] = (
        common["acq_date"]
        .dt.dayofyear
    )

    # --------------------------------------------------------
    # Harmonization
    # --------------------------------------------------------

    section(
        "STEP 4 — HISTORICAL HARMONIZATION"
    )

    result = calculate_harmonization(
        common
    )

    # --------------------------------------------------------
    # Historical harmonized baseline
    # --------------------------------------------------------

    subsection(
        "Harmonized Historical Baseline"
    )

    result = add_harmonized_baseline(
        result
    )

    # --------------------------------------------------------
    # Anomaly
    # --------------------------------------------------------

    subsection(
        "Anomaly Detection"
    )

    result = calculate_anomaly(
        result
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_science_output(
        result,
        start,
        end
    )

    write_project_summary(
        result,
        start,
        end
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    section(
        "MILESTONE 3 COMPLETE"
    )

    print(
        "\nDate range:"
    )

    print(
        f"  {start} → {end}"
    )

    print(
        "\nYears:"
    )

    print(
        sorted(
            result[
                "year"
            ].unique()
        )
    )

    print(
        "\nRows:"
    )

    print(
        f"  {len(result):,}"
    )

    print(
        "\nSectors:"
    )

    print(
        f"  {result['sector'].nunique()}"
    )

    print(
        "\nSeverity:"
    )

    print(
        result[
            "severity"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nMain output:"
    )

    print(
        f"  {OUTPUT_DIR / 'harmonized_science_all_years.csv'}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()