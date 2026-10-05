"""
STEP 2 — PREPARE PROCESSED DAILY CHUNKS
============================================================

Purpose:
    Convert already-downloaded NASA FIRMS raw CSV chunks into
    daily sector-level data that can be read DIRECTLY by
    Milestone 3 build_science.py.

IMPORTANT:
    - NO NASA API calls
    - NO downloading
    - Uses only files already present in data/raw/
    - Safe to run repeatedly
    - Correctly skips chunks that are already processed
    - Automatically rebuilds old/incompatible _daily.csv files

EXPECTED OUTPUT SCHEMA:
    acq_date
    sector
    detection_count
    total_frp

This schema is required by build_science.py:

    combine_daily_chunks(source)

which expects:
    acq_date
    sector
    detection_count
    total_frp
============================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Sahd 2026 (Class 11-A)\General\The Horizon Watchers"
)

RAW_DIR = BASE_DIR / "data" / "raw"

PROCESSED_CHUNKS_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "chunks"
)


# ============================================================
# 2. GRID CONFIGURATION
# ============================================================

GRID_SIZE = 5

# Amazon Basin bounding box
MIN_LON = -82.0
MAX_LON = -44.0

MIN_LAT = -20.0
MAX_LAT = 10.0


# ============================================================
# 3. SOURCES
# ============================================================

SOURCES = [
    "MODIS_SP",
    "VIIRS_SNPP_SP",
]


# ============================================================
# 4. REQUIRED OUTPUT COLUMNS
# ============================================================

REQUIRED_OUTPUT_COLUMNS = [
    "acq_date",
    "sector",
    "detection_count",
    "total_frp",
]


# ============================================================
# 5. COLUMN DETECTION HELPERS
# ============================================================

def find_column(df, candidates):
    """
    Find the first matching column from a list of candidates.

    Matching is case-insensitive.
    """

    lookup = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in lookup:
            return lookup[key]

    return None


def find_latitude_column(df):
    """
    Find latitude column used by FIRMS data.
    """

    return find_column(
        df,
        [
            "latitude",
            "lat",
        ]
    )


def find_longitude_column(df):
    """
    Find longitude column used by FIRMS data.
    """

    return find_column(
        df,
        [
            "longitude",
            "lon",
            "long",
        ]
    )


def find_date_column(df):
    """
    Find acquisition date column.

    FIRMS files can use slightly different names.
    """

    return find_column(
        df,
        [
            "acq_date",
            "acquisition_date",
            "date",
        ]
    )


def find_frp_column(df):
    """
    Find Fire Radiative Power column.
    """

    return find_column(
        df,
        [
            "frp",
            "fire_radiative_power",
        ]
    )


# ============================================================
# 6. CHECK WHETHER OUTPUT IS ALREADY COMPATIBLE
# ============================================================

def output_is_compatible(output_path):
    """
    Check whether an existing _daily.csv already has the
    exact schema required by build_science.py.

    This prevents an old Step 2 file from being incorrectly
    skipped.
    """

    if not output_path.exists():
        return False

    try:

        header = pd.read_csv(
            output_path,
            nrows=0
        )

        columns = list(header.columns)

        return all(
            column in columns
            for column in REQUIRED_OUTPUT_COLUMNS
        )

    except Exception as exc:

        print(
            f"  ⚠ Could not validate existing output: {exc}"
        )

        return False


# ============================================================
# 7. FIND RAW CSV FILES
# ============================================================

def find_raw_csv_files(source):
    """
    Recursively locate raw CSV chunks for a source.

    Excludes:
        - modis.csv
        - viirs.csv
        - *_daily.csv

    because these are not raw chunks for this preparation step.
    """

    source_dir = RAW_DIR / source

    if not source_dir.exists():

        print(
            f"  ⚠ Raw directory does not exist:"
            f"\n    {source_dir}"
        )

        return []

    csv_files = []

    for path in source_dir.rglob("*.csv"):

        name = path.name.lower()

        # Ignore old combined/raw summary files
        if name == "modis.csv":
            continue

        if name == "viirs.csv":
            continue

        # Never treat already-created processed daily files
        # as raw input.
        if "_daily.csv" in name:
            continue

        csv_files.append(path)

    return sorted(csv_files)


# ============================================================
# 8. CALCULATE GRID SECTOR
# ============================================================

def calculate_grid_sector(lat, lon):
    """
    Convert latitude/longitude into a 5x5 sector.

    Sector numbering:

        C01 C02 C03 C04 C05
        C06 C07 C08 C09 C10
        C11 C12 C13 C14 C15
        C16 C17 C18 C19 C20
        C21 C22 C23 C24 C25

    Rows run north -> south.
    Columns run west -> east.
    """

    # Longitude -> column
    lon_fraction = (
        (lon - MIN_LON)
        / (MAX_LON - MIN_LON)
    )

    col = int(
        np.floor(
            lon_fraction * GRID_SIZE
        )
    )

    # Latitude -> row
    #
    # North edge = row 0
    # South edge = row 4
    lat_fraction = (
        (MAX_LAT - lat)
        / (MAX_LAT - MIN_LAT)
    )

    row = int(
        np.floor(
            lat_fraction * GRID_SIZE
        )
    )

    # Protect the maximum boundary.
    #
    # For example, lon == MAX_LON would mathematically
    # produce col == 5, which is outside a 0-4 grid.
    col = max(
        0,
        min(GRID_SIZE - 1, col)
    )

    row = max(
        0,
        min(GRID_SIZE - 1, row)
    )

    sector_number = (
        row * GRID_SIZE
        + col
        + 1
    )

    sector = f"C{sector_number:02d}"

    return row, col, sector


# ============================================================
# 9. PROCESS ONE RAW CHUNK
# ============================================================

def prepare_raw_chunk(raw_path, output_path, source):
    """
    Read one raw FIRMS CSV and convert it into the exact
    daily sector schema required by build_science.py.
    """

    print()
    print(
        f"  Processing:"
        f"\n    {raw_path.name}"
    )

    # --------------------------------------------------------
    # Read raw CSV
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            raw_path,
            low_memory=False
        )

    except Exception as exc:

        print(
            f"  ❌ Could not read CSV:"
            f"\n    {exc}"
        )

        return False

    if df.empty:

        print("  ⚠ Raw file is empty.")

        return False


    # --------------------------------------------------------
    # Find important columns
    # --------------------------------------------------------

    lat_col = find_latitude_column(df)
    lon_col = find_longitude_column(df)
    date_col = find_date_column(df)
    frp_col = find_frp_column(df)


    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    missing = []

    if lat_col is None:
        missing.append("latitude")

    if lon_col is None:
        missing.append("longitude")

    if date_col is None:
        missing.append("acq_date/date")

    if frp_col is None:
        missing.append("frp")


    if missing:

        print(
            "  ❌ Missing required columns:"
        )

        for item in missing:
            print(
                f"     - {item}"
            )

        print(
            "  Available columns:"
        )

        print(
            "    "
            + ", ".join(
                str(col)
                for col in df.columns
            )
        )

        return False


    # --------------------------------------------------------
    # Keep only required information
    # --------------------------------------------------------

    work = df[
        [
            lat_col,
            lon_col,
            date_col,
            frp_col,
        ]
    ].copy()

    work.columns = [
        "latitude",
        "longitude",
        "acq_date",
        "frp",
    ]


    # --------------------------------------------------------
    # Clean numeric values
    # --------------------------------------------------------

    work["latitude"] = pd.to_numeric(
        work["latitude"],
        errors="coerce"
    )

    work["longitude"] = pd.to_numeric(
        work["longitude"],
        errors="coerce"
    )

    work["frp"] = pd.to_numeric(
        work["frp"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Clean dates
    # --------------------------------------------------------

    work["acq_date"] = pd.to_datetime(
        work["acq_date"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    before_cleaning = len(work)

    work = work.dropna(
        subset=[
            "latitude",
            "longitude",
            "acq_date",
            "frp",
        ]
    ).copy()

    removed = (
        before_cleaning
        - len(work)
    )

    if removed > 0:

        print(
            f"  ℹ Removed {removed:,}"
            f" invalid rows."
        )


    # --------------------------------------------------------
    # Filter to Amazon Basin bounding box
    # --------------------------------------------------------

    work = work[
        (work["latitude"] >= MIN_LAT)
        & (work["latitude"] <= MAX_LAT)
        & (work["longitude"] >= MIN_LON)
        & (work["longitude"] <= MAX_LON)
    ].copy()


    if work.empty:

        print(
            "  ⚠ No detections inside the "
            "Amazon Basin bounding box."
        )

        # Still create a valid empty output file so the
        # chunk is known to have been processed.

        empty_output = pd.DataFrame(
            columns=REQUIRED_OUTPUT_COLUMNS
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        empty_output.to_csv(
            output_path,
            index=False
        )

        print(
            f"  ✅ Empty processed chunk written:"
            f"\n    {output_path}"
        )

        return True


    # --------------------------------------------------------
    # Convert coordinates into sectors
    # --------------------------------------------------------

    grid_values = [
        calculate_grid_sector(
            lat,
            lon
        )
        for lat, lon
        in zip(
            work["latitude"],
            work["longitude"]
        )
    ]

    work["row"] = [
        value[0]
        for value in grid_values
    ]

    work["col"] = [
        value[1]
        for value in grid_values
    ]

    work["sector"] = [
        value[2]
        for value in grid_values
    ]


    # --------------------------------------------------------
    # Normalize acquisition date
    # --------------------------------------------------------

    work["acq_date"] = (
        work["acq_date"]
        .dt.normalize()
    )


    # --------------------------------------------------------
    # DAILY SECTOR AGGREGATION
    # --------------------------------------------------------
    #
    # This is the critical output expected by
    # build_science.py.
    #
    # One row =
    #
    #   one acquisition date
    #   one sector
    #
    # with:
    #
    #   detection_count
    #   total_frp
    #
    # --------------------------------------------------------

    daily = (
        work
        .groupby(
            [
                "acq_date",
                "sector",
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
        )
    )


    # --------------------------------------------------------
    # Ensure correct numeric types
    # --------------------------------------------------------

    daily["detection_count"] = (
        pd.to_numeric(
            daily["detection_count"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    daily["total_frp"] = (
        pd.to_numeric(
            daily["total_frp"],
            errors="coerce"
        )
        .fillna(0.0)
    )


    # --------------------------------------------------------
    # Convert date to clean ISO format
    # --------------------------------------------------------
    #
    # build_science.py will later do:
    #
    # pd.to_datetime(combined["acq_date"])
    #
    # YYYY-MM-DD is therefore safe and unambiguous.
    # --------------------------------------------------------

    daily["acq_date"] = (
        daily["acq_date"]
        .dt.strftime("%Y-%m-%d")
    )


    # --------------------------------------------------------
    # Keep EXACTLY the columns expected by build_science.py
    # --------------------------------------------------------

    daily = daily[
        REQUIRED_OUTPUT_COLUMNS
    ]


    # --------------------------------------------------------
    # Sort output
    # --------------------------------------------------------

    daily = daily.sort_values(
        [
            "acq_date",
            "sector",
        ]
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # Write output
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    daily.to_csv(
        output_path,
        index=False
    )


    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print(
        f"  ✅ Prepared:"
        f"\n    {output_path}"
    )

    print(
        f"     Rows: {len(daily):,}"
    )

    print(
        f"     Dates: "
        f"{daily['acq_date'].nunique():,}"
    )

    print(
        f"     Sectors: "
        f"{daily['sector'].nunique():,}"
    )

    print(
        f"     Total detections: "
        f"{daily['detection_count'].sum():,}"
    )

    print(
        f"     Total FRP: "
        f"{daily['total_frp'].sum():,.2f}"
    )

    return True


# ============================================================
# 10. PROCESS ONE SOURCE
# ============================================================

def process_source(source):
    """
    Prepare all raw chunks belonging to one sensor.
    """

    print()
    print("=" * 72)
    print(
        f"PREPARING {source}"
    )
    print("=" * 72)


    raw_files = find_raw_csv_files(
        source
    )


    if not raw_files:

        print(
            f"  ⚠ No raw CSV chunks found for {source}."
        )

        return {
            "found": 0,
            "processed": 0,
            "skipped": 0,
            "rebuilt": 0,
            "failed": 0,
        }


    print(
        f"  Found {len(raw_files):,} raw CSV chunk(s)."
    )


    stats = {
        "found": len(raw_files),
        "processed": 0,
        "skipped": 0,
        "rebuilt": 0,
        "failed": 0,
    }


    # --------------------------------------------------------
    # Process each raw file
    # --------------------------------------------------------

    for raw_path in raw_files:

        # ----------------------------------------------------
        # Determine year
        # ----------------------------------------------------

        try:

            # Prefer year from path.
            #
            # Example:
            # raw/MODIS_SP/2024/file.csv
            #
            year = None

            for parent in raw_path.parents:

                if parent.name.isdigit():

                    possible_year = int(
                        parent.name
                    )

                    if (
                        2000
                        <= possible_year
                        <= 2100
                    ):

                        year = str(
                            possible_year
                        )

                        break

            # If no year directory exists, use the
            # file's data later. For normal FIRMS
            # directory structure this should not happen.

            if year is None:

                year = "unknown"

        except Exception:

            year = "unknown"


        # ----------------------------------------------------
        # Output filename
        # ----------------------------------------------------

        output_dir = (
            PROCESSED_CHUNKS_DIR
            / source
            / year
        )

        output_path = (
            output_dir
            / f"{raw_path.stem}_daily.csv"
        )


        # ----------------------------------------------------
        # EXISTING OUTPUT CHECK
        # ----------------------------------------------------
        #
        # Important:
        #
        # We do NOT blindly skip every existing file.
        #
        # If an old Step 2 script created:
        #
        #   date
        #   fire_count
        #
        # instead of:
        #
        #   acq_date
        #   detection_count
        #
        # it will be rebuilt automatically.
        # ----------------------------------------------------

        if output_path.exists():

            if output_is_compatible(
                output_path
            ):

                print()
                print(
                    f"  ⏭ Already correctly processed:"
                    f"\n    {raw_path.name}"
                )

                stats["skipped"] += 1

                continue

            else:

                print()
                print(
                    f"  🔄 Existing output has an "
                    f"incompatible schema:"
                    f"\n    {output_path}"
                )

                print(
                    "     Rebuilding from raw chunk..."
                )

                stats["rebuilt"] += 1


        # ----------------------------------------------------
        # Process chunk
        # ----------------------------------------------------

        success = prepare_raw_chunk(
            raw_path=raw_path,
            output_path=output_path,
            source=source,
        )


        if success:

            stats["processed"] += 1

        else:

            stats["failed"] += 1


    return stats


# ============================================================
# 11. VALIDATE ALL PROCESSED CHUNKS
# ============================================================

def validate_processed_chunks():
    """
    Verify that all generated _daily.csv files have the exact
    schema expected by build_science.py.
    """

    print()
    print("=" * 72)
    print(
        "VALIDATING PROCESSED CHUNKS"
    )
    print("=" * 72)


    total_files = 0
    valid_files = 0
    invalid_files = 0


    for source in SOURCES:

        source_dir = (
            PROCESSED_CHUNKS_DIR
            / source
        )

        if not source_dir.exists():
            continue


        files = sorted(
            source_dir.rglob(
                "*_daily.csv"
            )
        )


        for path in files:

            total_files += 1

            try:

                df = pd.read_csv(
                    path,
                    nrows=5
                )

                columns = list(
                    df.columns
                )


                missing = [
                    column
                    for column
                    in REQUIRED_OUTPUT_COLUMNS
                    if column not in columns
                ]


                if missing:

                    print()
                    print(
                        f"  ❌ INVALID:"
                        f"\n    {path}"
                    )

                    print(
                        f"     Missing: "
                        f"{', '.join(missing)}"
                    )

                    invalid_files += 1

                    continue


                valid_files += 1


            except Exception as exc:

                print()
                print(
                    f"  ❌ Could not read:"
                    f"\n    {path}"
                )

                print(
                    f"     {exc}"
                )

                invalid_files += 1


    print()
    print(
        f"  Total processed files: {total_files:,}"
    )

    print(
        f"  Valid files:           {valid_files:,}"
    )

    print(
        f"  Invalid files:         {invalid_files:,}"
    )


    return invalid_files == 0


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "STEP 2 — PREPARE DAILY FIRMS CHUNKS"
    )
    print("=" * 72)

    print()
    print(
        "Purpose:"
    )

    print(
        "  Convert existing raw FIRMS chunks into"
        " build_science.py-compatible daily data."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  No NASA API requests will be made."
    )

    print(
        "  No raw files will be downloaded."
    )

    print()
    print(
        f"Base directory:"
        f"\n  {BASE_DIR}"
    )

    print()
    print(
        f"Raw directory:"
        f"\n  {RAW_DIR}"
    )

    print()
    print(
        f"Processed directory:"
        f"\n  {PROCESSED_CHUNKS_DIR}"
    )

    print()
    print(
        "Output schema:"
    )

    for column in REQUIRED_OUTPUT_COLUMNS:

        print(
            f"  - {column}"
        )


    # --------------------------------------------------------
    # Verify directories
    # --------------------------------------------------------

    if not BASE_DIR.exists():

        print()
        print(
            "❌ BASE_DIR does not exist:"
        )

        print(
            f"   {BASE_DIR}"
        )

        return


    if not RAW_DIR.exists():

        print()
        print(
            "❌ RAW_DIR does not exist:"
        )

        print(
            f"   {RAW_DIR}"
        )

        return


    # --------------------------------------------------------
    # Process both sensors
    # --------------------------------------------------------

    all_stats = {}


    for source in SOURCES:

        all_stats[source] = process_source(
            source
        )


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    valid = validate_processed_chunks()


    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "STEP 2 COMPLETE"
    )
    print("=" * 72)


    for source in SOURCES:

        stats = all_stats[source]

        print()
        print(
            source
        )

        print(
            f"  Raw chunks found : "
            f"{stats['found']:,}"
        )

        print(
            f"  Newly processed  : "
            f"{stats['processed']:,}"
        )

        print(
            f"  Already skipped  : "
            f"{stats['skipped']:,}"
        )

        print(
            f"  Rebuilt old files: "
            f"{stats['rebuilt']:,}"
        )

        print(
            f"  Failed           : "
            f"{stats['failed']:,}"
        )


    print()

    if valid:

        print(
            "✅ ALL PROCESSED CHUNKS ARE "
            "BUILD_SCIENCE.PY COMPATIBLE."
        )

    else:

        print(
            "⚠ Some processed chunks have "
            "schema problems."
        )


    print()
    print(
        "The files are now ready for:"
    )

    print(
        "  build_science.py"
    )

    print()
    print(
        "Expected columns:"
    )

    print(
        "  acq_date | sector | detection_count | total_frp"
    )

    print()
    print(
        "=" * 72
    )


# ============================================================
# 13. RUN
# ============================================================

if __name__ == "__main__":
    main()