"""
FIRE HARMONIZER - MILESTONE 2
================================

DATA VALIDATION + M1 DIAGNOSTICS

Purpose:
    Inspect the outputs created by Milestone 1.

    This script DOES NOT change the harmonization method.

    It checks:
        - date coverage
        - sector coverage
        - missing values
        - zero counts
        - total detections
        - FRP
        - MODIS / VIIRS relationship
        - harmonized activity
        - sensor discrepancy
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "output"

MODIS_PROCESSED = (
    PROCESSED_DIR / "modis_processed.csv"
)

VIIRS_PROCESSED = (
    PROCESSED_DIR / "viirs_processed.csv"
)

MODIS_DAILY = (
    PROCESSED_DIR / "modis_daily.csv"
)

VIIRS_DAILY = (
    PROCESSED_DIR / "viirs_daily.csv"
)

COMMON_MATRIX = (
    PROCESSED_DIR / "common_daily_matrix.csv"
)

M1_OUTPUT = (
    OUTPUT_DIR / "harmonized_daily_sector.csv"
)


def check_file(path: Path) -> None:

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file:\n{path}"
        )


def sensor_summary(
    df: pd.DataFrame,
    sensor_name: str,
) -> dict:

    df = df.copy()

    df["acq_date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce"
    )

    result = {
        "sensor": sensor_name,
        "rows": len(df),
        "dates": df["acq_date"].nunique(),
        "sectors": (
            df["sector"].nunique()
            if "sector" in df.columns
            else np.nan
        ),
    }

    if "frp" in df.columns:
        result["total_frp"] = df["frp"].sum()

    if "detection_count" in df.columns:
        result["total_detections"] = (
            df["detection_count"].sum()
        )

    if "count" in df.columns:
        result["total_detections"] = (
            df["count"].sum()
        )

    return result


def main():

    print("=" * 70)
    print("FIRE HARMONIZER - MILESTONE 2")
    print("M1 DATA VALIDATION")
    print("=" * 70)

    files = [
        MODIS_PROCESSED,
        VIIRS_PROCESSED,
        MODIS_DAILY,
        VIIRS_DAILY,
        COMMON_MATRIX,
        M1_OUTPUT,
    ]

    for file in files:
        check_file(file)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    modis = pd.read_csv(
        MODIS_PROCESSED
    )

    viirs = pd.read_csv(
        VIIRS_PROCESSED
    )

    modis_daily = pd.read_csv(
        MODIS_DAILY
    )

    viirs_daily = pd.read_csv(
        VIIRS_DAILY
    )

    common = pd.read_csv(
        COMMON_MATRIX
    )

    result = pd.read_csv(
        M1_OUTPUT
    )

    # --------------------------------------------------------
    # DATE CONVERSION
    # --------------------------------------------------------

    for df in [
        modis,
        viirs,
        modis_daily,
        viirs_daily,
        common,
        result,
    ]:

        df["acq_date"] = pd.to_datetime(
            df["acq_date"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # BASIC SUMMARIES
    # --------------------------------------------------------

    summaries = pd.DataFrame(
        [
            sensor_summary(
                modis,
                "MODIS"
            ),
            sensor_summary(
                viirs,
                "VIIRS"
            ),
        ]
    )

    # --------------------------------------------------------
    # COMMON MATRIX CHECK
    # --------------------------------------------------------

    date_count = common[
        "acq_date"
    ].nunique()

    sector_count = common[
        "sector"
    ].nunique()

    expected_rows = (
        date_count * 25
    )

    actual_rows = len(common)

    matrix_complete = (
        expected_rows == actual_rows
    )

    # --------------------------------------------------------
    # OVERLAP
    # --------------------------------------------------------

    positive_modis = (
        common["modis_count"] > 0
    )

    positive_viirs = (
        common["viirs_count"] > 0
    )

    both_detect = (
        positive_modis
        & positive_viirs
    )

    only_modis = (
        positive_modis
        & ~positive_viirs
    )

    only_viirs = (
        positive_viirs
        & ~positive_modis
    )

    neither = (
        ~positive_modis
        & ~positive_viirs
    )

    total_rows = len(common)

    overlap_report = {
        "total_date_sector_rows": total_rows,
        "both_detect": int(both_detect.sum()),
        "only_modis": int(only_modis.sum()),
        "only_viirs": int(only_viirs.sum()),
        "neither_detect": int(neither.sum()),
        "both_detect_fraction":
            both_detect.mean(),
        "only_modis_fraction":
            only_modis.mean(),
        "only_viirs_fraction":
            only_viirs.mean(),
        "neither_fraction":
            neither.mean(),
    }

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    pearson_counts = (
        common[
            [
                "modis_count",
                "viirs_count"
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    pearson_z = (
        result[
            [
                "modis_z",
                "viirs_z"
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    mean_discrepancy = (
        result["sensor_discrepancy"]
        .mean()
    )

    median_discrepancy = (
        result["sensor_discrepancy"]
        .median()
    )

    max_harmonized = (
        result["harmonized_activity"]
        .max()
    )

    min_harmonized = (
        result["harmonized_activity"]
        .min()
    )

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    missing_report = (
        result.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    # --------------------------------------------------------
    # WRITE REPORT
    # --------------------------------------------------------

    report_file = (
        OUTPUT_DIR
        / "milestone_2_validation.txt"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "FIRE HARMONIZER - MILESTONE 2\n"
        )

        file.write(
            "====================================\n\n"
        )

        file.write(
            "DATE COVERAGE\n"
        )

        file.write(
            f"Start: {common['acq_date'].min()}\n"
        )

        file.write(
            f"End:   {common['acq_date'].max()}\n"
        )

        file.write(
            f"Dates: {date_count}\n\n"
        )

        file.write(
            "SECTORS\n"
        )

        file.write(
            f"Unique sectors: {sector_count}\n"
        )

        file.write(
            f"Expected matrix rows: {expected_rows}\n"
        )

        file.write(
            f"Actual matrix rows: {actual_rows}\n"
        )

        file.write(
            f"Complete matrix: {matrix_complete}\n\n"
        )

        file.write(
            "SENSOR OVERLAP\n"
        )

        for key, value in overlap_report.items():
            file.write(
                f"{key}: {value}\n"
            )

        file.write("\n")

        file.write(
            "CORRELATION\n"
        )

        file.write(
            f"Raw count Pearson correlation: "
            f"{pearson_counts:.6f}\n"
        )

        file.write(
            f"Z-score Pearson correlation: "
            f"{pearson_z:.6f}\n\n"
        )

        file.write(
            "HARMONIZATION\n"
        )

        file.write(
            f"Mean discrepancy: "
            f"{mean_discrepancy:.6f}\n"
        )

        file.write(
            f"Median discrepancy: "
            f"{median_discrepancy:.6f}\n"
        )

        file.write(
            f"Minimum harmonized activity: "
            f"{min_harmonized:.6f}\n"
        )

        file.write(
            f"Maximum harmonized activity: "
            f"{max_harmonized:.6f}\n\n"
        )

        file.write(
            "MISSING VALUES\n"
        )

        for column, count in missing_report.items():

            file.write(
                f"{column}: {count}\n"
            )

    # --------------------------------------------------------
    # SAVE CSV SUMMARY
    # --------------------------------------------------------

    summaries.to_csv(
        PROCESSED_DIR
        / "sensor_summary.csv",
        index=False
    )

    pd.DataFrame(
        [overlap_report]
    ).to_csv(
        PROCESSED_DIR
        / "sensor_overlap.csv",
        index=False
    )

    print("\nValidation complete.")

    print(
        f"Report saved to:\n"
        f"  {report_file}"
    )

    print(
        "\nRaw count correlation:",
        round(
            pearson_counts,
            4
        )
    )

    print(
        "Z-score correlation:",
        round(
            pearson_z,
            4
        )
    )

    print(
        "Mean sensor discrepancy:",
        round(
            mean_discrepancy,
            4
        )
    )


if __name__ == "__main__":
    main()