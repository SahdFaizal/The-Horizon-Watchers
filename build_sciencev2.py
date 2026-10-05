"""
FIRE HARMONIZER - MILESTONE 7
================================

MULTI-EVIDENCE SCIENCE CORE

Purpose
-------
Extend the validated M1-M6 Fire Harmonizer science layer without
replacing the original M3 reference metrics.

Input
-----
data/processed/common_daily_matrix.csv

Optional observation-level context inputs
------------------------------------------
data/processed/modis_processed.csv
data/processed/viirs_processed.csv

Primary output
--------------
output/harmonized_science_all_years.csv

Additional output
-----------------
output/m7_science_summary.txt

M7 additions
------------
C : diagnostic-driven count anomaly evidence
F : FRP anomaly evidence (sensor-specific + harmonized)
G : bounded cross-sensor concordance
S : spatial coherence of anomalous neighboring sectors
U : observation/sampling context flags and diagnostics

Scientific design principles
----------------------------
1. M3 fields remain present as the historical reference model.
2. M7 fields are prefixed with m7_ so downstream milestones can
   explicitly choose the new science layer.
3. MODIS and VIIRS are standardized independently before comparison.
4. Count anomalies use diagnostics-driven residuals:
     - Poisson-like residual when count variance is near the mean;
     - Negative-binomial residual when overdispersion is substantial;
     - robust empirical score for strongly zero-inflated or degenerate
       baselines;
     - ordinary standardized score for regular continuous/count regimes.
5. FRP is treated as a physical power measurement, not converted into
   "energy" by summation. A harmonized FRP evidence score combines
   sensor-standardized FRP anomalies; it is not a physical total-energy
   product.
6. Cross-sensor concordance is an agreement descriptor, not ground truth.
7. Spatial coherence is based on independent neighboring sectors crossing
   the anomaly threshold, not simple spatial smoothing.
8. Observation context is kept as separate fields instead of collapsing
   quality into an unexplained confidence percentage.
9. If an optional observation-level context file does not contain a field,
   M7 records its availability honestly rather than inventing it.

This remains deterministic, inspectable statistical analysis.
No machine-learning model is required.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED_DIR / "common_daily_matrix.csv"
MODIS_OBSERVATIONS_FILE = PROCESSED_DIR / "modis_processed.csv"
VIIRS_OBSERVATIONS_FILE = PROCESSED_DIR / "viirs_processed.csv"

ALL_YEARS_OUTPUT = OUTPUT_DIR / "harmonized_science_all_years.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "m7_science_summary.txt"


# ============================================================
# CONFIGURATION
# ============================================================

GRID_SIZE = 5
EXPECTED_SECTORS = [f"C{i:02d}" for i in range(1, GRID_SIZE * GRID_SIZE + 1)]

MODIS_WEIGHT = 0.5
VIIRS_WEIGHT = 0.5

MIN_BASELINE_SAMPLES = 20
COUNT_ZERO_INFLATION_THRESHOLD = 0.80
COUNT_OVERDISPERSION_THRESHOLD = 1.50
ROBUST_SCALE_FLOOR = 1.0

ANOMALY_THRESHOLD = 2.0
CONCORDANCE_EPSILON = 1e-9
CONCORDANCE_MIN_SIGNAL = 0.50

OBS_CHUNK_SIZE = 500_000


# ============================================================
# REQUIRED INPUT COLUMNS
# ============================================================

REQUIRED_MATRIX_COLUMNS = [
    "acq_date",
    "sector",
    "modis_count",
    "viirs_count",
    "modis_frp",
    "viirs_frp",
]


# ============================================================
# BASIC HELPERS
# ============================================================


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns:\n  "
            + "\n  ".join(missing)
        )


def safe_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def safe_std(series: pd.Series) -> float:
    value = series.std(ddof=1)
    if pd.isna(value):
        return 0.0
    return float(value)


def safe_mad(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def nearest_valid_scale(*values: float, floor: float = ROBUST_SCALE_FLOOR) -> float:
    for value in values:
        if value is not None and np.isfinite(value) and value > floor * 1e-12:
            return float(max(value, floor))
    return float(floor)


# ============================================================
# LOAD COMMON SCIENCE MATRIX
# ============================================================


def load_data(input_file: Path = INPUT_FILE) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(
            f"Missing common matrix:\n{input_file}\n\n"
            "Run the existing M1 preprocessing pipeline first."
        )

    df = pd.read_csv(input_file)
    require_columns(df, REQUIRED_MATRIX_COLUMNS, "common_daily_matrix.csv")

    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.dropna(subset=["acq_date", "sector"]).copy()

    df = safe_numeric(
        df,
        ["modis_count", "viirs_count", "modis_frp", "viirs_frp"],
    )

    numeric = ["modis_count", "viirs_count", "modis_frp", "viirs_frp"]
    df[numeric] = df[numeric].fillna(0.0)

    df["year"] = df["acq_date"].dt.year.astype(int)
    df["month"] = df["acq_date"].dt.month.astype(int)

    # Guard against accidental duplicate sector-days.
    if df.duplicated(["acq_date", "sector"]).any():
        duplicate_count = int(df.duplicated(["acq_date", "sector"]).sum())
        raise ValueError(
            "Duplicate DATE × SECTOR rows detected in the common matrix: "
            f"{duplicate_count:,}. Duplicate rows would corrupt M7 continuity "
            "and spatial evidence."
        )

    df = df.sort_values(["acq_date", "sector"]).reset_index(drop=True)

    return df


# ============================================================
# OPTIONAL M3 REFERENCE SNAPSHOT
# ============================================================

def load_m3_reference(output_file: Path) -> pd.DataFrame | None:
    """
    Snapshot the existing M3 all-years science output before M7 overwrites it.

    The reference is used only to preserve the exact original M3 science fields
    for side-by-side comparison and reproducibility.
    """

    if not output_file.exists():
        return None

    try:
        reference = pd.read_csv(output_file)
    except Exception as exc:
        print(f"Warning: could not read existing M3 reference: {exc}")
        return None

    required_keys = ["acq_date", "sector"]
    if any(column not in reference.columns for column in required_keys):
        return None

    available = [
        column
        for column in [
            "modis_z",
            "viirs_z",
            "harmonized_activity",
            "anomaly_z",
        ]
        if column in reference.columns
    ]

    if not available:
        return None

    reference["acq_date"] = pd.to_datetime(
        reference["acq_date"],
        errors="coerce",
    )
    reference = reference.dropna(subset=["acq_date", "sector"]).copy()

    rename = {column: f"m3_{column}" for column in available}
    reference = reference.rename(columns=rename)

    return reference[
        ["acq_date", "sector"] + list(rename.values())
    ]


def merge_m3_reference(
    df: pd.DataFrame,
    reference: pd.DataFrame | None,
) -> tuple[pd.DataFrame, bool]:
    """Merge legacy M3 science fields without changing M7 calculations."""

    if reference is None or reference.empty:
        return df, False

    result = df.merge(
        reference,
        on=["acq_date", "sector"],
        how="left",
        validate="one_to_one",
    )

    return result, True


# ============================================================
# BASELINE SUMMARIES
# ============================================================


def grouped_statistics(
    frame: pd.DataFrame,
    value_column: str,
    group_columns: list[str],
) -> pd.DataFrame:
    """
    Summarize a baseline reference distribution.

    Robust distribution descriptors are included so the scoring method can
    be selected from diagnostics instead of assuming every variable is normal.
    """

    work = frame[group_columns + [value_column]].copy()
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work = work.dropna(subset=[value_column])

    output_columns = group_columns + [
        "baseline_n",
        "baseline_mean",
        "baseline_variance",
        "baseline_std",
        "baseline_zero_fraction",
        "baseline_median",
        "baseline_q1",
        "baseline_q3",
        "baseline_mad",
    ]

    if work.empty:
        return pd.DataFrame(columns=output_columns)

    if not group_columns:
        series = work[value_column]
        result = pd.DataFrame([{
            "baseline_n": int(series.count()),
            "baseline_mean": float(series.mean()),
            "baseline_variance": float(series.var(ddof=1)) if series.count() > 1 else 0.0,
            "baseline_std": float(series.std(ddof=1)) if series.count() > 1 else 0.0,
            "baseline_zero_fraction": float(np.mean(series.to_numpy() == 0)),
            "baseline_median": float(series.median()),
            "baseline_q1": float(series.quantile(0.25)),
            "baseline_q3": float(series.quantile(0.75)),
            "baseline_mad": safe_mad(series),
        }])
        return result

    grouped = work.groupby(group_columns, dropna=False)[value_column]

    stats = grouped.agg(
        baseline_n="count",
        baseline_mean="mean",
        baseline_variance="var",
        baseline_std="std",
        baseline_zero_fraction=lambda x: float(np.mean(np.asarray(x) == 0)),
        baseline_median="median",
    ).reset_index()

    quantiles = (
        grouped.quantile([0.25, 0.75])
        .unstack(level=-1)
        .rename(columns={0.25: "baseline_q1", 0.75: "baseline_q3"})
        .reset_index()
    )

    stats = stats.merge(
        quantiles[group_columns + ["baseline_q1", "baseline_q3"]],
        on=group_columns,
        how="left",
    )

    mad_frame = (
        work.groupby(group_columns, dropna=False)[value_column]
        .apply(safe_mad)
        .rename("baseline_mad")
        .reset_index()
    )

    stats = stats.merge(mad_frame, on=group_columns, how="left")
    stats["baseline_variance"] = stats["baseline_variance"].fillna(0.0).clip(lower=0.0)
    stats["baseline_std"] = stats["baseline_std"].fillna(0.0).clip(lower=0.0)
    stats["baseline_n"] = stats["baseline_n"].fillna(0).astype(int)

    return stats


# ============================================================
# LEAVE-ONE-YEAR-OUT BASELINE ATTACHMENT
# ============================================================


def add_diagnostic_baseline(
    df: pd.DataFrame,
    value_column: str,
    prefix: str,
    minimum_samples: int = MIN_BASELINE_SAMPLES,
) -> pd.DataFrame:
    """
    Attach a leave-one-year-out baseline using the established M3 hierarchy:

        sector + month
        sector
        month
        global

    A baseline is only accepted at a level when at least `minimum_samples`
    historical observations are available.  The selected level is stored in
    `<prefix>_baseline_level` for auditability.
    """

    result = df.copy()

    baseline_columns = [
        "baseline_n",
        "baseline_mean",
        "baseline_variance",
        "baseline_std",
        "baseline_zero_fraction",
        "baseline_median",
        "baseline_q1",
        "baseline_q3",
        "baseline_mad",
    ]

    for column in baseline_columns:
        result[f"{prefix}_{column}"] = np.nan

    result[f"{prefix}_baseline_level"] = "Unavailable"

    years = sorted(int(year) for year in result["year"].unique())
    levels = [
        ("sector+month", ["sector", "month"]),
        ("sector", ["sector"]),
        ("month", ["month"]),
        ("global", []),
    ]

    for evaluation_year in years:
        target_mask = result["year"] == evaluation_year
        target_index = result.index[target_mask]

        if len(target_index) == 0:
            continue

        history = result.loc[
            result["year"] != evaluation_year,
            ["year", "sector", "month", value_column],
        ].copy()

        if history.empty:
            continue

        assigned_mask = result.loc[target_index, f"{prefix}_baseline_mean"].notna()

        for level_name, groups in levels:
            if assigned_mask.all():
                break

            stats = grouped_statistics(history, value_column, groups)

            if groups:
                target = result.loc[target_index, groups].merge(
                    stats,
                    on=groups,
                    how="left",
                )
            else:
                overall = grouped_statistics(history, value_column, [])
                if overall.empty:
                    continue
                values = overall.iloc[0].to_dict()
                target = pd.DataFrame(
                    {column: [values.get(column)] * len(target_index) for column in baseline_columns}
                )

            target.index = target_index
            current_mean = result.loc[target_index, f"{prefix}_baseline_mean"]
            valid = current_mean.isna() & (
                pd.to_numeric(target["baseline_n"], errors="coerce") >= minimum_samples
            )

            if not valid.any():
                continue

            valid_index = target.index[valid]
            for column in baseline_columns:
                result.loc[valid_index, f"{prefix}_{column}"] = target.loc[
                    valid_index, column
                ].to_numpy()

            result.loc[valid_index, f"{prefix}_baseline_level"] = level_name

            assigned_mask = result.loc[target_index, f"{prefix}_baseline_mean"].notna()

    # Final deterministic fallback for exceptional/sparse situations.
    # Preserve leave-one-year-out semantics even at the last-resort level.
    remaining = result[f"{prefix}_baseline_mean"].isna()
    for evaluation_year in years:
        year_remaining = remaining & (result["year"] == evaluation_year)
        if not year_remaining.any():
            continue

        history = result.loc[
            result["year"] != evaluation_year,
            [value_column],
        ]
        global_stats = grouped_statistics(
            history,
            value_column,
            [],
        )

        if global_stats.empty:
            continue

        row = global_stats.iloc[0]
        for column in baseline_columns:
            result.loc[year_remaining, f"{prefix}_{column}"] = row[column]
        result.loc[year_remaining, f"{prefix}_baseline_level"] = "loo-global-fallback"

    # Numerical protection while keeping baseline diagnostics visible.
    result[f"{prefix}_baseline_variance"] = (
        pd.to_numeric(result[f"{prefix}_baseline_variance"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )

    result[f"{prefix}_baseline_std"] = (
        pd.to_numeric(result[f"{prefix}_baseline_std"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )

    result[f"{prefix}_baseline_n"] = (
        pd.to_numeric(result[f"{prefix}_baseline_n"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    for column in [
        f"{prefix}_baseline_zero_fraction",
        f"{prefix}_baseline_median",
        f"{prefix}_baseline_q1",
        f"{prefix}_baseline_q3",
        f"{prefix}_baseline_mad",
    ]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    return result


# ============================================================
# DIAGNOSTIC-DRIVEN SCORING
# ============================================================


def _score_row(
    observed: float,
    mean: float,
    variance: float,
    std: float,
    zero_fraction: float,
    median: float,
    q1: float,
    q3: float,
    mad: float,
    is_count: bool,
) -> tuple[float, str]:
    """Return (standardized evidence, method_name)."""

    if not np.isfinite(observed):
        return np.nan, "unavailable"

    mean = 0.0 if not np.isfinite(mean) else float(mean)
    variance = max(0.0, 0.0 if not np.isfinite(variance) else float(variance))
    std = max(0.0, 0.0 if not np.isfinite(std) else float(std))
    zero_fraction = min(1.0, max(0.0, 0.0 if not np.isfinite(zero_fraction) else float(zero_fraction)))
    median = mean if not np.isfinite(median) else float(median)
    q1 = median if not np.isfinite(q1) else float(q1)
    q3 = median if not np.isfinite(q3) else float(q3)
    mad = 0.0 if not np.isfinite(mad) else max(0.0, float(mad))

    if is_count:
        # Strong zero inflation or degenerate dispersion -> robust empirical score.
        if zero_fraction >= COUNT_ZERO_INFLATION_THRESHOLD or std == 0.0:
            iqr = max(0.0, q3 - q1)
            robust_scale = nearest_valid_scale(
                1.4826 * mad,
                iqr / 1.349 if iqr > 0 else 0.0,
            )
            return (observed - median) / robust_scale, "robust_empirical"

        # Variance substantially above the mean -> negative-binomial residual.
        if mean > 0 and variance / mean >= COUNT_OVERDISPERSION_THRESHOLD:
            alpha = max((variance - mean) / (mean * mean), 0.0)
            denominator = math.sqrt(max(mean + alpha * mean * mean, 1e-12))
            return (observed - mean) / denominator, "negative_binomial_residual"

        # Poisson-like count behavior: variance is approximately the mean.
        if mean > 0:
            denominator = math.sqrt(max(mean, 1e-12))
            return (observed - mean) / denominator, "poisson_residual"

        # All baseline values are zero or near zero.
        if observed == 0:
            return 0.0, "zero_baseline"

        return (observed - median) / nearest_valid_scale(
            1.4826 * mad,
            (q3 - q1) / 1.349 if q3 > q1 else 0.0,
        ), "robust_empirical"

    # FRP/continuous evidence: use a robust empirical score when the
    # baseline is degenerate or strongly skewed; otherwise use ordinary Z.
    if std > 0:
        skew_proxy = (q3 - median) - (median - q1)
        dispersion_ratio = abs(skew_proxy) / max(q3 - q1, 1e-12)

        if zero_fraction >= COUNT_ZERO_INFLATION_THRESHOLD or dispersion_ratio > 1.5:
            robust_scale = nearest_valid_scale(
                1.4826 * mad,
                (q3 - q1) / 1.349 if q3 > q1 else 0.0,
            )
            return (observed - median) / robust_scale, "robust_empirical"

        return (observed - mean) / std, "standard_z"

    robust_scale = nearest_valid_scale(
        1.4826 * mad,
        (q3 - q1) / 1.349 if q3 > q1 else 0.0,
    )
    return (observed - median) / robust_scale, "robust_empirical"


def calculate_diagnostic_score(
    df: pd.DataFrame,
    value_column: str,
    prefix: str,
    is_count: bool,
) -> pd.DataFrame:
    result = df.copy()

    score_column = f"m7_{prefix}_z"
    method_column = f"m7_{prefix}_method"

    score_values: list[float] = []
    methods: list[str] = []

    for row in result[
        [
            value_column,
            f"{prefix}_baseline_mean",
            f"{prefix}_baseline_variance",
            f"{prefix}_baseline_std",
            f"{prefix}_baseline_zero_fraction",
            f"{prefix}_baseline_median",
            f"{prefix}_baseline_q1",
            f"{prefix}_baseline_q3",
            f"{prefix}_baseline_mad",
        ]
    ].itertuples(index=False, name=None):
        score, method = _score_row(
            observed=float(row[0]) if np.isfinite(row[0]) else np.nan,
            mean=float(row[1]) if np.isfinite(row[1]) else np.nan,
            variance=float(row[2]) if np.isfinite(row[2]) else np.nan,
            std=float(row[3]) if np.isfinite(row[3]) else np.nan,
            zero_fraction=float(row[4]) if np.isfinite(row[4]) else np.nan,
            median=float(row[5]) if np.isfinite(row[5]) else np.nan,
            q1=float(row[6]) if np.isfinite(row[6]) else np.nan,
            q3=float(row[7]) if np.isfinite(row[7]) else np.nan,
            mad=float(row[8]) if np.isfinite(row[8]) else np.nan,
            is_count=is_count,
        )
        score_values.append(score)
        methods.append(method)

    result[score_column] = score_values
    result[method_column] = methods

    return result


# ============================================================
# M7 SENSOR + FRP EVIDENCE
# ============================================================


def add_m7_sensor_evidence(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result = add_diagnostic_baseline(
        result,
        value_column="modis_count",
        prefix="modis_count",
    )
    result = calculate_diagnostic_score(
        result,
        value_column="modis_count",
        prefix="modis_count",
        is_count=True,
    )

    result = add_diagnostic_baseline(
        result,
        value_column="viirs_count",
        prefix="viirs_count",
    )
    result = calculate_diagnostic_score(
        result,
        value_column="viirs_count",
        prefix="viirs_count",
        is_count=True,
    )

    result = add_diagnostic_baseline(
        result,
        value_column="modis_frp",
        prefix="modis_frp",
    )
    result = calculate_diagnostic_score(
        result,
        value_column="modis_frp",
        prefix="modis_frp",
        is_count=False,
    )

    result = add_diagnostic_baseline(
        result,
        value_column="viirs_frp",
        prefix="viirs_frp",
    )
    result = calculate_diagnostic_score(
        result,
        value_column="viirs_frp",
        prefix="viirs_frp",
        is_count=False,
    )

    # Short aliases for the evidence vector.
    result["m7_C"] = (
        MODIS_WEIGHT * result["m7_modis_count_z"]
        + VIIRS_WEIGHT * result["m7_viirs_count_z"]
    )

    result["m7_F"] = (
        MODIS_WEIGHT * result["m7_modis_frp_z"]
        + VIIRS_WEIGHT * result["m7_viirs_frp_z"]
    )

    return result


# ============================================================
# CROSS-SENSOR CONCORDANCE G
# ============================================================


def add_cross_sensor_concordance(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    modis_z = result["m7_modis_count_z"]
    viirs_z = result["m7_viirs_count_z"]

    denominator = modis_z.abs() + viirs_z.abs() + CONCORDANCE_EPSILON

    result["m7_G"] = 1.0 - (modis_z - viirs_z).abs() / denominator
    result["m7_G"] = result["m7_G"].clip(lower=0.0, upper=1.0)

    signal_strength = pd.concat(
        [modis_z.abs(), viirs_z.abs()],
        axis=1,
    ).max(axis=1)

    result["m7_concordance_signal_strength"] = signal_strength
    result["m7_concordance_interpretable"] = (
        signal_strength >= CONCORDANCE_MIN_SIGNAL
    )

    # A categorical description prevents users from reading G≈1 at small
    # signals as evidence of a major event.
    result["m7_G_interpretation"] = np.select(
        [
            signal_strength < CONCORDANCE_MIN_SIGNAL,
            result["m7_G"] >= 0.80,
            result["m7_G"] >= 0.50,
        ],
        [
            "High agreement at low signal",
            "High sensor concordance",
            "Moderate sensor concordance",
        ],
        default="Low sensor concordance",
    )

    return result


# ============================================================
# SPATIAL COHERENCE S
# ============================================================


def sector_to_position(sector: str) -> tuple[int, int]:
    number = int(str(sector).replace("C", ""))
    zero_based = number - 1
    return zero_based // GRID_SIZE, zero_based % GRID_SIZE


def add_spatial_coherence(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["m7_S"] = np.nan
    result["m7_neighbor_anomaly_fraction"] = np.nan
    result["m7_anomalous_neighbor_count"] = 0

    sector_position = {sector: sector_to_position(sector) for sector in EXPECTED_SECTORS}

    all_dates = sorted(result["acq_date"].dropna().unique())

    # Work in a compact date × 5 × 5 array rather than repeatedly pivoting.
    date_index = pd.Index(all_dates)
    date_lookup = {date: index for index, date in enumerate(all_dates)}

    anomaly_grid = np.full(
        (len(all_dates), GRID_SIZE, GRID_SIZE),
        np.nan,
        dtype=float,
    )

    for row in result[["acq_date", "sector", "m7_C"]].itertuples(index=False, name=None):
        date_value, sector, score = row
        if sector not in sector_position:
            continue
        r, c = sector_position[sector]
        anomaly_grid[date_lookup[date_value], r, c] = score

    qualifying = anomaly_grid >= ANOMALY_THRESHOLD
    neighbor_counts = np.zeros_like(anomaly_grid, dtype=float)
    neighbor_totals = np.zeros_like(anomaly_grid, dtype=float)

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue

            source = qualifying
            shifted = np.full_like(source, False, dtype=bool)
            valid = np.full_like(source, False, dtype=bool)

            row_source_start = max(0, -dr)
            row_source_end = min(GRID_SIZE, GRID_SIZE - dr)
            col_source_start = max(0, -dc)
            col_source_end = min(GRID_SIZE, GRID_SIZE - dc)

            row_target_start = max(0, dr)
            row_target_end = min(GRID_SIZE, GRID_SIZE + dr)
            col_target_start = max(0, dc)
            col_target_end = min(GRID_SIZE, GRID_SIZE + dc)

            shifted[
                :,
                row_target_start:row_target_end,
                col_target_start:col_target_end,
            ] = source[
                :,
                row_source_start:row_source_end,
                col_source_start:col_source_end,
            ]

            valid[
                :,
                row_target_start:row_target_end,
                col_target_start:col_target_end,
            ] = ~np.isnan(
                anomaly_grid[
                    :,
                    row_source_start:row_source_end,
                    col_source_start:col_source_end,
                ]
            )

            neighbor_counts += shifted.astype(float)
            neighbor_totals += valid.astype(float)

    coherence_grid = np.divide(
        neighbor_counts,
        neighbor_totals,
        out=np.full_like(neighbor_counts, np.nan),
        where=neighbor_totals > 0,
    )

    for row_index, row in result[["acq_date", "sector"]].iterrows():
        sector = row["sector"]
        date_value = row["acq_date"]
        if sector not in sector_position or date_value not in date_lookup:
            continue

        r, c = sector_position[sector]
        date_number = date_lookup[date_value]

        result.at[row_index, "m7_S"] = coherence_grid[date_number, r, c]
        result.at[row_index, "m7_neighbor_anomaly_fraction"] = coherence_grid[date_number, r, c]
        result.at[row_index, "m7_anomalous_neighbor_count"] = int(
            neighbor_counts[date_number, r, c]
        )

    return result


# ============================================================
# OBSERVATION CONTEXT U
# ============================================================


def detect_optional_columns(columns: list[str]) -> dict[str, str | None]:
    lower_map = {str(column).lower(): column for column in columns}

    def choose(*candidates: str) -> str | None:
        for candidate in candidates:
            if candidate.lower() in lower_map:
                return lower_map[candidate.lower()]
        return None

    return {
        "confidence": choose("confidence"),
        "scan": choose("scan"),
        "track": choose("track"),
        "daynight": choose("daynight", "day_night"),
        "cloud": choose("cloud", "cloud_cover", "cloudcover", "cloud_percentage"),
        "acq_time": choose("acq_time", "acqtime"),
        "type": choose("type", "hotspot_type"),
        "version": choose("version", "product_version"),
    }


def observation_context_for_file(
    path: Path,
    sensor_name: str,
    chunk_size: int = OBS_CHUNK_SIZE,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """
    Stream the processed detection-level file and aggregate only the context
    fields required by U. This avoids loading tens of millions of observations
    into memory at once.
    """

    if not path.exists():
        return pd.DataFrame(), {"file_available": False}

    header = pd.read_csv(path, nrows=0)
    column_map = detect_optional_columns(header.columns.tolist())

    use_columns = ["acq_date", "sector"]
    for key, column in column_map.items():
        if column and column not in use_columns:
            use_columns.append(column)

    frames: list[pd.DataFrame] = []

    for chunk in pd.read_csv(path, usecols=use_columns, chunksize=chunk_size):
        chunk["acq_date"] = pd.to_datetime(chunk["acq_date"], errors="coerce")
        chunk = chunk.dropna(subset=["acq_date", "sector"]).copy()
        if chunk.empty:
            continue

        aggregation = {}

        if column_map["confidence"]:
            column = column_map["confidence"]
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
            aggregation["mean_confidence"] = (column, "mean")

        if column_map["scan"]:
            column = column_map["scan"]
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
            aggregation["mean_scan"] = (column, "mean")

        if column_map["track"]:
            column = column_map["track"]
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
            aggregation["mean_track"] = (column, "mean")

        if column_map["cloud"]:
            column = column_map["cloud"]
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
            aggregation["mean_cloud"] = (column, "mean")

        if column_map["daynight"]:
            column = column_map["daynight"]
            values = chunk[column].astype(str).str.lower().str.strip()
            chunk["day_fraction"] = values.isin(["d", "day", "daytime"]).astype(float)
            aggregation["day_fraction"] = ("day_fraction", "mean")

        if column_map["acq_time"]:
            column = column_map["acq_time"]
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
            aggregation["mean_acq_time"] = (column, "mean")

        aggregation["observation_count"] = ("sector", "size")

        grouped = (
            chunk.groupby(["acq_date", "sector"], as_index=False)
            .agg(**aggregation)
        )

        frames.append(grouped)

    if not frames:
        return pd.DataFrame(), {
            "file_available": True,
            "row_data_available": False,
            **{f"{key}_available": bool(value) for key, value in column_map.items()},
        }

    result = pd.concat(frames, ignore_index=True)

    numeric_columns = [
        column
        for column in result.columns
        if column not in {"acq_date", "sector"}
    ]

    result = (
        result.groupby(["acq_date", "sector"], as_index=False)[numeric_columns]
        .agg({column: "sum" if column in {"observation_count"} else "mean" for column in numeric_columns})
    )

    availability = {
        "file_available": True,
        "row_data_available": True,
        **{f"{key}_available": bool(value) for key, value in column_map.items()},
    }

    # Avoid interpreting missing optional columns as zeros.
    return result, availability


def add_observation_context(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, bool]]]:
    result = df.copy()

    modis_context, modis_availability = observation_context_for_file(
        MODIS_OBSERVATIONS_FILE,
        "MODIS",
    )
    viirs_context, viirs_availability = observation_context_for_file(
        VIIRS_OBSERVATIONS_FILE,
        "VIIRS",
    )

    if not modis_context.empty:
        result = result.merge(
            modis_context,
            on=["acq_date", "sector"],
            how="left",
            suffixes=("", "_modis_context"),
        )
        rename = {
            column: f"u_modis_{column}"
            for column in modis_context.columns
            if column not in {"acq_date", "sector"}
        }
        result = result.rename(columns=rename)
    else:
        for column in [
            "observation_count",
            "mean_confidence",
            "mean_scan",
            "mean_track",
            "mean_cloud",
            "day_fraction",
            "mean_acq_time",
        ]:
            result[f"u_modis_{column}"] = np.nan

    if not viirs_context.empty:
        result = result.merge(
            viirs_context,
            on=["acq_date", "sector"],
            how="left",
            suffixes=("", "_viirs_context"),
        )
        rename = {
            column: f"u_viirs_{column}"
            for column in viirs_context.columns
            if column not in {"acq_date", "sector"}
        }
        result = result.rename(columns=rename)
    else:
        for column in [
            "observation_count",
            "mean_confidence",
            "mean_scan",
            "mean_track",
            "mean_cloud",
            "day_fraction",
            "mean_acq_time",
        ]:
            result[f"u_viirs_{column}"] = np.nan

    # Matrix-level sensor state is always available.
    modis_detect = result["modis_count"] > 0
    viirs_detect = result["viirs_count"] > 0

    result["u_modis_detected"] = modis_detect
    result["u_viirs_detected"] = viirs_detect
    result["u_both_sensors_detect"] = modis_detect & viirs_detect
    result["u_only_modis"] = modis_detect & ~viirs_detect
    result["u_only_viirs"] = viirs_detect & ~modis_detect
    result["u_neither_sensor_detects"] = ~modis_detect & ~viirs_detect

    result["u_sensor_state"] = np.select(
        [
            result["u_both_sensors_detect"],
            result["u_only_modis"],
            result["u_only_viirs"],
        ],
        [
            "both_sensors",
            "modis_only",
            "viirs_only",
        ],
        default="neither",
    )

    # Explicitly expose what context data are actually present.
    result["u_context_available_modis"] = bool(modis_availability.get("row_data_available", False))
    result["u_context_available_viirs"] = bool(viirs_availability.get("row_data_available", False))
    result["u_cloud_context_available_modis"] = bool(modis_availability.get("cloud_available", False))
    result["u_cloud_context_available_viirs"] = bool(viirs_availability.get("cloud_available", False))

    return result, {
        "MODIS": modis_availability,
        "VIIRS": viirs_availability,
    }


# ============================================================
# M7 SUMMARY / DIAGNOSTICS
# ============================================================


def build_summary(
    df: pd.DataFrame,
    context_availability: dict[str, dict[str, bool]],
) -> str:
    lines = []

    lines.append("FIRE HARMONIZER - MILESTONE 7")
    lines.append("MULTI-EVIDENCE SCIENCE CORE")
    lines.append("=" * 70)
    lines.append("")

    lines.append("DATA")
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Dates: {df['acq_date'].nunique():,}")
    lines.append(f"Sectors: {df['sector'].nunique():,}")
    lines.append(f"Years: {df['year'].nunique():,}")
    lines.append(f"Start: {df['acq_date'].min()}")
    lines.append(f"End: {df['acq_date'].max()}")
    lines.append("")

    lines.append("M7 EVIDENCE VECTOR")
    lines.append("C = count anomaly evidence")
    lines.append("F = FRP anomaly evidence")
    lines.append("G = cross-sensor concordance")
    lines.append("S = neighboring-sector anomaly coherence")
    lines.append("U = observation/sampling context fields")
    lines.append("")

    lines.append("COUNT SCORING METHODS")
    for column in ["m7_modis_count_method", "m7_viirs_count_method"]:
        if column in df.columns:
            lines.append(df[column].value_counts(dropna=False).to_string())
            lines.append("")

    lines.append("FRP SCORING METHODS")
    for column in ["m7_modis_frp_method", "m7_viirs_frp_method"]:
        if column in df.columns:
            lines.append(df[column].value_counts(dropna=False).to_string())
            lines.append("")

    lines.append("BASELINE SUPPORT")
    for column in [
        "m7_modis_count_baseline_level",
        "m7_viirs_count_baseline_level",
        "m7_modis_frp_baseline_level",
        "m7_viirs_frp_baseline_level",
    ]:
        if column in df.columns:
            lines.append(column)
            lines.append(df[column].value_counts(dropna=False).to_string())
            lines.append("")

    lines.append("G / CONCORDANCE")
    if "m7_G" in df.columns:
        lines.append(f"Mean G: {df['m7_G'].mean():.6f}")
        lines.append(f"Median G: {df['m7_G'].median():.6f}")
        lines.append(
            "Rows with interpretable signal strength: "
            f"{int(df['m7_concordance_interpretable'].sum()):,}"
        )
    lines.append("")

    lines.append("S / SPATIAL COHERENCE")
    if "m7_S" in df.columns:
        valid_s = df["m7_S"].dropna()
        if not valid_s.empty:
            lines.append(f"Mean S: {valid_s.mean():.6f}")
            lines.append(f"Median S: {valid_s.median():.6f}")
    lines.append("")

    lines.append("U / OBSERVATION CONTEXT AVAILABILITY")
    for sensor_name, availability in context_availability.items():
        lines.append(f"{sensor_name}:")
        for key, value in sorted(availability.items()):
            lines.append(f"  {key}: {value}")
    lines.append("")

    lines.append("M3 REFERENCE PRESERVATION")
    legacy_columns = [
        "m3_modis_z",
        "m3_viirs_z",
        "m3_harmonized_activity",
        "m3_anomaly_z",
    ]
    preserved_count = sum(column in df.columns for column in legacy_columns)
    lines.append(f"Legacy M3 columns preserved: {preserved_count}/4")
    lines.append("")

    lines.append("U / SENSOR STATE")
    if "u_sensor_state" in df.columns:
        lines.append(df["u_sensor_state"].value_counts(dropna=False).to_string())
    lines.append("")

    lines.append("SCIENTIFIC NOTES")
    lines.append(
        "G is an observational concordance descriptor. It is not ground truth and "
        "does not prove that a sector-day represents a unique physical fire."
    )
    lines.append(
        "F is standardized FRP anomaly evidence. It is not a physical energy total."
    )
    lines.append(
        "S measures the proportion of available adjacent sectors whose M7 count "
        f"evidence crosses {ANOMALY_THRESHOLD:.1f}."
    )
    lines.append(
        "U is intentionally represented as separate context fields instead of an "
        "arbitrary composite confidence percentage."
    )
    lines.append(
        "Missing optional fields are reported as unavailable rather than imputed."
    )
    lines.append("")

    return "\n".join(lines)


# ============================================================
# MAIN M7 PIPELINE
# ============================================================


def run_m7(
    input_file: Path = INPUT_FILE,
    output_file: Path = ALL_YEARS_OUTPUT,
    write_context: bool = True,
) -> pd.DataFrame:
    print("=" * 70)
    print("FIRE HARMONIZER - MILESTONE 7")
    print("MULTI-EVIDENCE SCIENCE CORE")
    print("=" * 70)

    # Snapshot the pre-M7 M3 file before it is overwritten.
    m3_reference = load_m3_reference(output_file)

    print("\nLoading M1/M3 common matrix...")
    df = load_data(input_file)

    print(f"Rows loaded: {len(df):,}")
    if m3_reference is not None:
        print("Existing M3 reference detected; legacy M3 metrics will be preserved as m3_*.")
    else:
        print("No existing M3 reference file detected; M7 will create a new science file without m3_* snapshot fields.")

    print("\n[1/5] Diagnostic-driven count and FRP baselines...")
    df = add_m7_sensor_evidence(df)

    print("[2/5] Cross-sensor concordance G...")
    df = add_cross_sensor_concordance(df)

    print("[3/5] Spatial coherence S...")
    df = add_spatial_coherence(df)

    if write_context:
        print("[4/5] Observation context U from retained detection-level records...")
        df, context_availability = add_observation_context(df)
    else:
        print("[4/5] Observation context U skipped by command-line option...")
        context_availability = {
            "MODIS": {"file_available": False, "skipped": True},
            "VIIRS": {"file_available": False, "skipped": True},
        }

    df, m3_reference_preserved = merge_m3_reference(
        df,
        m3_reference,
    )

    # --------------------------------------------------------
    # Final common fields
    # --------------------------------------------------------
    df["m7_E_label"] = "E = (C, F, G, S, U)"
    df["m7_anomaly_threshold"] = ANOMALY_THRESHOLD

    # Keep rows in exact DATE × SECTOR order.
    df = df.sort_values(["acq_date", "sector"]).reset_index(drop=True)

    # --------------------------------------------------------
    # Write output
    # --------------------------------------------------------
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    summary_text = build_summary(df, context_availability)
    SUMMARY_OUTPUT.write_text(summary_text, encoding="utf-8")

    print("[5/5] Outputs written...")
    print(f"\nSaved M7 science file:\n  {output_file}")
    print(f"Saved M7 summary:\n  {SUMMARY_OUTPUT}")

    print("\n" + "=" * 70)
    print("MILESTONE 7 COMPLETE")
    print("=" * 70)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    for column in ["m7_C", "m7_F", "m7_G", "m7_S"]:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            print(
                f"{column}: mean={values.mean():.6f}, "
                f"median={values.median():.6f}"
            )

    return df


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fire Harmonizer M7 Multi-Evidence Science Core"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
        help="Common DATE × SECTOR matrix input file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=ALL_YEARS_OUTPUT,
        help="M7 all-years science output file.",
    )

    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help=(
            "Also write an analysis-year subset after the complete all-years "
            "M7 science file has been generated. Historical baselines still use "
            "all years except the evaluation year."
        ),
    )

    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Skip streaming the observation-level MODIS/VIIRS context files.",
    )

    args = parser.parse_args()

    result = run_m7(
        input_file=args.input,
        output_file=args.output,
        write_context=not args.no_context,
    )

    if args.year is not None:
        available_years = sorted(result["year"].dropna().astype(int).unique())
        if args.year not in available_years:
            raise ValueError(
                f"Year {args.year} is not available. "
                f"Available years: {available_years}"
            )

        target_output = (
            args.output.parent
            / f"harmonized_science_{args.year}.csv"
        )
        result[result["year"] == args.year].to_csv(
            target_output,
            index=False,
        )
        print(f"Saved target-year M7 file:\n  {target_output}")


if __name__ == "__main__":
    main()
