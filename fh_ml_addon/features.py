from __future__ import annotations

import numpy as np
import pandas as pd

CURRENT_FEATURES = (
    "m7_C", "m7_F", "m7_G", "m7_S", "m3_harmonized_activity", "m3_anomaly_z"
)
LAG_PLAN = {
    "m7_C": (1, 2, 3, 7, 14),
    "m7_F": (1, 2, 3, 7, 14),
    "m7_G": (1, 7, 14),
    "m7_S": (1, 7, 14),
    "m3_harmonized_activity": (1, 7),
    "m3_anomaly_z": (1, 7),
}
ROLLING_WINDOWS = (3, 7, 30)


def build_features(science: pd.DataFrame, c_threshold: float = 2.0) -> tuple[pd.DataFrame, list[str]]:
    """Create a deterministic, row-aligned feature matrix using only current/past data.

    Issue-time convention: a row dated t is forecast after the available harmonized
    observations for calendar date t have been produced. Lags are exact calendar-day
    lookups; absent sector-days remain NaN rather than being treated as zero activity.
    """
    work = science.sort_values(["sector", "date"]).reset_index(drop=True).copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work["sector"] = work["sector"].astype(str)
    feature = pd.DataFrame(index=work.index)

    for col in CURRENT_FEATURES:
        if col in work.columns:
            feature[f"now_{col}"] = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)

    keyed = work.set_index(["sector", "date"])
    for col, lags in LAG_PLAN.items():
        if col not in keyed.columns:
            continue
        for lag in lags:
            lookup = pd.MultiIndex.from_arrays(
                [work["sector"].to_numpy(), (work["date"] - pd.to_timedelta(lag, unit="D")).to_numpy()],
                names=["sector", "date"],
            )
            feature[f"{col}_lag_{lag}d"] = keyed[col].reindex(lookup).to_numpy(dtype=float)

    # Calendar-time rolling features. Offset windows depend on dates, not row counts.
    for col in ("m7_C", "m7_F"):
        if col not in work.columns:
            continue
        for window in ROLLING_WINDOWS:
            mean = np.full(len(work), np.nan, dtype=float)
            std = np.full(len(work), np.nan, dtype=float)
            maximum = np.full(len(work), np.nan, dtype=float)
            count = np.zeros(len(work), dtype=float)
            active_days = np.zeros(len(work), dtype=float)
            for _, group in work.groupby("sector", sort=False):
                s = group.set_index("date")[col].sort_index()
                roll = s.rolling(f"{window}D", min_periods=1)
                idx = group.index.to_numpy()
                mean[idx] = roll.mean().to_numpy(dtype=float)
                std[idx] = roll.std(ddof=0).where(roll.count() >= 2).to_numpy(dtype=float)
                maximum[idx] = roll.max().to_numpy(dtype=float)
                count[idx] = roll.count().to_numpy(dtype=float)
                if col == "m7_C":
                    active = (s >= c_threshold).astype(float)
                    active_days[idx] = active.rolling(f"{window}D", min_periods=1).sum().to_numpy(dtype=float)
            feature[f"{col}_rolling_mean_{window}d"] = mean
            feature[f"{col}_rolling_std_{window}d"] = std
            feature[f"{col}_rolling_max_{window}d"] = maximum
            feature[f"{col}_coverage_days_{window}d"] = count
            if col == "m7_C":
                feature[f"C_active_days_{window}d"] = active_days

    if "m7_C_rolling_mean_3d" in feature and "m7_C_rolling_mean_30d" in feature:
        feature["C_momentum_3d_vs_30d"] = feature["m7_C_rolling_mean_3d"] - feature["m7_C_rolling_mean_30d"]
    if "m7_F_rolling_mean_3d" in feature and "m7_F_rolling_mean_30d" in feature:
        feature["F_momentum_3d_vs_30d"] = feature["m7_F_rolling_mean_3d"] - feature["m7_F_rolling_mean_30d"]

    day = work["date"].dt.dayofyear.to_numpy(dtype=float)
    period = 365.2425
    for harmonic in (1, 2):
        feature[f"sin_doy_h{harmonic}"] = np.sin(2 * np.pi * harmonic * day / period)
        feature[f"cos_doy_h{harmonic}"] = np.cos(2 * np.pi * harmonic * day / period)

    # Keep the sector vocabulary from the observed science file. A later unseen sector
    # produces zeros for this vocabulary during reindexing and will not crash inference.
    sector_dummies = pd.get_dummies(work["sector"], prefix="sector", prefix_sep="__", dtype=np.uint8)
    feature = pd.concat([feature, sector_dummies], axis=1)
    feature = feature.replace([np.inf, -np.inf], np.nan).astype(float)
    feature.insert(0, "sector", work["sector"].to_numpy())
    feature.insert(0, "date", work["date"].to_numpy())
    feature_names = [c for c in feature.columns if c not in ("date", "sector")]
    return feature, feature_names
