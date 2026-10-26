from __future__ import annotations

import numpy as np
import pandas as pd


def make_target(
    science: pd.DataFrame,
    horizon_days: int,
    c_threshold: float = 2.0,
    elevated_min_active_days: int = 3,
) -> tuple[pd.Series, pd.Series, str]:
    """Make a forward-looking target using exact future calendar dates.

    For 7/14 days, positive means at least one future day has m7_C >= c_threshold.
    For 30 days, positive means at least elevated_min_active_days of the next 30 days
    meet the threshold. The full future window must be observed to get any label.
    """
    if horizon_days not in (7, 14, 30):
        raise ValueError("Supported horizons are 7, 14 and 30 calendar days.")
    work = science.sort_values(["sector", "date"]).reset_index(drop=True).copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work["sector"] = work["sector"].astype(str)
    keyed = work.set_index(["sector", "date"])["m7_C"]
    future = np.full((len(work), horizon_days), np.nan, dtype=float)
    for offset in range(1, horizon_days + 1):
        lookup = pd.MultiIndex.from_arrays(
            [work["sector"].to_numpy(), (work["date"] + pd.to_timedelta(offset, unit="D")).to_numpy()],
            names=["sector", "date"],
        )
        future[:, offset - 1] = keyed.reindex(lookup).to_numpy(dtype=float)
    valid = pd.Series(np.isfinite(future).all(axis=1), index=work.index, name=f"label_valid_{horizon_days}d")
    active_days = (future >= c_threshold).sum(axis=1)
    if horizon_days == 30:
        target = active_days >= int(elevated_min_active_days)
        definition = (
            f"1 if at least {int(elevated_min_active_days)} of the next 30 calendar days have "
            f"m7_C >= {c_threshold:g}; otherwise 0"
        )
    else:
        target = active_days >= 1
        definition = f"1 if any of the next {horizon_days} calendar days has m7_C >= {c_threshold:g}; otherwise 0"
    target = pd.Series(target.astype("int8"), index=work.index, name=f"target_{horizon_days}d")
    # Avoid carrying meaningless negative targets from incomplete horizons.
    target.loc[~valid] = 0
    return target, valid, definition
