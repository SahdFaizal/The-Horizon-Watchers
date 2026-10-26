from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

DATE_CANDIDATES = ("acq_date", "date")
OPTIONAL_NUMERIC = (
    "m7_C", "m7_F", "m7_G", "m7_S",
    "m3_harmonized_activity", "m3_anomaly_z",
)


def discover_science_file(project_root: Path | str = ".") -> Path | None:
    """Find the canonical science CSV without falling back to synthetic fixtures."""
    root = Path(project_root).resolve()
    candidates = (
        root / "output" / "harmonized_science_all_years.csv",
        root / "data" / "output" / "harmonized_science_all_years.csv",
        root / "harmonized_science_all_years.csv",
    )
    return next((p for p in candidates if p.is_file()), None)


def load_science(path: Path | str) -> pd.DataFrame:
    """Load and validate the sector-day science matrix.

    Rows with an unparseable date or blank sector are removed with counts recorded
    by callers. Missing m7_C values are retained as unknown, not converted to zero;
    they cannot be used in future target labels.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Science CSV not found: {source}. Expected output/harmonized_science_all_years.csv "
            "or pass --science with the correct path."
        )
    frame = pd.read_csv(source, low_memory=False)
    date_col = next((c for c in DATE_CANDIDATES if c in frame.columns), None)
    missing = [c for c in ("sector", "m7_C") if c not in frame.columns]
    if date_col is None or missing:
        raise ValueError(
            f"Required fields: date/acq_date, sector, m7_C. Found date column={date_col!r}; "
            f"missing={missing}; available={list(frame.columns)}"
        )

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame[date_col], errors="coerce").dt.normalize()
    frame["sector"] = frame["sector"].astype("string").str.strip()
    invalid = frame["date"].isna() | frame["sector"].isna() | frame["sector"].eq("")
    frame.attrs["invalid_date_or_sector_rows_dropped"] = int(invalid.sum())
    frame = frame.loc[~invalid].copy()
    frame["sector"] = frame["sector"].astype(str)
    for col in OPTIONAL_NUMERIC:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    duplicate = frame.duplicated(["date", "sector"], keep=False)
    if duplicate.any():
        sample = frame.loc[duplicate, ["date", "sector"]].head(8).to_dict("records")
        raise ValueError(
            "Duplicate (date, sector) records found. Aggregate/fix these upstream rather than "
            f"silently choosing one. Examples: {sample}"
        )
    if frame.empty:
        raise ValueError("No usable rows remain after parsing date and sector.")
    if not frame["m7_C"].notna().any():
        raise ValueError("m7_C contains no numeric observations.")
    frame = frame.sort_values(["sector", "date"]).reset_index(drop=True)
    frame.attrs["source_path"] = str(source.resolve())
    return frame


def complete_calendar_years(frame: pd.DataFrame) -> list[int]:
    """Return years with all sector × calendar-day rows present and m7_C observed."""
    sectors = set(frame["sector"].astype(str).unique())
    years: list[int] = []
    for year, group in frame.groupby(frame["date"].dt.year):
        start = pd.Timestamp(year=int(year), month=1, day=1)
        end = pd.Timestamp(year=int(year), month=12, day=31)
        expected_days = len(pd.date_range(start, end, freq="D"))
        per_sector = group.groupby("sector")["date"].nunique()
        sector_set = set(group["sector"].astype(str).unique())
        values_complete = bool(group["m7_C"].notna().all())
        if (
            sector_set == sectors
            and len(per_sector) == len(sectors)
            and bool((per_sector == expected_days).all())
            and len(group) == expected_days * len(sectors)
            and values_complete
        ):
            years.append(int(year))
    return sorted(years)
