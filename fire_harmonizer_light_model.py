#!/usr/bin/env python3
"""Fire Harmonizer lightweight multi-year, walk-forward activity model.

Optional research add-on; does not replace the M7 C/F/G/S/U evidence system or M8 events.
Target: whether the same sector has any m7_C >= threshold on one of the next N calendar days.
This is future M7 anomaly activity, not confirmed fire occurrence, burned area, or a calibrated risk probability.

Dependencies: pandas, numpy, scikit-learn, joblib (joblib normally comes with scikit-learn).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, precision_recall_curve)

MODEL_NAME = "HistGradientBoostingClassifier"
DEFAULT_HORIZON_DAYS = 7
DEFAULT_C_THRESHOLD = 2.0


def json_safe(x: Any) -> Any:
    if isinstance(x, dict): return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [json_safe(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, (pd.Timestamp,)): return x.isoformat()
    return x


def load_science(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Science CSV not found: {path}\nPass --science or place it at output/harmonized_science_all_years.csv")
    df = pd.read_csv(path, low_memory=False)
    date_col = "acq_date" if "acq_date" in df.columns else ("date" if "date" in df.columns else None)
    missing = [c for c in ("sector", "m7_C") if c not in df.columns]
    if date_col is None or missing:
        raise ValueError(f"Need date/acq_date, sector, m7_C. Missing columns: {missing}; date column found: {date_col}")
    df = df.copy()
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df["sector"] = df["sector"].astype("string").str.strip()
    for c in ("m7_C", "m7_F", "m7_G", "m7_S", "m3_harmonized_activity", "m3_anomaly_z"):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "sector", "m7_C"]).copy()
    df["sector"] = df["sector"].astype(str)
    dup = df.duplicated(["sector", "date"], keep=False)
    if dup.any():
        examples = df.loc[dup, ["date", "sector"]].head(5).to_dict("records")
        raise ValueError(f"Duplicate sector-day rows found; aggregate them upstream. Examples: {examples}")
    df = df.sort_values(["sector", "date"]).reset_index(drop=True)
    if df.empty: raise ValueError("No valid sector-day rows after parsing.")
    return df


def multikey(sectors: pd.Series, dates: pd.Series) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([sectors.astype(str).to_numpy(), pd.to_datetime(dates).to_numpy()], names=["sector", "date"])


def make_features(df: pd.DataFrame, c_threshold: float) -> tuple[pd.DataFrame, list[str]]:
    """Predictors use only values available on the current date or earlier."""
    work = df.copy().sort_values(["sector", "date"]).reset_index(drop=True)
    feature = pd.DataFrame(index=work.index)
    keyed = work.set_index(["sector", "date"])
    current_cols = [c for c in ("m7_C", "m7_F", "m7_G", "m7_S", "m3_harmonized_activity", "m3_anomaly_z") if c in work.columns]
    for col in current_cols: feature[f"now_{col}"] = pd.to_numeric(work[col], errors="coerce").to_numpy()

    # Exact calendar-day lags; missing dates remain missing rather than shifting to the previous row.
    lag_plan = {"m7_C": [1, 2, 3, 7, 14], "m7_F": [1, 3, 7], "m3_harmonized_activity": [1, 7], "m3_anomaly_z": [1, 7]}
    for col, lags in lag_plan.items():
        if col not in keyed.columns: continue
        for lag in lags:
            past_idx = multikey(work["sector"], work["date"] - pd.to_timedelta(lag, unit="D"))
            feature[f"{col}_lag_{lag}d"] = keyed[col].reindex(past_idx).to_numpy()

    # Calendar-window summaries, including current date and earlier observations only.
    for col, windows in {"m7_C": [3, 7, 30], "m7_F": [7, 30]}.items():
        if col not in work.columns: continue
        for window in windows:
            out = np.full(len(work), np.nan, dtype=float)
            for _, group in work.groupby("sector", sort=False):
                values = group.set_index("date")[col].sort_index().rolling(f"{window}D", min_periods=1).mean()
                out[group.index.to_numpy()] = values.to_numpy(dtype=float)
            feature[f"{col}_rolling_mean_{window}d"] = out

    # FIXED: real current/past 7-calendar-day activity feature, not an all-NaN placeholder.
    peak = np.full(len(work), np.nan, dtype=float)
    active = np.full(len(work), np.nan, dtype=float)
    for _, group in work.groupby("sector", sort=False):
        g = group.set_index("date").sort_index()
        rolling_peak = g["m7_C"].rolling("7D", min_periods=1).max()
        rolling_active = (g["m7_C"] >= c_threshold).astype(float).rolling("7D", min_periods=1).max()
        peak[group.index.to_numpy()] = rolling_peak.to_numpy(dtype=float)
        active[group.index.to_numpy()] = rolling_active.to_numpy(dtype=float)
    feature["past_7d_peak_C"] = peak
    feature["past_7d_has_C_active"] = active

    doy = work["date"].dt.dayofyear.to_numpy(dtype=float)
    feature["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    feature["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    dummies = pd.get_dummies(work["sector"], prefix="sector", dtype=np.uint8)
    feature = pd.concat([feature, dummies], axis=1).replace([np.inf, -np.inf], np.nan)
    return feature.astype(float), list(feature.columns)


def make_future_target(df: pd.DataFrame, horizon: int, threshold: float) -> tuple[pd.Series, pd.Series]:
    work = df.reset_index(drop=True)
    keyed = work.set_index(["sector", "date"])["m7_C"]
    future = {}
    for offset in range(1, horizon + 1):
        ix = multikey(work["sector"], work["date"] + pd.to_timedelta(offset, unit="D"))
        future[offset] = keyed.reindex(ix).to_numpy(dtype=float)
    fut = pd.DataFrame(future, index=work.index)
    valid = fut.notna().all(axis=1)
    target = fut.ge(threshold).any(axis=1).astype("int8")
    return target, valid


def complete_calendar_years(df: pd.DataFrame) -> list[int]:
    years = []
    sectors = set(df["sector"].unique())
    for year, group in df.groupby(df["date"].dt.year):
        days = 366 if pd.Timestamp(year=int(year), month=12, day=31).dayofyear == 366 else 365
        if group["date"].nunique() == days and len(group) == days * len(sectors):
            per_day = group.groupby("date")["sector"].nunique()
            if set(group["sector"].unique()) == sectors and (per_day == len(sectors)).all(): years.append(int(year))
    return sorted(years)


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(learning_rate=0.08, max_iter=120, max_leaf_nodes=15,
        min_samples_leaf=35, l2_regularization=1.0, max_bins=127,
        early_stopping=False, random_state=42)


def fit_model(X: pd.DataFrame, y: pd.Series) -> HistGradientBoostingClassifier:
    values = y.to_numpy(dtype=int)
    counts = np.bincount(values, minlength=2)
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(f"Training data needs both target classes; negatives={counts[0]}, positives={counts[1]}.")
    weights = np.where(values == 1, len(values) / (2 * counts[1]), len(values) / (2 * counts[0]))
    model = make_model()
    model.fit(X, values, sample_weight=weights)
    return model


def choose_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2: return 0.5, float("nan")
    p, r, thresholds = precision_recall_curve(y, scores)
    if len(thresholds) == 0: return 0.5, float("nan")
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    i = int(np.nanargmax(f1))
    return float(thresholds[i]), float(f1[i])


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    out = {"rows": int(len(y)), "positive_rate": float(np.mean(y)) if len(y) else None,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)), "threshold": float(threshold)}
    out["average_precision"] = float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else None
    out["balanced_accuracy"] = float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else None
    return out


def annual_walk_forward(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, label_valid: pd.Series,
                        horizon: int, complete_years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each test year is predicted using earlier years; immediately preceding year selects method/threshold."""
    prediction_frames, summary_rows = [], []
    year_values = df["date"].dt.year
    valid_arr = label_valid.to_numpy(dtype=bool)
    # At least 3 complete years before/including validation and one future test year.
    for test_year in complete_years[3:]:
        val_year = test_year - 1
        if val_year not in complete_years: continue
        val_start = pd.Timestamp(val_year, 1, 1)
        test_start = pd.Timestamp(test_year, 1, 1)
        train = (year_values < val_year) & (df["date"] < val_start - pd.Timedelta(days=horizon)) & valid_arr
        val = (year_values == val_year) & (df["date"] < test_start - pd.Timedelta(days=horizon)) & valid_arr
        test = (year_values == test_year) & valid_arr
        if train.sum() == 0 or val.sum() == 0 or test.sum() == 0: continue
        if y.loc[train].nunique() < 2 or y.loc[val].nunique() < 2 or y.loc[test].nunique() < 2: continue
        model = fit_model(X.loc[train], y.loc[train])
        vy = y.loc[val].to_numpy(dtype=int)
        vml = model.predict_proba(X.loc[val])[:, 1]
        ml_threshold, _ = choose_threshold(vy, vml)
        vp = X.loc[val, "past_7d_peak_C"].fillna(0).to_numpy(dtype=float)
        p_threshold, _ = choose_threshold(vy, vp)
        ml_val_f1 = metrics(vy, vml, ml_threshold)["f1"]
        p_val_f1 = metrics(vy, vp, p_threshold)["f1"]
        method = "hist_gradient_boosting" if ml_val_f1 >= p_val_f1 else "persistence_baseline"
        threshold = ml_threshold if method == "hist_gradient_boosting" else p_threshold
        ty = y.loc[test].to_numpy(dtype=int)
        ml_scores = model.predict_proba(X.loc[test])[:, 1]
        pscores = X.loc[test, "past_7d_peak_C"].fillna(0).to_numpy(dtype=float)
        chosen = ml_scores if method == "hist_gradient_boosting" else pscores
        year_metrics = metrics(ty, chosen, threshold)
        summary_rows.append({"test_year": int(test_year), "validation_year": int(val_year),
            "training_start_year": int(df.loc[train, "date"].dt.year.min()),
            "training_end_year": int(df.loc[train, "date"].dt.year.max()),
            "train_rows": int(train.sum()), "validation_rows": int(val.sum()), "test_rows": int(test.sum()),
            "validation_ml_f1": ml_val_f1, "validation_persistence_f1": p_val_f1,
            "selected_method": method, **{f"test_{k}": v for k, v in year_metrics.items() if k != "threshold"},
            "threshold": threshold})
        out = df.loc[test, ["date", "sector", "m7_C"]].copy()
        out["observed_future_active"] = ty
        out["test_year"] = int(test_year)
        out["validation_year"] = int(val_year)
        out["selected_method"] = method
        out["selected_score"] = chosen
        out["selected_flag"] = (chosen >= threshold).astype(int)
        out["ml_score"] = ml_scores
        out["ml_flag"] = (ml_scores >= ml_threshold).astype(int)
        out["persistence_score"] = pscores
        out["persistence_flag"] = (pscores >= p_threshold).astype(int)
        prediction_frames.append(out)
    pred_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise ValueError("No annual walk-forward folds could be evaluated. Need at least four complete calendar years with both target classes in each fold.")
    return pred_df, summary_df


def run_model(science_path: Path, output_dir: Path, horizon: int = DEFAULT_HORIZON_DAYS,
              c_threshold: float = DEFAULT_C_THRESHOLD) -> dict[str, Any]:
    if horizon < 1: raise ValueError("--horizon must be at least 1 day")
    df = load_science(science_path)
    X, feature_names = make_features(df, c_threshold)
    y, label_valid = make_future_target(df, horizon, c_threshold)
    complete_years = complete_calendar_years(df)

    # Multi-year rolling-origin backtest: each evaluated year is held out once.
    all_predictions, annual_summary = annual_walk_forward(df, X, y, label_valid, horizon, complete_years)
    overall_y = all_predictions["observed_future_active"].to_numpy(dtype=int)
    overall_scores = all_predictions["selected_score"].to_numpy(dtype=float)
    overall_flags = all_predictions["selected_flag"].to_numpy(dtype=int)
    overall_metrics = {"rows": int(len(overall_y)), "positive_rate": float(np.mean(overall_y)),
        "precision": float(precision_score(overall_y, overall_flags, zero_division=0)),
        "recall": float(recall_score(overall_y, overall_flags, zero_division=0)),
        "f1": float(f1_score(overall_y, overall_flags, zero_division=0)),
        "average_precision": float(average_precision_score(overall_y, overall_scores)) if len(np.unique(overall_y)) > 1 else None,
        "test_years": sorted(int(x) for x in all_predictions["test_year"].unique())}

    # Final model/threshold choice: use the latest complete year as validation, then fit final model on all label-valid history.
    if len(complete_years) < 3: raise ValueError("Need at least 3 complete calendar years for final model selection and training.")
    val_year = complete_years[-1]
    val_start = pd.Timestamp(val_year, 1, 1)
    train_for_choice = (df["date"].dt.year < val_year) & (df["date"] < val_start - pd.Timedelta(days=horizon)) & label_valid
    val_for_choice = (df["date"].dt.year == val_year) & label_valid
    if train_for_choice.sum() == 0 or val_for_choice.sum() == 0 or y.loc[train_for_choice].nunique() < 2 or y.loc[val_for_choice].nunique() < 2:
        raise ValueError("Could not choose final model/threshold from latest complete year. Check coverage and target class balance.")
    selection_model = fit_model(X.loc[train_for_choice], y.loc[train_for_choice])
    vy = y.loc[val_for_choice].to_numpy(dtype=int)
    val_ml = selection_model.predict_proba(X.loc[val_for_choice])[:, 1]
    ml_threshold, ml_best_f1 = choose_threshold(vy, val_ml)
    val_persist = X.loc[val_for_choice, "past_7d_peak_C"].fillna(0).to_numpy(dtype=float)
    persist_threshold, persist_best_f1 = choose_threshold(vy, val_persist)
    selected_method = "hist_gradient_boosting" if metrics(vy, val_ml, ml_threshold)["f1"] >= metrics(vy, val_persist, persist_threshold)["f1"] else "persistence_baseline"
    selected_threshold = ml_threshold if selected_method == "hist_gradient_boosting" else persist_threshold
    reason = "ML matched/exceeded persistence F1 on the latest complete validation year" if selected_method == "hist_gradient_boosting" else "Persistence achieved higher F1 on the latest complete validation year"

    # Refit final ML model on all rows with fully observed future labels (all available years).
    max_date = df["date"].max()
    final_mask = label_valid.to_numpy(dtype=bool)
    final_model = fit_model(X.loc[final_mask], y.loc[final_mask])
    latest_date = max_date
    latest_mask = df["date"].eq(latest_date)
    latest = df.loc[latest_mask, ["date", "sector"]].copy()
    if latest.empty: raise ValueError("No rows on latest input date")
    latest_ml = final_model.predict_proba(X.loc[latest.index])[:, 1]
    latest_persist = X.loc[latest.index, "past_7d_peak_C"].fillna(0).to_numpy(dtype=float)
    latest_scores = latest_ml if selected_method == "hist_gradient_boosting" else latest_persist
    latest["as_of_date"] = latest_date.date().isoformat()
    latest["forecast_start"] = (latest_date + pd.Timedelta(days=1)).date().isoformat()
    latest["forecast_end"] = (latest_date + pd.Timedelta(days=horizon)).date().isoformat()
    latest["horizon_days"] = horizon
    latest["selected_method"] = selected_method
    latest["model_score"] = latest_scores
    latest["ml_score"] = latest_ml
    latest["persistence_score"] = latest_persist
    latest["rank"] = latest["model_score"].rank(method="first", ascending=False).astype(int)
    latest["watchlist_flag"] = (latest["model_score"] >= selected_threshold).astype(int)
    latest["score_threshold_from_validation"] = selected_threshold
    latest = latest.sort_values(["model_score", "sector"], ascending=[False, True]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    p_annual = output_dir / "annual_walk_forward_metrics.csv"
    p_predictions = output_dir / "backtest_all_years_predictions.csv"
    p_watch = output_dir / "next7d_watchlist.csv"
    p_metrics = output_dir / "model_metrics.json"
    p_model = output_dir / "fire_harmonizer_light_model.joblib"
    annual_summary.to_csv(p_annual, index=False)
    all_predictions.to_csv(p_predictions, index=False, date_format="%Y-%m-%d")
    latest.to_csv(p_watch, index=False)

    report = {
        "model_name": MODEL_NAME,
        "purpose": "experimental multi-year walk-forward forecast of future M7 C-active sector-days",
        "target_definition": f"1 if same sector has any m7_C >= {c_threshold:g} in the next {horizon} calendar days",
        "horizon_days": horizon, "C_active_threshold": c_threshold,
        "input_file": str(science_path.resolve()), "input_rows": len(df), "sector_count": int(df["sector"].nunique()),
        "input_start_date": df["date"].min().date().isoformat(), "input_end_date": max_date.date().isoformat(),
        "feature_count": len(feature_names), "features": feature_names,
        "complete_calendar_years": complete_years,
        "walk_forward_backtest": overall_metrics,
        "annual_metrics_file": str(p_annual), "all_year_predictions_file": str(p_predictions),
        "annual_fold_summary": annual_summary.to_dict(orient="records"),
        "final_method_selection": {"validation_year": val_year, "training_rows_for_selection": int(train_for_choice.sum()),
            "validation_rows": int(val_for_choice.sum()), "ml_validation_best_f1": ml_best_f1,
            "persistence_validation_best_f1": persist_best_f1, "selected_method": selected_method,
            "selected_threshold": selected_threshold, "reason": reason},
        "final_model_training_rows": int(final_mask.sum()),
        "final_model_training_years": sorted(int(v) for v in df.loc[final_mask, "date"].dt.year.unique()),
        "latest_as_of_date": latest_date.date().isoformat(),
        "latest_forecast_window": {"start": (latest_date + pd.Timedelta(days=1)).date().isoformat(), "end": (latest_date + pd.Timedelta(days=horizon)).date().isoformat()},
        "interpretation_warning": "Scores are ranking scores, not calibrated probabilities. The target is future M7 C-active anomaly activity, not confirmed fires, burned area, fire severity, or an operational alert. If M7 uses retrospective baselines, forecasts of M7 anomalies are not fully causal real-time forecasts. Each walk-forward test year is predicted using only earlier years; threshold/method are chosen from the immediately preceding validation year.",
    }
    p_metrics.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    try:
        import joblib
        joblib.dump({"model": final_model, "feature_names": feature_names, "horizon_days": horizon,
            "c_threshold": c_threshold, "score_threshold": selected_threshold,
            "selected_method": selected_method, "model_name": MODEL_NAME}, p_model, compress=3)
    except Exception as exc:
        print(f"Warning: could not save model artifact: {exc}", file=sys.stderr)
        p_model = None

    print("\nFire Harmonizer multi-year lightweight model complete")
    print(f"Rows: {len(df):,} | sectors: {df['sector'].nunique()} | features: {len(feature_names)}")
    print(f"Input dates: {df['date'].min().date()} to {max_date.date()} | complete years: {complete_years}")
    print(f"Walk-forward test years: {overall_metrics['test_years']}")
    print("\nAggregated out-of-year metrics (all walk-forward test years):")
    print(json.dumps(json_safe(overall_metrics), indent=2))
    print("\nAnnual walk-forward summary:")
    print(annual_summary.to_string(index=False))
    print(f"\nFinal selected method: {selected_method}; threshold={selected_threshold:.4f}; {reason}")
    print("\nLatest-sector watchlist:")
    print(latest[["sector", "model_score", "rank", "watchlist_flag", "selected_method"]].to_string(index=False))
    print(f"\nWrote: {p_annual}\nWrote: {p_predictions}\nWrote: {p_watch}\nWrote: {p_metrics}")
    if p_model: print(f"Wrote: {p_model}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--science", type=Path, default=Path("output/harmonized_science_all_years.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/model"))
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--c-threshold", type=float, default=DEFAULT_C_THRESHOLD)
    args = parser.parse_args()
    run_model(args.science, args.output_dir, args.horizon, args.c_threshold)
main()