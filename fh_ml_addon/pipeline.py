from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import __version__
from .data import complete_calendar_years, load_science
from .features import build_features
from .modeling import best_f1_threshold, fit_model, metric_dict, persistence_scores
from .targets import make_target

HORIZONS = (7, 14, 30)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False), encoding="utf-8")


def _year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year, month=12, day=31)


def _select_method(
    y_val: pd.Series,
    ml_score: np.ndarray | None,
    persistence_score: np.ndarray,
) -> dict[str, Any]:
    persistence_threshold, persistence_f1 = best_f1_threshold(y_val, persistence_score)
    result: dict[str, Any] = {
        "persistence_threshold": persistence_threshold,
        "persistence_validation_f1": persistence_f1,
        "ml_threshold": 0.5,
        "ml_validation_f1": None,
        "selected_method": "persistence_baseline",
        "selected_threshold": persistence_threshold,
        "selection_reason": "Persistence baseline selected by default until a valid ML validation fit is available.",
    }
    if ml_score is not None:
        ml_threshold, ml_f1 = best_f1_threshold(y_val, ml_score)
        result["ml_threshold"] = ml_threshold
        result["ml_validation_f1"] = ml_f1
        if ml_f1 > persistence_f1 + 1e-12:
            result["selected_method"] = "hist_gradient_boosting"
            result["selected_threshold"] = ml_threshold
            result["selection_reason"] = "ML achieved higher validation F1 than the persistence baseline."
        else:
            result["selection_reason"] = "Persistence baseline matched or exceeded ML validation F1; simpler method retained."
    return result


def _aggregate_from_folds(predictions: pd.DataFrame, score_col: str, flag_col: str) -> dict[str, Any]:
    if predictions.empty:
        return {"n": 0, "positive_rate": None, "average_precision": None,
                "precision": None, "recall": None, "f1": None, "balanced_accuracy": None}
    y = predictions["observed_target"].astype(int).to_numpy()
    s = pd.to_numeric(predictions[score_col], errors="coerce").to_numpy(dtype=float)
    flags = pd.to_numeric(predictions[flag_col], errors="coerce").fillna(0).to_numpy(dtype=int)
    keep = np.isfinite(s)
    y, s, flags = y[keep], s[keep], flags[keep]
    two_classes = len(np.unique(y)) == 2 if len(y) else False
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) else None,
        "average_precision": float(average_precision_score(y, s)) if two_classes else None,
        "precision": float(precision_score(y, flags, zero_division=0)) if len(y) else None,
        "recall": float(recall_score(y, flags, zero_division=0)) if len(y) else None,
        "f1": float(f1_score(y, flags, zero_division=0)) if len(y) else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, flags)) if two_classes else None,
        "note": "Precision/recall/F1 use validation-selected thresholds for each annual fold; AP uses the model scores.",
    }


def _fit_valid_model(x: pd.DataFrame, y: pd.Series):
    if len(y) < 100 or y.nunique() < 2:
        return None
    try:
        return fit_model(x, y)
    except (ValueError, FloatingPointError):
        return None


def run_pipeline(
    science_path: Path | str,
    output_dir: Path | str,
    c_threshold: float = 2.0,
    elevated_min_active_days: int = 3,
    horizons: tuple[int, ...] = HORIZONS,
    save_feature_table: bool = False,
    status_callback: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Run feature generation, annual walk-forward evaluation, final fits and watchlists.

    All new outputs are written only beneath output_dir. This add-on does not modify
    Fire Harmonizer's original science CSV, existing model folder, or app code.
    """
    def status(stage: str, message: str) -> None:
        if status_callback:
            status_callback(stage, message)
        print(f"[{stage}] {message}", flush=True)

    source = Path(science_path).resolve()
    outdir = Path(output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    status("loading", f"Loading science data from {source}")
    df = load_science(source)
    input_hash = sha256_file(source)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    invalid_rows_dropped = int(df.attrs.get("invalid_date_or_sector_rows_dropped", 0))
    # Data contract loader sorts; explicitly preserve this row order for alignment.
    df = df.sort_values(["sector", "date"]).reset_index(drop=True)
    years_present = sorted(int(y) for y in df["date"].dt.year.unique())
    complete_years = complete_calendar_years(df)
    max_date = pd.Timestamp(df["date"].max()).normalize()
    min_date = pd.Timestamp(df["date"].min()).normalize()
    status("features", f"Generating calendar lags, rolling statistics and seasonality for {len(df):,} sector-days")
    feature_table, feature_names = build_features(df, c_threshold=c_threshold)
    x = feature_table.loc[:, feature_names].copy()
    if save_feature_table:
        feature_table.to_csv(outdir / "feature_table.csv", index=False, date_format="%Y-%m-%d")
    feature_schema = {
        "schema_version": "1.0",
        "package_version": __version__,
        "issue_time_convention": "The t-dated row is forecast after t's available harmonized observations are produced.",
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "feature_notes": [
            "Lags are exact calendar-day lookups; missing dates remain missing.",
            "Rolling windows are calendar-duration windows and include the as-of day.",
            "Sector one-hot columns represent sectors present in the source matrix.",
            "m7_C threshold is used as supplied; a value >= 2 is an anomaly-evidence criterion, not confirmed fire ground truth.",
        ],
    }
    _write_json(outdir / "feature_schema.json", feature_schema)

    all_annual_rows: list[dict[str, Any]] = []
    all_prediction_frames: list[pd.DataFrame] = []
    horizon_reports: dict[str, Any] = {}
    # Immutable, run-specific model folders; latest watchlists/model card are pointers to this run.
    model_folder = outdir / "models" / run_id
    model_folder.mkdir(parents=True, exist_ok=False)

    for horizon in horizons:
        if horizon not in HORIZONS:
            raise ValueError(f"Invalid horizon {horizon}; supported values are {HORIZONS}.")
        status(f"horizon_{horizon}d", f"Building labels and evaluating {horizon}-day horizon")
        y_all, valid_all, target_definition = make_target(
            df, horizon, c_threshold=c_threshold, elevated_min_active_days=elevated_min_active_days
        )
        active_valid = valid_all.to_numpy(dtype=bool)
        y_values = y_all.to_numpy(dtype=int)
        feature_meta = feature_table[["date", "sector"]].copy()
        persistence = persistence_scores(feature_table)
        predictions_for_horizon: list[pd.DataFrame] = []
        annual_rows: list[dict[str, Any]] = []

        # Every test year uses the immediately preceding complete year for method and
        # threshold selection; training labels are purged so their future window ends
        # before validation starts. Validation windows are fully contained in that year.
        possible_test_years = [
            year for year in complete_years
            if year - 1 in complete_years and sum(prev < year - 1 for prev in complete_years) >= 2
        ]
        for test_year in possible_test_years:
            validation_year = test_year - 1
            validation_start, _ = _year_bounds(validation_year)
            _, validation_end = _year_bounds(validation_year)
            train_cutoff = validation_start - pd.Timedelta(days=horizon + 1)
            validation_cutoff = validation_end - pd.Timedelta(days=horizon)
            train_mask = active_valid & (feature_meta["date"] <= train_cutoff).to_numpy()
            val_mask = (
                active_valid
                & (feature_meta["date"] >= validation_start).to_numpy()
                & (feature_meta["date"] <= validation_cutoff).to_numpy()
            )
            test_mask = active_valid & (feature_meta["date"].dt.year == test_year).to_numpy()
            train_idx = np.flatnonzero(train_mask)
            val_idx = np.flatnonzero(val_mask)
            test_idx = np.flatnonzero(test_mask)
            if len(train_idx) < 100 or len(val_idx) < 30 or len(test_idx) < 1:
                continue
            y_train = pd.Series(y_values[train_idx])
            y_val = pd.Series(y_values[val_idx])
            if y_train.nunique() < 2 or y_val.nunique() < 2:
                continue
            fold_model = _fit_valid_model(x.iloc[train_idx], y_train)
            val_ml_score = fold_model.predict_proba(x.iloc[val_idx])[:, 1] if fold_model is not None else None
            selection = _select_method(y_val, val_ml_score, persistence[val_idx])
            test_ml_score = (
                fold_model.predict_proba(x.iloc[test_idx])[:, 1]
                if fold_model is not None else np.full(len(test_idx), np.nan)
            )
            test_persistence_score = persistence[test_idx]
            ml_threshold = float(selection["ml_threshold"])
            persistence_threshold = float(selection["persistence_threshold"])
            ml_metrics = metric_dict(y_values[test_idx], test_ml_score, ml_threshold)
            persistence_metrics = metric_dict(y_values[test_idx], test_persistence_score, persistence_threshold)
            if selection["selected_method"] == "hist_gradient_boosting" and fold_model is not None:
                selected_score = test_ml_score
                selected_threshold = float(selection["selected_threshold"])
            else:
                selected_score = test_persistence_score
                selected_threshold = float(selection["selected_threshold"])
            selected_metrics = metric_dict(y_values[test_idx], selected_score, selected_threshold)

            rows = feature_meta.iloc[test_idx].copy()
            rows["horizon_days"] = horizon
            rows["test_year"] = test_year
            rows["validation_year"] = validation_year
            rows["observed_target"] = y_values[test_idx]
            rows["ml_score"] = test_ml_score
            rows["ml_flag"] = (test_ml_score >= ml_threshold).astype(int) if fold_model is not None else 0
            rows["ml_threshold_from_validation"] = ml_threshold
            rows["persistence_score"] = test_persistence_score
            rows["persistence_flag"] = (test_persistence_score >= persistence_threshold).astype(int)
            rows["persistence_threshold_from_validation"] = persistence_threshold
            rows["selected_method"] = selection["selected_method"]
            rows["selected_score"] = selected_score
            rows["selected_flag"] = (selected_score >= selected_threshold).astype(int)
            rows["selected_threshold_from_validation"] = selected_threshold
            predictions_for_horizon.append(rows)
            predictions_summary = {
                "horizon_days": horizon,
                "test_year": test_year,
                "validation_year": validation_year,
                "training_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "test_rows": int(len(test_idx)),
                "training_start_date": str(feature_meta.iloc[train_idx]["date"].min().date()),
                "training_end_date": str(feature_meta.iloc[train_idx]["date"].max().date()),
                "training_years": sorted(int(v) for v in feature_meta.iloc[train_idx]["date"].dt.year.unique()),
                "validation_start_date": str(feature_meta.iloc[val_idx]["date"].min().date()),
                "validation_end_date": str(feature_meta.iloc[val_idx]["date"].max().date()),
                "selected_method": selection["selected_method"],
                "selection_reason": selection["selection_reason"],
                "validation_ml_f1": selection["ml_validation_f1"],
                "validation_persistence_f1": selection["persistence_validation_f1"],
                "test_ml_metrics": ml_metrics,
                "test_persistence_metrics": persistence_metrics,
                "test_selected_metrics": selected_metrics,
            }
            annual_rows.append(predictions_summary)
            annual_flat = {
                k: v for k, v in predictions_summary.items()
                if k not in ("test_ml_metrics", "test_persistence_metrics", "test_selected_metrics")
            }
            for prefix, metric_value in (
                ("ml", ml_metrics), ("persistence", persistence_metrics), ("selected", selected_metrics)
            ):
                for metric, value in metric_value.items():
                    annual_flat[f"test_{prefix}_{metric}"] = value
            all_annual_rows.append(annual_flat)

        if predictions_for_horizon:
            horizon_predictions = pd.concat(predictions_for_horizon, ignore_index=True)
            all_prediction_frames.append(horizon_predictions)
        else:
            horizon_predictions = pd.DataFrame(columns=[
                "date", "sector", "horizon_days", "test_year", "validation_year", "observed_target",
                "ml_score", "ml_flag", "persistence_score", "persistence_flag", "selected_method",
                "selected_score", "selected_flag",
            ])

        # Final operational selection: use the most recent complete calendar year as
        # validation, then refit the selected model on every fully labelled historical row.
        validation_year = max(complete_years) if complete_years else None
        final_selection: dict[str, Any] = {
            "validation_year": validation_year,
            "selected_method": "persistence_baseline",
            "selected_threshold": 1.0,
            "ml_threshold": 0.5,
            "persistence_threshold": 1.0,
            "ml_validation_f1": None,
            "persistence_validation_f1": None,
            "selection_reason": "Insufficient complete chronological validation data; persistence baseline retained.",
            "training_rows_for_selection": 0,
            "validation_rows": 0,
        }
        if validation_year is not None:
            val_start, val_end = _year_bounds(validation_year)
            train_cutoff = val_start - pd.Timedelta(days=horizon + 1)
            val_cutoff = val_end - pd.Timedelta(days=horizon)
            final_train_mask = active_valid & (feature_meta["date"] <= train_cutoff).to_numpy()
            final_val_mask = (
                active_valid
                & (feature_meta["date"] >= val_start).to_numpy()
                & (feature_meta["date"] <= val_cutoff).to_numpy()
            )
            ftrain_idx = np.flatnonzero(final_train_mask)
            fval_idx = np.flatnonzero(final_val_mask)
            if len(ftrain_idx) >= 100 and len(fval_idx) >= 30 and pd.Series(y_values[ftrain_idx]).nunique() == 2 and pd.Series(y_values[fval_idx]).nunique() == 2:
                final_selection_model = _fit_valid_model(x.iloc[ftrain_idx], pd.Series(y_values[ftrain_idx]))
                val_score = (
                    final_selection_model.predict_proba(x.iloc[fval_idx])[:, 1]
                    if final_selection_model is not None else None
                )
                final_selection.update(_select_method(pd.Series(y_values[fval_idx]), val_score, persistence[fval_idx]))
                final_selection["training_rows_for_selection"] = int(len(ftrain_idx))
                final_selection["training_years_for_selection"] = sorted(int(v) for v in feature_meta.iloc[ftrain_idx]["date"].dt.year.unique())
                final_selection["validation_rows"] = int(len(fval_idx))

        # Refit ML using labels fully observed by the end of this source dataset.
        labelled_idx = np.flatnonzero(active_valid)
        final_model = _fit_valid_model(x.iloc[labelled_idx], pd.Series(y_values[labelled_idx])) if len(labelled_idx) else None
        if final_model is None and final_selection["selected_method"] == "hist_gradient_boosting":
            final_selection["selected_method"] = "persistence_baseline"
            final_selection["selected_threshold"] = final_selection["persistence_threshold"]
            final_selection["selection_reason"] += " Final ML fit unavailable; fell back to persistence."

        latest_date = max_date
        latest_idx = np.flatnonzero((feature_meta["date"] == latest_date).to_numpy())
        latest_df = feature_meta.iloc[latest_idx].copy()
        if len(latest_idx):
            latest_ml = (
                final_model.predict_proba(x.iloc[latest_idx])[:, 1]
                if final_model is not None else np.full(len(latest_idx), np.nan)
            )
            latest_persistence = persistence[latest_idx]
            if final_selection["selected_method"] == "hist_gradient_boosting" and final_model is not None:
                latest_score = latest_ml
            else:
                latest_score = latest_persistence
            start_date = (latest_date + pd.Timedelta(days=1)).date().isoformat()
            end_date = (latest_date + pd.Timedelta(days=horizon)).date().isoformat()
            latest_df["as_of_date"] = latest_date.date().isoformat()
            latest_df["forecast_start"] = start_date
            latest_df["forecast_end"] = end_date
            latest_df["horizon_days"] = horizon
            latest_df["target_definition"] = target_definition
            latest_df["selected_method"] = final_selection["selected_method"]
            latest_df["model_score"] = latest_score
            latest_df["ml_score"] = latest_ml
            latest_df["persistence_score"] = latest_persistence
            latest_df["score_type"] = (
                "uncalibrated_model_score" if final_selection["selected_method"] == "hist_gradient_boosting"
                else "prior_7d_active_day_count"
            )
            latest_df["score_threshold_from_validation"] = float(final_selection["selected_threshold"])
            latest_df["watchlist_flag"] = (latest_score >= float(final_selection["selected_threshold"])).astype(int)
            latest_df["rank"] = latest_df["model_score"].rank(method="first", ascending=False).astype(int)
            latest_df = latest_df.sort_values(["model_score", "sector"], ascending=[False, True]).reset_index(drop=True)
        latest_path = outdir / ({7: "watchlist_7d.csv", 14: "watchlist_14d.csv", 30: "outlook_30d.csv"}[horizon])
        latest_df.to_csv(latest_path, index=False, date_format="%Y-%m-%d")

        artifact_path = model_folder / f"fire_harmonizer_{horizon}d.joblib"
        model_bundle = {
            "model": final_model,
            "feature_names": feature_names,
            "horizon_days": horizon,
            "c_threshold": float(c_threshold),
            "elevated_min_active_days": int(elevated_min_active_days),
            "target_definition": target_definition,
            "selected_method": final_selection["selected_method"],
            "selected_threshold": float(final_selection["selected_threshold"]),
            "ml_threshold": float(final_selection["ml_threshold"]),
            "persistence_threshold": float(final_selection["persistence_threshold"]),
            "package_version": __version__,
            "run_id": run_id,
            "model_version": f"{__version__}+{run_id}",
            "input_sha256_at_training": input_hash,
            "training_end_date": max_date.date().isoformat(),
            "final_model_training_rows": int(len(labelled_idx)),
            "final_model_training_years": sorted(int(v) for v in feature_meta.iloc[labelled_idx]["date"].dt.year.unique()),
            "score_is_calibrated_probability": False,
        }
        joblib.dump(model_bundle, artifact_path, compress=3)

        if not horizon_predictions.empty:
            ml_aggregate = _aggregate_from_folds(horizon_predictions, "ml_score", "ml_flag")
            persistence_aggregate = _aggregate_from_folds(horizon_predictions, "persistence_score", "persistence_flag")
            selected_aggregate = {
                "n": int(len(horizon_predictions)),
                "positive_rate": float(horizon_predictions["observed_target"].mean()),
                "precision": float((horizon_predictions["selected_flag"] * horizon_predictions["observed_target"]).sum() / max(1, horizon_predictions["selected_flag"].sum())),
                "recall": float((horizon_predictions["selected_flag"] * horizon_predictions["observed_target"]).sum() / max(1, horizon_predictions["observed_target"].sum())),
                "f1": float(2 * ((horizon_predictions["selected_flag"] * horizon_predictions["observed_target"]).sum()) / max(1, horizon_predictions["selected_flag"].sum() + horizon_predictions["observed_target"].sum())),
                "average_precision": None,
                "note": "Thresholds and method are selected independently for each annual validation fold; selected scores can therefore mix scales.",
            }
            tp = int((horizon_predictions["selected_flag"] * horizon_predictions["observed_target"]).sum())
            tn = int(((1 - horizon_predictions["selected_flag"]) * (1 - horizon_predictions["observed_target"])).sum())
            fp = int((horizon_predictions["selected_flag"] * (1 - horizon_predictions["observed_target"])).sum())
            fn = int(((1 - horizon_predictions["selected_flag"]) * horizon_predictions["observed_target"]).sum())
            selected_aggregate["balanced_accuracy"] = float(0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp)))
        else:
            ml_aggregate = persistence_aggregate = selected_aggregate = {"n": 0, "note": "No eligible walk-forward folds; see warnings."}

        horizon_reports[str(horizon)] = {
            "horizon_days": horizon,
            "target_definition": target_definition,
            "target_rows_eligible": int(active_valid.sum()),
            "target_positive_rate_eligible": float(y_values[active_valid].mean()) if active_valid.any() else None,
            "feature_count": len(feature_names),
            "walk_forward_fold_count": len(annual_rows),
            "walk_forward_folds": annual_rows,
            "walk_forward_aggregate": {
                "ml": ml_aggregate,
                "persistence": persistence_aggregate,
                "selected_per_fold_threshold": selected_aggregate,
            },
            "final_selection": final_selection,
            "final_model_training_rows": int(len(labelled_idx)),
            "final_model_training_years": sorted(int(v) for v in feature_meta.iloc[labelled_idx]["date"].dt.year.unique()),
            "model_artifact": str(artifact_path.relative_to(outdir)),
            "latest_output": str(latest_path.relative_to(outdir)),
            "model_available": final_model is not None,
        }

    annual_df = pd.DataFrame(all_annual_rows)
    prediction_df = pd.concat(all_prediction_frames, ignore_index=True) if all_prediction_frames else pd.DataFrame()
    annual_df.to_csv(outdir / "annual_walk_forward_metrics.csv", index=False)
    prediction_df.to_csv(outdir / "backtest_predictions.csv", index=False, date_format="%Y-%m-%d")

    warnings = [
        "Outputs estimate future harmonized M7 C-anomaly evidence, not confirmed fire occurrence, burned area, severity, damage, or emergency risk.",
        "Model scores are not calibrated probabilities.",
        "If upstream M7 anomaly reference statistics were fitted using future dates, a retrospective score may not be a fully causal live forecast; verify upstream reference-time causality before deployment.",
        "The persistence baseline is the number of C-threshold-active days in the prior seven calendar days; it is a simple comparator, not a physical fire-spread model.",
        "The 30-day target is defined as at least the configured number of future C-threshold-active days. This is an explicit operational definition of elevated activity and should be reviewed before external use.",
    ]
    if not complete_years:
        warnings.append("No complete calendar years were found; annual walk-forward folds may be empty.")
    if len(years_present) < 3:
        warnings.append("Fewer than three distinct calendar years are present; chronology-based method selection may be unsupported.")
    if not all_prediction_frames:
        warnings.append("No eligible annual walk-forward folds completed. Inspect data coverage and target class counts before interpreting outputs.")
    latest_sector_count = int(df.loc[df["date"] == max_date, "sector"].nunique())
    omitted_sector_count = int(df["sector"].nunique()) - latest_sector_count
    if omitted_sector_count > 0:
        warnings.append(
            f"Latest watchlists include only the {latest_sector_count} sectors with a row on the global data cutoff date; "
            f"{omitted_sector_count} sectors were omitted because no row exists on that date."
        )

    report = {
        "package": "Fire Harmonizer Additive ML Upgrade",
        "package_version": __version__,
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "model_family": "HistGradientBoostingClassifier",
        "input_file": str(source),
        "input_sha256": input_hash,
        "input_rows_after_cleaning": int(len(df)),
        "invalid_date_or_sector_rows_dropped": invalid_rows_dropped,
        "sector_count": int(df["sector"].nunique()),
        "sectors_present_on_latest_date": latest_sector_count,
        "data_start_date": min_date.date().isoformat(),
        "data_end_date": max_date.date().isoformat(),
        "calendar_years_present": years_present,
        "complete_calendar_years": complete_years,
        "horizons_days": list(horizons),
        "c_threshold": float(c_threshold),
        "elevated_min_active_days_30d": int(elevated_min_active_days),
        "feature_schema_file": "feature_schema.json",
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "metrics_by_horizon": horizon_reports,
        "annual_walk_forward_metrics_file": "annual_walk_forward_metrics.csv",
        "backtest_predictions_file": "backtest_predictions.csv",
        "warnings": warnings,
        "interpretation": "Experimental ranking of future harmonized anomaly evidence; not a verified fire prediction or operational alert.",
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
    }
    _write_json(outdir / "model_metrics.json", report)
    status("complete", f"Completed all requested horizons. New outputs written to {outdir}")
    return report
