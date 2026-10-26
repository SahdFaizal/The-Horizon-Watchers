"""Optional FastAPI wrapper serving the latest saved add-on watchlists.

This service does not train a model per request. Run the CLI pipeline separately;
then the API serves its last successfully written watchlist files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


class DispatchRequest(BaseModel):
    horizon_days: Literal[7, 14, 30] = 7
    sectors: list[str] | None = None
    limit: int = Field(default=100, ge=1, le=5000)


def create_app(output_dir: Path | str | None = None) -> FastAPI:
    outdir = Path(output_dir or Path.cwd() / "output" / "ml_addon").resolve()
    app = FastAPI(
        title="Fire Harmonizer ML Add-on API",
        version="0.1.0",
        description=(
            "Serves saved experimental forecasts of future harmonized M7 C-anomaly evidence. "
            "It is not a confirmed-fire prediction or emergency alert service."
        ),
    )

    def metrics() -> dict:
        path = outdir / "model_metrics.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Model metrics not generated. Run the add-on CLI first.")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not read model metrics: {exc}") from exc

    def watchlist(horizon_days: int, sectors: list[str] | None, limit: int) -> dict:
        filename = {7: "watchlist_7d.csv", 14: "watchlist_14d.csv", 30: "outlook_30d.csv"}[horizon_days]
        path = outdir / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Saved {horizon_days}-day output not found. Run the add-on CLI first.")
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not read {filename}: {exc}") from exc
        if sectors:
            frame = frame[frame["sector"].astype(str).isin({str(s) for s in sectors})]
        frame = frame.head(limit)
        records = json.loads(frame.where(pd.notna(frame), None).to_json(orient="records", date_format="iso"))
        first = records[0] if records else {}
        return {
            "model_version": metrics().get("run_id", metrics().get("package_version", "unknown")),
            "horizon_days": horizon_days,
            "data_cutoff_date": first.get("as_of_date"),
            "generated_at_utc": metrics().get("created_at_utc"),
            "interpretation": "Future harmonized anomaly-evidence ranking; score is not a calibrated probability or confirmed fire forecast.",
            "count": len(records),
            "watchlist": records,
        }

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "output_dir_exists": outdir.exists(), "metrics_exists": (outdir / "model_metrics.json").exists()}

    @app.get("/api/v1/models")
    def models() -> dict:
        report = metrics()
        return {
            "package_version": report.get("package_version"),
            "run_id": report.get("run_id"),
            "data_start_date": report.get("data_start_date"),
            "data_end_date": report.get("data_end_date"),
            "horizons": report.get("metrics_by_horizon", {}),
            "warnings": report.get("warnings", []),
        }

    @app.get("/api/v1/metrics")
    def get_metrics() -> dict:
        return metrics()

    @app.get("/api/v1/watchlist")
    def get_watchlist(
        days: int = Query(default=7),
        limit: int = Query(default=100, ge=1, le=5000),
    ) -> dict:
        if days not in (7, 14, 30):
            raise HTTPException(status_code=422, detail="days must be one of 7, 14 or 30")
        return watchlist(days, None, limit)

    @app.post("/api/v1/forecast/dispatch")
    def dispatch(request: DispatchRequest) -> dict:
        """Return a filtered slice of the most recently saved watchlist; never train here."""
        return watchlist(request.horizon_days, request.sectors, request.limit)

    return app


app = create_app()
