from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data import complete_calendar_years, load_science
from .pipeline import HORIZONS, run_pipeline


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _write_status(path: Path, **updates: Any) -> None:
    current = _read_json(path)
    current.update(updates)
    current["updated_at_utc"] = _timestamp()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(current, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    except AttributeError:
        return False
    return True


class RunLock:
    """Simple cross-process lock preventing concurrent full training runs per output dir."""
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.write(self.fd, json.dumps({"pid": os.getpid(), "started_at_utc": _timestamp()}).encode("utf-8"))
                os.close(self.fd)
                self.fd = None
                return self
            except FileExistsError:
                lock = _read_json(self.path)
                try:
                    pid = int(lock.get("pid", -1))
                except (TypeError, ValueError):
                    pid = -1
                if pid > 0 and _pid_is_alive(pid):
                    raise RuntimeError(f"A model run is already active (PID {pid}; lock {self.path}).")
                # Recover only a stale lock whose owner no longer exists.
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
        raise RuntimeError(f"Could not acquire training lock {self.path}.")

    def __exit__(self, exc_type, exc, tb):
        try:
            self.path.unlink(missing_ok=True)
        except Exception:
            pass


def command_validate(args: argparse.Namespace) -> int:
    frame = load_science(args.science)
    years = complete_calendar_years(frame)
    print("Fire Harmonizer data validation passed")
    print(f"Rows: {len(frame):,}")
    print(f"Sectors: {frame['sector'].nunique():,}")
    print(f"Date range: {frame['date'].min().date()} to {frame['date'].max().date()}")
    print(f"Complete calendar years: {years}")
    print(f"Missing m7_C values: {int(frame['m7_C'].isna().sum()):,}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    status_path = outdir / "job_status.json"
    lock_path = outdir / ".training.lock"
    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    try:
        with RunLock(lock_path):
            job = {
                "job_id": job_id,
                "status": "running",
                "pid": os.getpid(),
                "started_at_utc": _timestamp(),
                "stage": "starting",
                "message": "Starting Fire Harmonizer multi-horizon ML run.",
                "science_file": str(args.science.resolve()),
                "output_dir": str(outdir),
                "command": "run",
            }
            status_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
            def on_status(stage: str, message: str) -> None:
                _write_status(status_path, status="running", pid=os.getpid(), stage=stage, message=message)
            report = run_pipeline(
                args.science,
                outdir,
                c_threshold=args.c_threshold,
                elevated_min_active_days=args.elevated_min_active_days,
                horizons=tuple(args.horizons),
                save_feature_table=args.save_feature_table,
                status_callback=on_status,
            )
            _write_status(
                status_path,
                status="succeeded",
                job_id=job_id,
                pid=os.getpid(),
                stage="complete",
                message="All requested horizons completed.",
                finished_at_utc=_timestamp(),
                report_file=str(outdir / "model_metrics.json"),
                input_sha256=report.get("input_sha256"),
            )
        return 0
    except Exception as exc:
        _write_status(
            status_path,
            status="failed",
            job_id=job_id,
            pid=os.getpid(),
            stage="failed",
            message=str(exc),
            error_type=type(exc).__name__,
            finished_at_utc=_timestamp(),
        )
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise


def command_status(args: argparse.Namespace) -> int:
    path = args.output_dir.resolve() / "job_status.json"
    if not path.exists():
        print(f"No job status file found at {path}")
        return 1
    print(path.read_text(encoding="utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fh-ml-addon",
        description="Additive Fire Harmonizer 7/14/30-day anomaly-evidence forecasting tools.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate the existing harmonized science CSV without training.")
    validate.add_argument("--science", type=Path, default=Path("output/harmonized_science_all_years.csv"))
    validate.set_defaults(func=command_validate)

    run = sub.add_parser("run", help="Run temporal backtests, train models and create horizon watchlists.")
    run.add_argument("--science", type=Path, default=Path("output/harmonized_science_all_years.csv"))
    run.add_argument("--output-dir", type=Path, default=Path("output/ml_addon"))
    run.add_argument("--c-threshold", type=float, default=2.0)
    run.add_argument("--elevated-min-active-days", type=int, default=3,
                     help="30-day target is positive when at least this many future days have m7_C >= threshold.")
    run.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=list(HORIZONS))
    run.add_argument("--save-feature-table", action="store_true", help="Export the full generated feature table; can be large.")
    run.set_defaults(func=command_run)

    status = sub.add_parser("status", help="Print the most recent job status JSON.")
    status.add_argument("--output-dir", type=Path, default=Path("output/ml_addon"))
    status.set_defaults(func=command_status)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
