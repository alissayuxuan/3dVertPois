"""
Convenience runner: generate the POI reports and aggregate them in one go.

Runs, in order:
  1. report_poi.py             -> per-subject normal POI reports
  2. report_neighbor_angle.py  -> per-subject corpus-lane angle reports
  3. aggregate_reports.py       -> aggregated_poi_report.xlsx
  4. aggregate_angle_reports.py -> aggregated_poi_neighbor_angle_report.xlsx

Any extra CLI args are forwarded to the two *report* scripts (which use
Class_to_ArgParse), e.g.:
    python run_all_reports.py --num_threads 4
The aggregation scripts take no args and are run as-is.

The aggregation scripts skip a directory whose aggregated_*.xlsx already exists,
so on a re-run pass --reaggregate to delete stale aggregated files first.
"""

import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DATA_ANALYSIS_ROOT = Path("/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/data_analysis")

# (script, forwards_extra_args)
REPORT_SCRIPTS = [
    ("report_poi.py", True),
    ("report_neighbor_angle.py", True),
]
AGGREGATE_SCRIPTS = [
    ("aggregate_reports.py", False),
    ("aggregate_angle_reports.py", False),
]
AGGREGATED_FILES = [
    "aggregated_poi_report.xlsx",
    "aggregated_poi_neighbor_angle_report.xlsx",
]


def _run(script: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(THIS_DIR / script), *extra_args]
    print(f"\n{'=' * 80}\n>> {' '.join(cmd)}\n{'=' * 80}", flush=True)
    subprocess.run(cmd, cwd=str(THIS_DIR), check=True)


def _clear_aggregated() -> None:
    if not DATA_ANALYSIS_ROOT.exists():
        return
    for sub in DATA_ANALYSIS_ROOT.iterdir():
        if not sub.is_dir():
            continue
        for fname in AGGREGATED_FILES:
            f = sub / fname
            if f.exists():
                print(f"   removing stale {f}", flush=True)
                f.unlink()


if __name__ == "__main__":
    extra_args = sys.argv[1:]

    reaggregate = "--reaggregate" in extra_args
    if reaggregate:
        extra_args = [a for a in extra_args if a != "--reaggregate"]

    for script, forwards in REPORT_SCRIPTS:
        _run(script, extra_args if forwards else [])

    if reaggregate:
        print(f"\n{'=' * 80}\n>> clearing existing aggregated reports (--reaggregate)\n{'=' * 80}", flush=True)
        _clear_aggregated()

    for script, forwards in AGGREGATE_SCRIPTS:
        _run(script, extra_args if forwards else [])

    print(f"\n{'=' * 80}\nDone. Aggregated reports under {DATA_ANALYSIS_ROOT}\n{'=' * 80}", flush=True)
