"""
Per-subject rollup of both aggregated reports -> aggregated_subject_error_rollup.xlsx.

Consumes `aggregated_poi_report.xlsx` + `aggregated_poi_neighbor_angle_report.xlsx` (so run
aggregate_reports.py and aggregate_angle_reports.py first) and emits one row per subject,
sorted by how much attention that subject needs:

    subject_name, num_vertebra_with_error, num_unique_pois_affected,
    num_total_error_rows, num_hi_severity_rows

No vertebra filter is applied and `num_hi_severity_rows` counts rows with severity >= 5.
Both were reverse-engineered from the existing rollups and reproduce them exactly
(231/231 subjects on derivatives_poi_SecondIterSweep2Winner_combined).

Usage:
    python3 aggregate_subject_rollup.py                 # every TEST_* directory
    python3 aggregate_subject_rollup.py DERIVATIVE ...  # only these (name without TEST_)
"""

import sys
from pathlib import Path

import pandas as pd

from TPTBox import Log_Type, No_Logger

ROOT = Path("/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/data_analysis/")
PREFIX = "TEST_"
NORMAL_NAME = "aggregated_poi_report.xlsx"
ANGLE_NAME = "aggregated_poi_neighbor_angle_report.xlsx"
OUT_NAME = "aggregated_subject_error_rollup.xlsx"
HI_SEVERITY = 5.0

logger = No_Logger(prefix="aggregate_subject_rollup")


def build_rollup(directory: Path) -> pd.DataFrame | None:
    frames = [pd.read_excel(directory / n) for n in (NORMAL_NAME, ANGLE_NAME) if (directory / n).exists()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)

    grouped = df.groupby("subject_name")
    rollup = pd.DataFrame(
        {
            "num_vertebra_with_error": grouped.vertebra.nunique(),
            "num_unique_pois_affected": grouped.apply(
                lambda d: len(set(zip(d.vertebra, d.location))), include_groups=False
            ),
            "num_total_error_rows": grouped.size(),
            "num_hi_severity_rows": grouped.apply(
                lambda d: int((d.severity >= HI_SEVERITY).sum()), include_groups=False
            ),
        }
    ).reset_index()

    # single descending key with a stable sort, so ties keep groupby's (alphabetical)
    # subject order - this is what the existing rollups use
    return rollup.sort_values(by="num_unique_pois_affected", ascending=False, kind="stable").reset_index(drop=True)


def directories(selected: list[str]) -> list[Path]:
    if selected:
        # a name may already include its prefix, or be a bare derivative name
        return [ROOT / n if (ROOT / n).is_dir() else ROOT / f"{PREFIX}{n}" for n in selected]
    # any directory holding an aggregated report, including nested ones (e.g. winner/<model>)
    return sorted({p.parent for p in ROOT.glob(f"{PREFIX}*/**/{NORMAL_NAME}")} | {p.parent for p in ROOT.glob(f"{PREFIX}*/{NORMAL_NAME}")})


if __name__ == "__main__":
    for directory in directories(sys.argv[1:]):
        if not directory.is_dir():
            logger.print(f"Not a directory, skipping: {directory}", Log_Type.FAIL)
            continue
        rollup = build_rollup(directory)
        if rollup is None:
            logger.print(f"No aggregated reports in {directory}, skipping", Log_Type.WARNING)
            continue
        out = directory / OUT_NAME
        rollup.to_excel(out, index=False)
        logger.print(f"{len(rollup)} subjects -> {out}", Log_Type.SAVE)
