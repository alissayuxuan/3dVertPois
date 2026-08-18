"""
Are the outermost vertebrae of a scan worse than the ones in the middle?

They are treated differently in two places, and the two effects pull in opposite directions:

  * `report_poi.py` exempts the first and last vertebra from **all** spatial ordering
    constraints (`is_first_or_last`), so their reported error count is not comparable to a
    middle vertebra's out of the box;
  * `combine_poi_cases-order*.py` sets `vert_orientations = None` for them, so they also
    receive none of the orientation-based corrections.

This script reports both halves:

  1. rates for the checks that apply everywhere (subregion, surface distance), which *are*
     comparable across positions;
  2. rates for the spatial constraints, which requires a report generated with
     `--check_spatial_on_first_last` and tells you what the exemption is hiding.

Usage:
    python3 analyze_scan_position.py DERIVATIVE [--out_root=PATH] [--prefix=TEST_] [DATASET ...]
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from TPTBox import POI, v_idx_order

DATA_ROOT = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge")
ANALYSIS_ROOT = Path("/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/data_analysis")
PREFIX = "TEST_"
ALL_DATASETS = [
    "dataset-verse19training_1mmiso",
    "dataset-verse20training_1mmiso",
    "dataset-verse19validation_1mmiso",
    "dataset-verse20validation_1mmiso",
    "dataset-verse19test_1mmiso",
    "dataset-verse20test_1mmiso",
]
VERTEBRA_FROM = 6


def _entities(name: str, drop_prefixes=("mod-", "seg-")) -> set[str]:
    """BIDS key-value tokens of a filename stem, ignoring the suffix and the given keys."""
    parts = [p for p in name.split("_") if "-" in p]
    return {p for p in parts if not p.startswith(drop_prefixes)}


def poi_path_for(der: str, ds: str, subject_ct_id: str) -> Path | None:
    """Report file `<X>_ct_poi_report.xlsx` -> the POI file it was generated from.

    Matched on the set of BIDS entities rather than by prefix: the entities are reordered
    between the two names (`sub-verse648_dir-iso_ct` vs `sub-verse648_mod-ct_dir-iso_...`),
    so a `startswith` test drops every subject carrying a `dir-` or `acq-` key, and comparing
    entity sets also keeps the `split-` subjects apart from one another.
    """
    stem = subject_ct_id[:-3] if subject_ct_id.endswith("_ct") else subject_ct_id
    wanted = _entities(stem)
    subject_dir = DATA_ROOT / ds / der / stem.split("_")[0]
    if not subject_dir.is_dir():
        return None
    for p in sorted(subject_dir.rglob("*_poi.json")):
        if p.name.endswith("mrk.json"):
            continue
        if _entities(p.name.rsplit(".", 1)[0]) == wanted:
            return p
    return None


def vertebrae_in_order(der: str, ds: str, subject_ct_id: str) -> list[int] | None:
    p = poi_path_for(der, ds, subject_ct_id)
    if p is None:
        return None
    verts = sorted(POI.load(p).keys_region(), key=lambda v: v_idx_order.index(v))
    verts = [v for v in verts if v >= VERTEBRA_FROM - 1]
    return verts or None


def collect(der: str, datasets: list[str], analysis_root: Path, prefix: str) -> tuple[pd.DataFrame, int]:
    rows, unmatched = [], 0
    for ds in datasets:
        d = analysis_root / f"{prefix}{der}" / ds
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*_poi_report.xlsx")):
            if "neighbor_angle" in p.name:
                continue
            subject = p.name.replace("_poi_report.xlsx", "")
            verts = vertebrae_in_order(der, ds, subject)
            if verts is None:
                unmatched += 1
                continue
            df = pd.read_excel(p)
            if df.empty:
                df = pd.DataFrame(columns=["vertebra", "location", "severity", "description"])
            kind = {
                "subregion": df[df.description.str.contains("subregion", na=False)],
                "surface": df[df.description.str.contains("surface distance", na=False)],
                "spatial": df[df.description.str.contains("Spatial", na=False)],
            }
            for i, v in enumerate(verts):
                row = {
                    "ds": ds,
                    "subject": subject,
                    "vert": v,
                    "position": "first" if i == 0 else ("last" if i == len(verts) - 1 else "middle"),
                    "n_vert_in_scan": len(verts),
                }
                for name, sub in kind.items():
                    hit = sub[sub.vertebra == v] if len(sub) else sub
                    row[f"{name}_pois"] = len(set(hit.location)) if len(hit) else 0
                    row[f"{name}_rows"] = len(hit)
                rows.append(row)
    return pd.DataFrame(rows), unmatched


def label_matched(R: pd.DataFrame, kind: str = "subregion", min_per_cell: int = 15) -> pd.DataFrame:
    """Compare the *same* vertebra label in an outermost vs an interior slot.

    "first" is usually C6/T1 and "last" is L5 in most scans, so a plain position comparison
    confounds scan position with anatomy - those vertebrae might simply be harder. Restricting
    to labels that appear in both roles removes that: any remaining gap is attributable to the
    vertebra sitting at the edge of the scan.
    """
    R = R.copy()
    R["outer"] = R.position != "middle"
    rows = []
    for label, grp in R.groupby("vert"):
        inner, outer = grp[~grp.outer], grp[grp.outer]
        if len(inner) < min_per_cell or len(outer) < min_per_cell:
            continue
        rows.append(
            {
                "vert": label,
                "n_interior": len(inner),
                "n_outermost": len(outer),
                f"interior_pct": round((inner[f"{kind}_pois"] > 0).mean() * 100, 1),
                f"outermost_pct": round((outer[f"{kind}_pois"] > 0).mean() * 100, 1),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["ratio"] = (out.outermost_pct / out.interior_pct.replace(0, np.nan)).round(2)
    return out.sort_values("n_outermost", ascending=False)


def summarise(R: pd.DataFrame, kinds: list[str]) -> pd.DataFrame:
    g = R.groupby("position")
    out = {"n_vertebrae": g.size(), "n_subjects": g.subject.nunique()}
    for k in kinds:
        out[f"{k}_per_vert"] = g[f"{k}_pois"].mean().round(3)
        out[f"pct_verts_{k}"] = (g[f"{k}_pois"].apply(lambda s: (s > 0).mean()) * 100).round(1)
    return pd.DataFrame(out).reindex(["first", "middle", "last"])


if __name__ == "__main__":
    args = sys.argv[1:]
    analysis_root, prefix = ANALYSIS_ROOT, PREFIX
    for a in list(args):
        if a.startswith("--out_root="):
            analysis_root = Path(a.split("=", 1)[1])
            args.remove(a)
        elif a.startswith("--prefix="):
            prefix = a.split("=", 1)[1]
            args.remove(a)
    if not args:
        print(__doc__)
        sys.exit(2)
    der, datasets = args[0], (args[1:] or ALL_DATASETS)

    R, unmatched = collect(der, datasets, analysis_root, prefix)
    if R.empty:
        print("no data")
        sys.exit(1)
    print(f"derivative: {der}")
    print(f"subjects: {R.subject.nunique()}   vertebrae: {len(R)}   unmatched report files: {unmatched}\n")
    has_spatial = R.spatial_pois.sum() > 0
    print(summarise(R, ["subregion", "surface"] + (["spatial"] if has_spatial else [])).to_string())
    if has_spatial:
        print("\nspatial-flagged POIs by position:", R.groupby("position").spatial_pois.sum().to_dict())
    print("\nvertebra labels most often in the outermost slot:")
    for pos in ("first", "last"):
        print(f"   {pos}: {Counter(R[R.position == pos].vert).most_common(6)}")

    lm = label_matched(R)
    print("\nsame vertebra label, interior vs outermost slot (% of vertebrae with a subregion error):")
    print(lm.to_string(index=False) if not lm.empty else "   (no label appears often enough in both roles)")
    if not lm.empty:
        w = lm.dropna(subset=["ratio"])
        if len(w):
            print(f"   -> median ratio outermost/interior = {w.ratio.median():.2f} over {len(w)} labels")
