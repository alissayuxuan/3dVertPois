"""
Compare the reports of two derivatives and split the difference into fixed vs. introduced.

The headline number is the count of *unique* `(subject, vertebra, location)` flagged across
both report kinds. Raw row counts are misleading: one bad POI produces one row per violated
constraint, so a change that fixes a point can look like it fixed a dozen errors.

Splitting the delta into fixed and introduced is the part that matters. A correction step
can lower the total while still breaking points that were fine - which is exactly how the
old `auto_correct_angle.py` behaved (it fixed 11 and introduced 504) - so a step is only
worth keeping if it introduces (close to) nothing.

Usage:
    python3 compare_reports.py [--prefix=TEST_] BASE_DERIVATIVE NEW_DERIVATIVE [DATASET ...]

Both derivatives must already have per-subject reports under
`<out_root>/TEST_<derivative>/<dataset>/`; the script refuses to compare a half-finished
report run rather than silently reporting an over-optimistic number.
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path("/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/data_analysis")
PREFIX = "TEST_"
VERTEBRA_FROM = 8


def report_files(der: str, ds: str, kind: str) -> list[Path]:
    d = ROOT / f"{PREFIX}{der}" / ds
    if kind == "normal":
        return sorted(p for p in d.glob("*_poi_report.xlsx") if "neighbor_angle" not in p.name)
    return sorted(d.glob("*_poi_neighbor_angle_report.xlsx"))


def report_keys(der: str, ds: str, kind: str) -> set[tuple]:
    out: set[tuple] = set()
    for p in report_files(der, ds, kind):
        df = pd.read_excel(p)
        if df.empty:
            continue
        df = df[df.vertebra >= VERTEBRA_FROM]
        out |= set(zip(df.subject_name, df.vertebra, df.location))
    return out


def main(base: str, new: str, datasets: list[str]) -> int:
    for ds in datasets:
        for kind in ("normal", "angle"):
            n_base, n_new = len(report_files(base, ds, kind)), len(report_files(new, ds, kind))
            if n_base != n_new:
                print(f"INCOMPLETE: {ds}/{kind}: base has {n_base} report files, new has {n_new}", file=sys.stderr)
                return 1
            if n_base == 0:
                # 0 == 0 is not agreement, it means the dataset name matched nothing. In zsh an
                # unquoted "$DS" is passed as a single argument, which lands here silently.
                print(f"NO DATA: {ds}/{kind}: no report files found for either derivative", file=sys.stderr)
                return 1

    total_base: set[tuple] = set()
    total_new: set[tuple] = set()
    for kind in ("normal", "angle"):
        kb: set[tuple] = set()
        kn: set[tuple] = set()
        for ds in datasets:
            kb |= report_keys(base, ds, kind)
            kn |= report_keys(new, ds, kind)
        total_base |= kb
        total_new |= kn
        print(f"{kind:7s} base {len(kb):5d} -> new {len(kn):5d}   fixed {len(kb - kn):4d}  introduced {len(kn - kb):4d}")
        if kn - kb:
            print("         introduced by location:", Counter(loc for _, _, loc in (kn - kb)).most_common(8))

    print(
        f"{'TOTAL':7s} base {len(total_base):5d} -> new {len(total_new):5d}   "
        f"fixed {len(total_base - total_new):4d}  introduced {len(total_new - total_base):4d}"
    )
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    for a in [a for a in args if a.startswith("--prefix=")]:
        PREFIX = a.split("=", 1)[1]
        args.remove(a)
    if len(args) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(args[0], args[1], args[2:]))
