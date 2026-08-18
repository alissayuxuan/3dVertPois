"""
Shared CLI config for the POI report / correction pipeline.

Imported by report_poi.py, report_neighbor_angle.py and auto_correct_angle.py so
the whole pipeline shares a single set of CLI args (root, datasets, num_threads, ...).
Not every field is used by every script; unused fields are simply ignored.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

from TPTBox import Location
from TypeSaveArgParse import Class_to_ArgParse


@dataclass
class PipelineConfig(Class_to_ArgParse):
    # ---- shared I/O ----
    root: Path = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/")
    rawdata: str = "rawdata"
    der_subreg: str = "derivatives_combined"  # "derivatives_subreg"
    der_vert: str = "derivatives_combined"
    der_direction: str = "derivatives_poi_deterministic"
    out_root: Path = Path("/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/data_analysis")
    vertebra_from: int = 6
    overwrite: bool = False  # regenerate reports / corrected POIs even if the output already exists

    # ---- report scripts (report_poi.py / report_neighbor_angle.py) ----
    der_poi_mainpred: list[str] = field(
        default_factory=lambda: [
            # "derivatives_poi_deterministic",
            # "derivatives_poi_automatic_correction-v4-onlygood",
            # "derivatives_poi_automatic_correction-v5-onlygood",
            "derivatives_poi_automatic_correction-v5-onlygood-anglecorr",
        ]
    )
    der_out_prefix: str = "TEST_"
    ignore_poi: list[Location] = field(
        default_factory=lambda: [
            Location.Vertebra_Direction_Inferior,
            Location.Vertebra_Direction_Posterior,
            Location.Vertebra_Direction_Right,
            Location.Vertebra_Corpus,
        ]
    )
    # Which vertebrae are exempt from the spatial ordering constraints:
    #   "truncated"  - only those whose mask runs into the edge of the scanned volume, i.e. the
    #                  ones whose POIs describe anatomy that was not imaged (default)
    #   "first_last" - the outermost vertebra of the scan regardless of truncation (the old rule)
    #   "none"       - no exemption
    spatial_exemption: str = "truncated"
    truncation_margin: int = 0  # count a vertebra as truncated if it comes within this many voxels of a face
    cprofile_this: bool = False

    # ---- correction script (auto_correct_angle.py) ----
    der_poi_src: str = "derivatives_poi_automatic_correction-v5-onlygood"
    der_out_suffix: str = "-anglecorr"
    analysis_prefix: str = "TEST_"
    min_spline_pts: int = 4  # minimum points in LOO set to attempt spline; else use linear interp
    constrain_to_vertebra: bool = True  # snap each correction onto its own vertebra surface; reject off-vertebra / oversized moves
    max_shift_mm: float = 30.0  # reject a correction that would move the point farther than this (sanity cap vs. spline blow-ups)
    use_report_guard: bool = True  # keep a correction only if it strictly lowers the reported severity and blames no new POI
    requires_filling: bool = False  # surface projection convention; False matches combine_poi_cases-order.py
    do_reposition_midpoints: bool = False  # 124/108/116 are already placed by combine_poi_cases-order.py
    do_surface_project: bool = False  # unused: corrected points are projected individually, never the whole subject
    save_unchanged: bool = True  # also copy subjects with zero detected errors to der_out

    # ---- report-driven corrector (auto_correct_report.py) ----
    do_subregion_snap: bool = True  # snap POIs that touch a disallowed subregion onto an allowed one
    snap_max_shift_voxel: float = 8.0  # give up rather than move a POI further than this
    snap_max_candidates: int = 64  # allowed-subregion voxels tested, nearest first
    do_axis_ordering: bool = True  # move POIs that violate an ordering constraint just far enough to satisfy it
    order_axes: list[str] = field(default_factory=lambda: ["A"])  # which constraint axes to repair
    order_margin_voxel: float = 1.0  # overshoot the deficit by this much so the constraint is met, not just tied
    order_max_shift_voxel: float = 6.0  # a larger deficit means the POI is wrong, not merely misordered
    correct_iterations: int = 2  # re-report and retry per vertebra, since one fix can expose the next

    # ---- shared run control ----
    datasets: list[str] = field(
        default_factory=lambda: [
            "dataset-verse19training_1mmiso",
            "dataset-verse20training_1mmiso",
            "dataset-verse19validation_1mmiso",
            "dataset-verse20validation_1mmiso",
            "dataset-verse19test_1mmiso",
            "dataset-verse20test_1mmiso",
        ]
    )
    num_threads: int = 10
