import ast
import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.interpolate import splprep, splev
from tqdm import tqdm
from report_config import PipelineConfig as CorrectionConfig

from TPTBox import BIDS_FILE, NII, POI, Location, No_Logger, Log_Type, v_idx_order
from utils.filepaths import search_path, search_path_single
from utils.misc import surface_project_poi
from utils.vertebra_rotation import calc_orientation_from_poi, get_axis_direction_vector
from report_poi import make_vertebra_poi_report
from report_neighbor_angle import make_poi_report_corpus_lanes
from report_utils import guarded_apply

logger = No_Logger(prefix="auto_correct_angle")

# The POIs that define each vertebra's own coordinate frame; needed to lock corrections to
# the vertebra's superior/inferior plane. Copied from the source used by report_poi.py.
_DIRECTION_LOCS = [
    Location.Vertebra_Direction_Inferior,
    Location.Vertebra_Direction_Posterior,
    Location.Vertebra_Direction_Right,
    Location.Vertebra_Corpus,
]

# Maps each anterior corpus location to its lane (0=left, 1=center, 2=right)
_LOC_TO_LANE: dict[int, int] = {
    117: 0, 119: 0,
    101: 1, 103: 1,
    109: 2, 111: 2,
}
_LANE_LOCS: dict[int, list[int]] = {
    0: [117, 119],
    1: [101, 103],
    2: [109, 111],
}


def _predict_coord(
    coords: np.ndarray,  # (N, 3) all lane points in order
    t: np.ndarray,  # (N,) strictly-increasing parameter
    pos: int,  # index of the point to predict
    exclude: set[int],  # indices to leave out of the reference fit (incl. pos and all known-bad)
    min_spline_pts: int,
) -> np.ndarray | None:
    """
    Predict the coordinate at index `pos` from a curve fit through every lane point
    NOT in `exclude`. Uses a cubic spline when enough reference points remain,
    otherwise linear interp/extrapolation from the nearest reference neighbors.
    """
    n = len(coords)
    ref_mask = np.array([i not in exclude for i in range(n)])
    ref_t = t[ref_mask]
    ref_c = coords[ref_mask]
    if len(ref_t) < 2:
        return None

    # Primary: spline through the reference points only (no other outliers included).
    # `t` is strictly increasing and ref_t is a subset, so it stays strictly increasing.
    if len(ref_t) >= min_spline_pts:
        span = max(ref_t[-1] - ref_t[0], 1e-6)
        u_ref = (ref_t - ref_t[0]) / span
        u_query = float(np.clip((t[pos] - ref_t[0]) / span, 0.0, 1.0))
        try:
            tck, _ = splprep(ref_c.T, u=u_ref, s=0, k=min(3, len(ref_t) - 1))
            return np.asarray(splev(u_query, tck)).flatten()
        except Exception:
            pass

    # Fallback: linear interp between nearest reference neighbors on each side.
    ref_idx = np.flatnonzero(ref_mask)
    left = ref_idx[ref_idx < pos]
    right = ref_idx[ref_idx > pos]
    if len(left) and len(right):
        a, b = left[-1], right[0]
        w = (t[pos] - t[a]) / max(t[b] - t[a], 1e-6)
        return coords[a] * (1 - w) + coords[b] * w
    if len(left) >= 2:  # extrapolate downward
        a, b = left[-1], left[-2]
        return coords[a] + (coords[a] - coords[b]) * ((t[pos] - t[a]) / max(t[a] - t[b], 1e-6))
    if len(right) >= 2:  # extrapolate upward
        a, b = right[0], right[1]
        return coords[a] + (coords[a] - coords[b]) * ((t[pos] - t[a]) / max(t[a] - t[b], 1e-6))
    return None


def _correct_lane(
    lane_pts: list[tuple[int, int, int, np.ndarray]],  # (spine_idx, vert_id, loc, coords)
    bad_pairs: set[tuple[int, int]],  # (vert_id, loc) reported as bad
    min_spline_pts: int = 4,
) -> dict[tuple[int, int], np.ndarray]:
    """
    For each reported bend, decide which point is actually the outlier, then predict a
    replacement for it from a curve fit through the *other* (non-bad) lane points.

    Key differences from the naive approach:
      * Parameter `t` is a strictly-increasing running index (NOT spine_idx, which is
        duplicated across the two locs of a vertebra and makes splprep fail).
      * The reference fit excludes ALL reported-bad points, so corrections are never
        fit through the remaining outliers.
      * The reported (vert, loc) only marks *where* a bend is; the true outlier among
        the bend's three points {j-1, j, j+1} is chosen by largest leave-one-out
        residual, instead of blindly moving the reported point. Candidates are limited
        to the *same vertebra* as the reported point - a bend reported on vertebra V is
        never "fixed" by moving a point of its neighbour.
    Returns {(vert_id, loc): corrected_coords}.
    """
    corrections: dict[tuple[int, int], np.ndarray] = {}
    n = len(lane_pts)
    if n < 2:
        return corrections

    coords = np.vstack([p for *_, p in lane_pts])
    t = np.arange(n, dtype=float)  # strictly increasing, one knot per point

    bad_positions = {i for i, (si, v, l, p) in enumerate(lane_pts) if (v, l) in bad_pairs}
    if not bad_positions:
        return corrections

    # Disambiguate: for each reported bend, pick the point with the largest residual
    # against a fit through everything except the candidate (and except known-bad).
    chosen_positions: set[int] = set()
    for pos in sorted(bad_positions):
        best_c, best_res = None, -1.0
        pos_vert = lane_pts[pos][1]
        for c in (pos - 1, pos, pos + 1):
            if not (0 <= c < n):
                continue
            if lane_pts[c][1] != pos_vert:
                continue  # never move a neighbouring vertebra's point
            pred = _predict_coord(coords, t, c, bad_positions | {c}, min_spline_pts)
            if pred is None:
                continue
            res = float(np.linalg.norm(coords[c] - pred))
            if res > best_res:
                best_res, best_c = res, c
        chosen_positions.add(best_c if best_c is not None else pos)

    # Predict a replacement for each chosen outlier, excluding all chosen outliers
    # from the reference so multiple corrections in one lane don't fit through each other.
    for pos in chosen_positions:
        corrected = _predict_coord(coords, t, pos, chosen_positions, min_spline_pts)
        if corrected is not None:
            si, v, l, p = lane_pts[pos]
            corrections[(v, l)] = corrected

    return corrections


def _lock_along_axis(cand: np.ndarray, old: np.ndarray, axis_vec: np.ndarray | None) -> np.ndarray:
    """Return `cand` with its component along `axis_vec` reset to that of `old`.

    A lane bend is a wobble in the anterior/lateral plane. The superior-inferior position
    of an anterior-longitudinal POI is pinned to the vertebral endplate, so a curve fit
    through the whole spine must not be allowed to slide the point up or down - doing so
    collapses the within-vertebra top/bottom separation and violates the `S` ordering
    constraints against 84/85/106/114/116/122/124.
    """
    if axis_vec is None:
        return cand
    return cand - axis_vec * float(np.dot(cand - old, axis_vec))


def make_score_fn(
    poi_reference: POI,
    subject_id: str,
    vert_nii: NII,
    subreg_nii: NII,
    vert_in_order: list[int],
    ignore_poi: list[Location],
):
    """Build the `score_fn` that `report_utils.guarded_apply` uses to accept/reject a change.

    Scores the normal POI report for the affected vertebrae only (all of its constraints -
    surface distance, subregion, spatial logic - are within a single vertebra) plus the
    corpus-lane report for the whole spine (bends and inter-lane distances are inherently
    multi-vertebra, and it needs no image data so it is cheap).
    """
    first_last = {vert_in_order[0], vert_in_order[-1]} if vert_in_order else set()
    has_directions = all((v, loc) in poi_reference for v in vert_in_order for loc in _DIRECTION_LOCS)

    def score_fn(poi: POI, affected_verts: set[int]):
        reports = []
        for vert in sorted(affected_verts):
            if vert not in vert_in_order:
                continue
            reports.extend(
                make_vertebra_poi_report(
                    poi,
                    None,
                    subject_id,
                    vert,
                    vert_nii,
                    subreg_nii,
                    is_first_or_last=vert in first_last,
                    do_rotate_around_corpus=has_directions,
                    ignore_poi=ignore_poi,
                )
            )
        reports.extend(
            make_poi_report_corpus_lanes(
                poi,
                None,
                subject_id,
                vert_nii,
                subreg_nii,
                vertebra_from=vert_in_order[0] + 1,
                verbose=False,
                show_progress=False,
            )
        )
        return reports

    return score_fn


def auto_correct_angle_errors(
    poi_ref: POI,
    bad_entries: list[dict],  # each dict has keys: vertebra (int), location (int)
    vertebra_from: int = 6,
    min_spline_pts: int = 4,
    vert_nii: NII | None = None,
    max_shift_mm: float = 30.0,
    s_axis_by_vert: dict[int, np.ndarray] | None = None,
    score_fn=None,
    requires_filling: bool = False,
) -> int:
    """
    In-place correction of bad (vert, loc) pairs in poi_ref.

    Each predicted replacement is validated before being applied:
      * the component along the vertebra's superior axis is reset to the original value,
        so the correction can only move the point anteriorly/laterally (see
        ``_lock_along_axis``);
      * if ``vert_nii`` is given, the candidate is surface-projected onto its OWN
        vertebra, which guarantees the corrected point lands on the correct
        vertebra's surface (and never on a neighbouring vertebra);
      * the correction is rejected if it would move the point more than
        ``max_shift_mm`` millimetres from its original position (a sanity cap
        against spline extrapolation blow-ups near the lane ends);
      * if ``score_fn`` is given, the corrections of a vertebra are applied through
        ``guarded_apply``, i.e. kept only if the reported severity strictly drops and no
        new (vertebra, location) is blamed.
    Returns the number of coordinates replaced.
    """
    vert_in_order = sorted(poi_ref.keys_region(), key=lambda v: v_idx_order.index(v))
    vert_in_order = [v for v in vert_in_order if v >= vertebra_from - 1]

    # Group by lane
    bad_by_lane: dict[int, set[tuple[int, int]]] = {0: set(), 1: set(), 2: set()}
    for entry in bad_entries:
        loc = int(entry["location"])
        vert = int(entry["vertebra"])
        lane_id = _LOC_TO_LANE.get(loc)
        if lane_id is None:
            continue
        bad_by_lane[lane_id].add((vert, loc))

    vert_labels: set[int] | None = set(int(v) for v in vert_nii.unique()) if vert_nii is not None else None
    zoom = np.asarray(poi_ref.zoom, dtype=float)  # mm per voxel along each axis
    s_axis_by_vert = s_axis_by_vert or {}

    # Collect the accepted candidates per vertebra first, so all corrections of one
    # vertebra are scored together instead of one at a time.
    candidates_by_vert: dict[int, dict[tuple[int, int], np.ndarray]] = {}

    for lane_id, bad_set in bad_by_lane.items():
        if not bad_set:
            continue

        # Collect all available points for this lane across all vertebrae
        lane_pts: list[tuple[int, int, int, np.ndarray]] = []
        for spine_idx, vert in enumerate(vert_in_order):
            for loc in _LANE_LOCS[lane_id]:
                if (vert, loc) in poi_ref:
                    lane_pts.append((spine_idx, vert, loc, np.asarray(poi_ref[vert, loc])))

        corrections = _correct_lane(lane_pts, bad_set, min_spline_pts)

        for (vert, loc), new_coords in corrections.items():
            old = np.asarray(poi_ref[vert, loc], dtype=float)
            cand = np.asarray(new_coords, dtype=float)

            # Keep the superior/inferior position of the point untouched.
            cand = _lock_along_axis(cand, old, s_axis_by_vert.get(vert))

            # Constrain to the point's own vertebra: snap the candidate onto that
            # vertebra's surface. This also rejects predictions that would land off
            # the vertebra entirely (projection has nothing to project onto).
            if vert_nii is not None and (vert_labels is None or vert in vert_labels):
                surf_v_nii = vert_nii.extract_label(vert)
                poi_single = poi_ref.make_empty_POI()
                poi_single[vert, loc] = tuple(cand.tolist())
                poi_proj = surface_project_poi(poi_single, surf_v_nii, requires_filling=requires_filling)
                if (vert, loc) not in poi_proj:
                    logger.print(f"  vert={vert} loc={loc}: surface projection failed, skipped", Log_Type.WARNING)
                    continue
                # projecting can push the point off the locked plane again - restore it
                cand = _lock_along_axis(np.asarray(poi_proj[vert, loc], dtype=float), old, s_axis_by_vert.get(vert))

            # Reject corrections that displace the point implausibly far (in mm)
            shift_mm = float(np.linalg.norm((cand - old) * zoom))
            if shift_mm > max_shift_mm:
                logger.print(
                    f"  vert={vert} loc={loc}: rejected, shift {shift_mm:.1f} mm > {max_shift_mm:.0f} mm",
                    Log_Type.WARNING,
                )
                continue

            candidates_by_vert.setdefault(vert, {})[(vert, loc)] = cand

    n_corrections = 0
    for vert, changes in sorted(candidates_by_vert.items()):
        olds = {k: np.asarray(poi_ref[k], dtype=float) for k in changes}
        if score_fn is not None:
            if not guarded_apply(poi_ref, changes, score_fn, verbose=False):
                logger.print(f"  vert={vert}: {len(changes)} correction(s) reverted by the report guard", Log_Type.WARNING)
                continue
        else:
            for key, cand in changes.items():
                poi_ref[key] = tuple(float(x) for x in cand)
        for (v, loc), cand in changes.items():
            shift_mm = float(np.linalg.norm((cand - olds[(v, loc)]) * zoom))
            logger.print(f"  vert={v} loc={loc}: {olds[(v, loc)].round(1)} -> {cand.round(1)} ({shift_mm:.1f} mm)", Log_Type.OK)
        n_corrections += len(changes)

    return n_corrections


# (top_loc, mid_loc, bottom_loc) per lane — mid is repositioned to the midpoint of top+bottom
_LANE_TRIPLETS: list[tuple[int, int, int]] = [
    (117, 124, 119),  # left
    (101, 108, 103),  # center
    (109, 116, 111),  # right
]


def reposition_lane_midpoints(
    poi_ref: POI,
    vert_nii: NII,
    s_axis_by_vert: dict[int, np.ndarray] | None = None,
    requires_filling: bool = False,
) -> int:
    """
    For each vertebra and each anterior corpus lane, move the middle point
    (124/108/116) halfway between the top (117/101/109) and bottom (119/103/111)
    points *along the superior axis only*, then surface-project it onto that vertebra.

    Only the superior component is taken from the midpoint: the middle point must stay on
    the anterior edge of the corpus, and the full 3-D midpoint of a curved edge lies behind
    it. This mirrors what `combine_poi_cases-order.py` already does for these locations.

    Works in-place on poi_ref. Returns the number of midpoints repositioned.
    """
    vert_labels = set(vert_nii.unique())
    s_axis_by_vert = s_axis_by_vert or {}
    n_repositioned = 0

    for vert in poi_ref.keys_region():
        if vert not in vert_labels:
            continue

        surf_v_nii = vert_nii.extract_label(vert)
        s_axis = s_axis_by_vert.get(vert)

        for top_loc, mid_loc, bot_loc in _LANE_TRIPLETS:
            if (vert, top_loc) not in poi_ref or (vert, bot_loc) not in poi_ref:
                continue
            if (vert, mid_loc) not in poi_ref:
                continue

            top_coords = np.asarray(poi_ref[vert, top_loc])
            bot_coords = np.asarray(poi_ref[vert, bot_loc])
            mid_current = np.asarray(poi_ref[vert, mid_loc])
            midpoint = (top_coords + bot_coords) / 2.0
            if s_axis is None:
                mid_coords = midpoint
            else:
                # keep the current anterior/lateral position, take only the superior level
                mid_coords = mid_current + s_axis * float(np.dot(midpoint - mid_current, s_axis))

            # Build a single-entry POI so surface_project_poi can crop tightly
            poi_single = poi_ref.make_empty_POI()
            poi_single[vert, mid_loc] = tuple(mid_coords.tolist())
            poi_proj = surface_project_poi(poi_single, surf_v_nii, requires_filling=requires_filling)

            if (vert, mid_loc) in poi_proj:
                poi_ref[vert, mid_loc] = poi_proj[vert, mid_loc]
                n_repositioned += 1

    return n_repositioned


def compute_rel_to_corpus(poi_directions: POI | None, vert_in_order: list[int]) -> dict[int, dict | None]:
    """Per-vertebra `rel_to_corpus` frame, or None where the direction POIs are missing."""
    out: dict[int, dict | None] = {}
    for vert in vert_in_order:
        rel_to_corpus = None
        if poi_directions is not None and all((vert, loc) in poi_directions for loc in _DIRECTION_LOCS):
            try:
                rel_to_corpus = calc_orientation_from_poi(poi_directions, vert)[2]
            except Exception:
                rel_to_corpus = None
        out[vert] = rel_to_corpus
    return out


def compute_s_axis_by_vert(poi_ref: POI, poi_directions: POI | None, vert_in_order: list[int]) -> dict[str, object]:
    """Per-vertebra unit vector pointing superior, in the vertebra's own frame.

    Falls back to the image's superior axis when the direction POIs are unavailable, which
    is what `utils.vertebra_rotation.get_axis_direction_vector` does for `rel_to_corpus=None`.
    Also reports whether every vertebra has real direction POIs, so the scorer knows whether
    it may rotate.
    """
    frames = compute_rel_to_corpus(poi_directions, vert_in_order)
    s_axis_by_vert = {
        vert: np.asarray(get_axis_direction_vector(rel, poi_ref, "S"), dtype=float) for vert, rel in frames.items()
    }
    complete = poi_directions is not None and all(rel is not None for rel in frames.values())
    return {"s_axis": s_axis_by_vert, "complete": complete, "frames": frames}


def _load_direction_poi(ds_dir: Path, subject_id: str, subject_ct_id: str, img_bidsf: BIDS_FILE, opt) -> POI | None:
    """Load the deterministic POI that carries the vertebra direction points.

    Same lookup as report_poi.py / report_neighbor_angle.py, so the guard scores the POIs
    in exactly the rotated frame the report uses.
    """
    split_info = img_bidsf.info["split"] if "split" in img_bidsf.info else None
    stem = subject_ct_id.split("_")[0]
    query = f"**/{stem}*_seg-vert*poi.json" if split_info is None else f"**/{stem}*split-{split_info}_seg-vert*poi.json"
    path = search_path_single(ds_dir / opt.der_direction / subject_id, query)
    if path is None or not path.exists():
        return None
    return POI.load(path).reorient_()


def _proc_subject(subject: Path, ds_dir: Path, opt) -> None:
    if not subject.is_dir():
        return

    der_out = opt.der_poi_src + opt.der_out_suffix
    subject_id = subject.name
    img_paths = search_path(ds_dir / opt.rawdata / subject_id, f"{subject_id}*_ct.nii.gz")
    if not img_paths:
        return

    for img_path in img_paths:
        subject_ct_id = img_path.name.split(".")[0]
        img_bidsf = BIDS_FILE(img_path, dataset=ds_dir)

        # Output path (skip if already done)
        poi_out_path = img_bidsf.get_changed_path(
            file_type="json", bids_format="poi", info={"seg": "vert", "mod": "ct"}, parent=der_out
        )
        poi_out_path_global = img_bidsf.get_changed_path(
            file_type="mrk.json",
            bids_format="poi",
            info={"seg": "vert", "mod": "ct"},
            parent=der_out,
        )
        if poi_out_path.exists() and not opt.overwrite:
            if not poi_out_path_global.exists():
                POI.load(poi_out_path).to_global().save_mrk(poi_out_path_global, split_by_region=True)
            continue

        # Source POI
        poi_src_path = img_bidsf.get_changed_path(
            file_type="json", bids_format="poi", info={"seg": "vert", "mod": "ct"}, parent=opt.der_poi_src
        )
        if not poi_src_path.exists():
            poi_src_path = img_bidsf.get_changed_path(
                file_type="json",
                bids_format="poi",
                info={"seg": "vert", "mod": None, "source": "deterministic"},
                parent=opt.der_poi_src,
            )
        if not poi_src_path.exists():
            logger.print(f"{subject_ct_id}: source POI not found at {poi_src_path}", Log_Type.FAIL)
            continue

        poi_ref = POI.load(poi_src_path).reorient_()

        # Per-subject angle report
        report_path = (
            opt.out_root
            / f"{opt.analysis_prefix}{opt.der_poi_src}"
            / ds_dir.name
            / f"{subject_ct_id}_poi_neighbor_angle_report.xlsx"
        )
        bad_entries: list[dict] = []
        if report_path.exists():
            report_df = pd.read_excel(report_path)
            for _, row in report_df.iterrows():
                loc_val = row["location"]
                vert_val = row["vertebra"]
                if _LOC_TO_LANE.get(int(loc_val)) is None:
                    continue  # not an anterior corpus lane point
                if "inter-lane distance" in str(row.get("description", "")):
                    # A spacing change between two lanes says nothing about where the bend
                    # in *this* lane is; re-fitting this lane's own spline is not the fix.
                    continue
                bad_entries.append({"vertebra": int(vert_val), "location": int(loc_val)})

        # Load vert mask up-front if any surface-dependent step needs it: midpoint
        # repositioning, constraining corrections to the vertebra, or surface projection.
        vert_nii: NII | None = None
        needs_vert = opt.do_reposition_midpoints or (
            bool(bad_entries) and (opt.constrain_to_vertebra or opt.do_surface_project or opt.use_report_guard)
        )
        if needs_vert:
            vert_path = img_bidsf.get_changed_path(
                file_type="nii.gz", bids_format="msk", info={"seg": "vert"}, parent=opt.der_vert
            )
            if vert_path.exists():
                vert_nii = NII.load(vert_path, seg=True).reorient_()
            else:
                logger.print(f"{subject_ct_id}: vert mask not found", Log_Type.WARNING)

        # Vertebra orientation: needed to lock corrections to the superior/inferior plane
        # and (for the guard) to reproduce the rotated frame report_poi.py scores in.
        poi_directions = _load_direction_poi(ds_dir, subject_id, subject_ct_id, img_bidsf, opt)
        vert_in_order = sorted(poi_ref.keys_region(), key=lambda v: v_idx_order.index(v))
        vert_in_order = [v for v in vert_in_order if v >= opt.vertebra_from - 1]
        if not vert_in_order:
            logger.print(f"{subject_ct_id}: no vertebrae at or below vertebra_from, skipping", Log_Type.WARNING)
            continue
        axes = compute_s_axis_by_vert(poi_ref, poi_directions, vert_in_order)
        s_axis_by_vert: dict[int, np.ndarray] = axes["s_axis"]  # type: ignore

        # Mirror report_poi.py: the direction POIs are merged in so the scorer can rotate
        # into the vertebra frame. They are stripped again before saving.
        added_direction_keys: list[tuple[int, int]] = []
        if poi_directions is not None:
            for v in vert_in_order:
                for loc in _DIRECTION_LOCS:
                    if (v, loc) in poi_directions and (v, loc) not in poi_ref:
                        poi_ref[v, loc] = poi_directions[v, loc]
                        added_direction_keys.append((v, loc.value))

        score_fn = None
        if opt.use_report_guard and vert_nii is not None:
            subreg_path = img_bidsf.get_changed_path(
                file_type="nii.gz", bids_format="msk", info={"seg": "subreg"}, parent=opt.der_subreg
            )
            if subreg_path.exists():
                subreg_nii = NII.load(subreg_path, seg=True).reorient_()
                subreg_nii.map_labels_({51: 49, 50: 49}, verbose=False)
                score_fn = make_score_fn(
                    poi_ref, subject_ct_id, vert_nii, subreg_nii, vert_in_order, list(opt.ignore_poi)
                )
            else:
                logger.print(f"{subject_ct_id}: subreg mask not found, guard disabled", Log_Type.WARNING)

        if not bad_entries:
            if not opt.save_unchanged:
                continue
            n_corrections = 0
        else:
            logger.print(f"{subject_ct_id}: correcting {len(bad_entries)} reported errors", Log_Type.STAGE)
            n_corrections = auto_correct_angle_errors(
                poi_ref,
                bad_entries,
                vertebra_from=opt.vertebra_from,
                min_spline_pts=opt.min_spline_pts,
                vert_nii=vert_nii if opt.constrain_to_vertebra else None,
                max_shift_mm=opt.max_shift_mm,
                s_axis_by_vert=s_axis_by_vert,
                score_fn=score_fn,
                requires_filling=opt.requires_filling,
            )

        # Reposition middle lane points (124/108/116) to the superior midpoint of top/bottom.
        # Off by default: combine_poi_cases-order.py already owns these locations.
        if opt.do_reposition_midpoints and vert_nii is not None:
            n_mid = reposition_lane_midpoints(
                poi_ref, vert_nii, s_axis_by_vert=s_axis_by_vert, requires_filling=opt.requires_filling
            )
            if n_mid > 0:
                logger.print(f"{subject_ct_id}: repositioned {n_mid} lane midpoints", Log_Type.OK)

        # Corrected points are already projected onto their own vertebra inside
        # auto_correct_angle_errors. Re-projecting the whole subject would move every other
        # POI too - a local repair must not rewrite points nobody complained about.

        for key in added_direction_keys:
            poi_ref.remove_(key)

        poi_out_path.parent.mkdir(parents=True, exist_ok=True)
        poi_ref.save(poi_out_path, verbose=False)
        poi_ref.to_global().save_mrk(poi_out_path_global, split_by_region=True)
        if n_corrections > 0:
            logger.print(f"{subject_ct_id}: saved ({n_corrections} corrections) -> {poi_out_path}", Log_Type.SAVE)


if __name__ == "__main__":
    opt = CorrectionConfig.get_opt()

    print(opt)

    for ds_dir in sorted(opt.root.iterdir()):
        if not ds_dir.is_dir() or not ds_dir.name.startswith("dataset-"):
            continue
        if ds_dir.name not in opt.datasets:
            continue

        raw_dir = ds_dir / opt.rawdata
        if not raw_dir.exists():
            continue

        logger.print(f"Dataset: {ds_dir.name}", Log_Type.STAGE)
        subjects = list(raw_dir.iterdir())

        Parallel(n_jobs=opt.num_threads)(
            delayed(_proc_subject)(subject, ds_dir, opt)
            for subject in tqdm(subjects, desc=ds_dir.name)
        )
