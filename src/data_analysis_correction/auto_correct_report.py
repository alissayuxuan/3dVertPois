"""
Report-driven POI correction.

Where `auto_correct_angle.py` repairs corpus-lane bends, this repairs the two error classes
that dominate `report_poi.py`'s output:

  * "Projected POI touches invalid subregion" - the POI sits on the wrong anatomical
    subregion (mostly the articulate processes 86/87/88/89). Fixed by snapping it to the
    nearest voxel of an allowed subregion *within its own vertebra*.
  * "Spatial logic constraint violated" on the anterior axis - a corpus point has drifted
    behind a point it must stay in front of. Fixed by moving it forward along the
    vertebra's own anterior axis by exactly the deficit plus a small margin.

Every proposed change goes through `report_utils.guarded_apply`, so it is kept only when it
strictly lowers the reported severity of that vertebra and blames no POI that was not
already blamed. A corrector can therefore not trade one error for another.
"""

import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from TPTBox import BIDS_FILE, NII, POI, No_Logger, Log_Type, v_idx_order
from utils.filepaths import search_path
from utils.misc import surface_project_poi
from utils.vertebra_rotation import get_axis_direction_vector

from report_config import PipelineConfig as CorrectionConfig
from report_utils import SUBREGION_CONSTRAINT_DICT, LogicReport, guarded_apply, poi_touches_subreg
from report_poi import make_vertebra_poi_report
from auto_correct_angle import _DIRECTION_LOCS, _load_direction_poi, compute_rel_to_corpus, make_score_fn

logger = No_Logger(prefix="auto_correct_report")

_SUBREGION_DESC = "invalid subregion"
_SPATIAL_DESC = "Spatial logic constraint violated"


def propose_subregion_snap(
    poi: POI,
    vert: int,
    loc: int,
    v_msk: NII,
    combined_subreg: NII,
    max_shift_voxel: float,
    max_candidates: int,
) -> np.ndarray | None:
    """Nearest position whose *projection* lands in an allowed subregion, else None.

    The report evaluates the surface-projected POI, not the stored one, so a candidate is
    only useful if it still satisfies the constraint after projection. Candidates are
    therefore restricted to voxels on the vertebra's own surface (where projection is
    essentially a no-op) and each is projected and re-tested with the same majority vote the
    report uses (`report_utils.poi_touches_subreg`) before being returned.
    """
    allowed = SUBREGION_CONSTRAINT_DICT.get(int(loc))
    if not allowed:
        return None

    c = np.asarray(poi[vert, loc], dtype=float)
    arr = combined_subreg.get_array()
    surface = v_msk.compute_surface_mask(connectivity=3, dilated_surface=False).get_array() != 0

    lo = np.maximum(np.floor(c - max_shift_voxel).astype(int), 0)
    hi = np.minimum(np.ceil(c + max_shift_voxel).astype(int) + 1, np.asarray(arr.shape))
    if np.any(hi <= lo):
        return None
    sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))

    hits = np.argwhere(np.isin(arr[sl], list(allowed)) & surface[sl])
    if len(hits) == 0:
        return None
    coords = hits + lo
    dists = np.linalg.norm(coords - c, axis=1)

    for i in np.argsort(dists)[:max_candidates]:
        if dists[i] > max_shift_voxel:
            break
        cand = coords[i].astype(float)
        single = poi.make_empty_POI()
        single[vert, loc] = tuple(cand.tolist())
        projected = surface_project_poi(single, v_msk, requires_filling=False)
        if (vert, loc) not in projected:
            continue
        if poi_touches_subreg(tuple(projected[vert, loc]), combined_subreg) in allowed:
            return cand
    return None


def propose_axis_shifts(
    reports: list[LogicReport],
    poi: POI,
    v_msk: NII,
    rel_to_corpus: dict | None,
    axes: tuple[str, ...],
    margin: float,
    max_shift_voxel: float,
) -> dict[tuple[int, int], np.ndarray]:
    """Move each POI that violates an ordering constraint just far enough to satisfy it.

    The report is computed in the vertebra's rotated frame, so `coord_diff[axis_idx]` is the
    signed deficit along `axis`; its magnitude carries over unchanged to a shift along the
    same direction vector in the image frame. When one POI violates several constraints on
    the same axis, the largest deficit wins.

    The shifted point is projected back onto the vertebra surface: the ordering constraints
    are evaluated on the *projected* POI, so a point pushed out into the air would simply be
    pulled back to where it started and would additionally trip the surface-distance check.
    Projecting here moves the point *along* the surface instead, which is what the
    constraint is really asking for.
    """
    needed: dict[tuple[int, int, str], float] = {}
    for rep in reports:
        if _SPATIAL_DESC not in rep.description:
            continue
        axis_info = rep.relevant_data.get("axis")
        coord_diff = rep.relevant_data.get("coord_diff")
        if axis_info is None or coord_diff is None:
            continue
        axis, axis_idx = axis_info[0], int(axis_info[1])
        if axis not in axes:
            continue
        deficit = abs(float(coord_diff[axis_idx])) + margin
        key = (int(rep.vertebra), int(rep.location.value), axis)
        needed[key] = max(needed.get(key, 0.0), deficit)

    changes: dict[tuple[int, int], np.ndarray] = {}
    for (vert, loc, axis), shift in needed.items():
        if shift > max_shift_voxel or (vert, loc) not in poi:
            continue
        direction = np.asarray(get_axis_direction_vector(rel_to_corpus, poi, axis), dtype=float)
        shifted = np.asarray(poi[vert, loc], dtype=float) + direction * shift
        single = poi.make_empty_POI()
        single[vert, loc] = tuple(shifted.tolist())
        projected = surface_project_poi(single, v_msk, requires_filling=False)
        if (vert, loc) not in projected:
            continue
        changes[(vert, loc)] = np.asarray(projected[vert, loc], dtype=float)
    return changes


def _apply_changes(poi: POI, changes: dict[tuple[int, int], np.ndarray], score_fn) -> int:
    """Apply as many of `changes` as the guard accepts. Returns how many were kept.

    Tries the whole batch first - scoring a vertebra means surface-projecting it, so one
    call for N changes is far cheaper than N. Only if the batch is rejected does it fall
    back to offering the changes one at a time, so a single bad proposal cannot block the
    good ones.
    """
    if not changes:
        return 0
    if len(changes) == 1:
        return 1 if guarded_apply(poi, changes, score_fn) else 0
    if guarded_apply(poi, changes, score_fn):
        return len(changes)
    return sum(1 for key, cand in changes.items() if guarded_apply(poi, {key: cand}, score_fn))


def correct_subject(
    poi: POI,
    subject_id: str,
    vert_nii: NII,
    subreg_nii: NII,
    vert_in_order: list[int],
    frames: dict[int, dict | None],
    score_fn,
    ignore_poi: list,
    opt,
) -> tuple[int, int]:
    """Run both correctors vertebra by vertebra. Returns (#subregion snaps, #axis shifts)."""
    first_last = {vert_in_order[0], vert_in_order[-1]} if vert_in_order else set()
    has_directions = all((v, loc) in poi for v in vert_in_order for loc in _DIRECTION_LOCS)
    n_snap = n_shift = 0

    for vert in vert_in_order:
        for _ in range(opt.correct_iterations):
            reports = make_vertebra_poi_report(
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
            if not reports:
                break

            v_msk = vert_nii.extract_label(vert)

            snaps: dict[tuple[int, int], np.ndarray] = {}
            if opt.do_subregion_snap and any(_SUBREGION_DESC in r.description for r in reports):
                combined_subreg = subreg_nii * v_msk
                for rep in reports:
                    if _SUBREGION_DESC not in rep.description:
                        continue
                    loc = int(rep.location.value)
                    cand = propose_subregion_snap(
                        poi, vert, loc, v_msk, combined_subreg, opt.snap_max_shift_voxel, opt.snap_max_candidates
                    )
                    if cand is not None:
                        snaps[(vert, loc)] = cand
            kept_snap = _apply_changes(poi, snaps, score_fn)

            shifts: dict[tuple[int, int], np.ndarray] = {}
            if opt.do_axis_ordering:
                shifts = propose_axis_shifts(
                    reports,
                    poi,
                    v_msk,
                    frames.get(vert),
                    tuple(opt.order_axes),
                    opt.order_margin_voxel,
                    opt.order_max_shift_voxel,
                )
                # a POI just snapped onto its subregion must not be dragged off it again
                shifts = {k: v for k, v in shifts.items() if k not in snaps}
            kept_shift = _apply_changes(poi, shifts, score_fn)

            n_snap += kept_snap
            n_shift += kept_shift
            if kept_snap + kept_shift == 0:
                break

    return n_snap, n_shift


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

        poi_out_path = img_bidsf.get_changed_path(
            file_type="json", bids_format="poi", info={"seg": "vert", "mod": "ct"}, parent=der_out
        )
        poi_out_path_global = img_bidsf.get_changed_path(
            file_type="mrk.json", bids_format="poi", info={"seg": "vert", "mod": "ct"}, parent=der_out
        )
        if poi_out_path.exists() and not opt.overwrite:
            continue

        poi_src_path = img_bidsf.get_changed_path(
            file_type="json", bids_format="poi", info={"seg": "vert", "mod": "ct"}, parent=opt.der_poi_src
        )
        if not poi_src_path.exists():
            logger.print(f"{subject_ct_id}: source POI not found at {poi_src_path}", Log_Type.FAIL)
            continue
        poi_ref = POI.load(poi_src_path).reorient_()

        vert_path = img_bidsf.get_changed_path(file_type="nii.gz", bids_format="msk", info={"seg": "vert"}, parent=opt.der_vert)
        subreg_path = img_bidsf.get_changed_path(
            file_type="nii.gz", bids_format="msk", info={"seg": "subreg"}, parent=opt.der_subreg
        )
        if not vert_path.exists() or not subreg_path.exists():
            logger.print(f"{subject_ct_id}: vert or subreg mask missing", Log_Type.FAIL)
            continue
        vert_nii = NII.load(vert_path, seg=True).reorient_()
        subreg_nii = NII.load(subreg_path, seg=True).reorient_()
        subreg_nii.map_labels_({51: 49, 50: 49}, verbose=False)

        vert_in_order = sorted(poi_ref.keys_region(), key=lambda v: v_idx_order.index(v))
        vert_in_order = [v for v in vert_in_order if v >= opt.vertebra_from - 1]
        if not vert_in_order:
            continue

        # Merge the direction POIs so the scorer works in the same rotated frame as
        # report_poi.py; they are stripped again before saving.
        poi_directions = _load_direction_poi(ds_dir, subject_id, subject_ct_id, img_bidsf, opt)
        frames = compute_rel_to_corpus(poi_directions, vert_in_order)
        added_direction_keys: list[tuple[int, int]] = []
        if poi_directions is not None:
            for v in vert_in_order:
                for loc in _DIRECTION_LOCS:
                    if (v, loc) in poi_directions and (v, loc) not in poi_ref:
                        poi_ref[v, loc] = poi_directions[v, loc]
                        added_direction_keys.append((v, loc.value))

        score_fn = make_score_fn(poi_ref, subject_ct_id, vert_nii, subreg_nii, vert_in_order, list(opt.ignore_poi))

        n_snap, n_shift = correct_subject(
            poi_ref, subject_ct_id, vert_nii, subreg_nii, vert_in_order, frames, score_fn, list(opt.ignore_poi), opt
        )

        for key in added_direction_keys:
            poi_ref.remove_(key)

        poi_out_path.parent.mkdir(parents=True, exist_ok=True)
        poi_ref.save(poi_out_path, verbose=False)
        poi_ref.to_global().save_mrk(poi_out_path_global, split_by_region=True)
        if n_snap or n_shift:
            logger.print(
                f"{subject_ct_id}: {n_snap} subregion snap(s), {n_shift} axis shift(s) -> {poi_out_path}", Log_Type.SAVE
            )


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
            delayed(_proc_subject)(subject, ds_dir, opt) for subject in tqdm(subjects, desc=ds_dir.name)
        )
