import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))
from pathlib import Path
from TPTBox import No_Logger, Log_Type, NII, BIDS_FILE, POI, Location, Vertebra_Instance, v_idx_order
from TPTBox.core.vert_constants import COORDINATE
from joblib import Parallel, delayed
from panoptica import Metric
from utils.filepaths import search_path_single, search_path
import numpy as np
from utils.misc import surface_project_poi
from utils.vertebra_rotation import rotate_3darray, calc_orientation_from_poi
from scipy.interpolate import splprep, splev
from report_utils import (
    LogicReport,
    SPATIAL_LOGIC_CONSTRAINTS_DICT,
    SUBREGION_CONSTRAINT_DICT,
    poi_touches_subreg,
    save_logic_report,
    SUBREGION_SOFTCONSTRAINT_DICT,
)
from report_config import PipelineConfig as InferenceConfig
from tqdm import tqdm

logger = No_Logger(prefix="verse_cseg")


def make_poi_report_corpus_lanes(
    poi_ref: POI,
    poi_ref_proj: POI | None,
    subject_id: str,
    vert_msk: NII,
    subreg_msk: NII,
    debug: bool = False,
    ignore_poi: list[Location] = [],
    vertebra_from: int = 6,
    verbose: bool = True,
    show_progress: bool = True,
    # surface_msk: NII,
) -> list[LogicReport]:
    report_lines: list[LogicReport] = []

    vert_in_order = sorted(poi_ref.keys_region(), key=lambda v: v_idx_order.index(v))
    vert_in_order = [v for v in vert_in_order if v >= vertebra_from - 1]

    corpus_points_anterior_locations = {
        0: [117, 119],  # left #124
        1: [101, 103],  # center #108
        2: [109, 111],  # right #116
    }

    #TODO do not throw out outermost vertebrae

    # lane -> list of (vert_idx, loc, coords), all three locs per vertebra appended if present
    corpus_points_anterior = {0: [], 1: [], 2: []}
    # lane -> {vert_idx: (loc, coords)} for inter-lane distance check
    corpus_points_per_vert: dict[int, dict[int, tuple[int, np.ndarray]]] = {0: {}, 1: {}, 2: {}}

    for idx, vert in enumerate(tqdm(vert_in_order, desc="Processing vertebrae", unit="vertebra", disable=not show_progress)):
        for lane_id in range(3):
            for loc in corpus_points_anterior_locations[lane_id]:
                if (vert, loc) in poi_ref:
                    picked = np.asarray(poi_ref[vert, loc])
                    corpus_points_anterior[lane_id].append((idx, loc, picked))
                    if idx not in corpus_points_per_vert[lane_id]:
                        corpus_points_per_vert[lane_id][idx] = (loc, picked)

    # for each lane, fit a spline through available points and detect sharp bends
    lane_names = {0: "left", 1: "center", 2: "right"}
    angle_threshold_deg = 30.0
    n_samples = 200
    for lane_id in range(3):
        lane_pts = corpus_points_anterior[lane_id]
        if len(lane_pts) < 2:
            continue

        vert_idxs_for_lane = [vi for vi, loc, p in lane_pts]
        locs_for_lane = [loc for vi, loc, p in lane_pts]
        pts = np.vstack([p for vi, loc, p in lane_pts])

        # track the maximum angle and the responsible loc per vertebra index
        per_vert_max_angle: dict[int, tuple[float, int]] = {}

        spline_ok = False
        u: np.ndarray = np.empty(0)
        u_fine: np.ndarray = np.empty(0)
        samples: np.ndarray = np.empty((0, 3))
        if len(lane_pts) >= 4:
            try:
                tck, u = splprep(pts.T, s=0, k=min(3, len(lane_pts) - 1))
                u_fine = np.linspace(0, 1, n_samples)
                x_f, y_f, z_f = splev(u_fine, tck)
                samples = np.vstack([x_f, y_f, z_f]).T
                spline_ok = True
            except Exception:
                spline_ok = False

        if spline_ok:
            # angles between consecutive segments on the densely sampled spline
            vecs1 = samples[1:-1] - samples[:-2]
            vecs2 = samples[2:] - samples[1:-1]
            norms1 = np.linalg.norm(vecs1, axis=1)
            norms2 = np.linalg.norm(vecs2, axis=1)
            valid = (norms1 > 1e-6) & (norms2 > 1e-6)
            angles = np.zeros(valid.shape)
            if np.any(valid):
                dots = np.sum(vecs1[valid] * vecs2[valid], axis=1)
                cross_norms = np.linalg.norm(np.cross(vecs1[valid], vecs2[valid]), axis=1)
                angles[valid] = np.degrees(np.arctan2(cross_norms, dots))

            u_arr = np.asarray(u)
            for bi in range(len(angles)):
                if not valid[bi]:
                    continue
                param = u_fine[bi + 1]
                nearest_pos = int(np.argmin(np.abs(u_arr - param)))
                vert_idx = vert_idxs_for_lane[nearest_pos]
                ang = float(angles[bi])
                if ang > per_vert_max_angle.get(vert_idx, (-np.inf, None))[0]:
                    per_vert_max_angle[vert_idx] = (ang, locs_for_lane[nearest_pos])
        else:
            # <4 points or spline failed: compute the bend angle directly at each
            # interior original POI between its incoming and outgoing chord.
            for j in range(1, len(pts) - 1):
                v1 = pts[j] - pts[j - 1]
                v2 = pts[j + 1] - pts[j]
                n1 = float(np.linalg.norm(v1))
                n2 = float(np.linalg.norm(v2))
                if n1 < 1e-6 or n2 < 1e-6:
                    continue
                dot = float(np.dot(v1, v2))
                cross = float(np.linalg.norm(np.cross(v1, v2)))
                per_vert_max_angle[vert_idxs_for_lane[j]] = (float(np.degrees(np.arctan2(cross, dot))), locs_for_lane[j])

        for vert_idx, (ang, loc) in per_vert_max_angle.items():
            if ang <= angle_threshold_deg:
                continue
            vert = vert_in_order[vert_idx]
            description = f"Lane {lane_names[lane_id]} sharp bend at loc {loc}: angle {ang:.1f} deg"
            relevant = {"lane": lane_names[lane_id], "angle_deg": ang, "loc": loc}
            report_lines.append(LogicReport(subject_id, vert, Location(loc), ang, relevant, description))
            if verbose:
                logger.print(f"{subject_id} vert {vert}: {description}", Log_Type.OK)

    # detect inter-lane distance changes between consecutive vertebrae
    # compute per-vert left-center and center-right distances
    abs_dist_threshold = 5.0
    rel_dist_threshold = 0.4
    prev_d_left_center = None
    prev_d_center_right = None
    for i, vert in enumerate(vert_in_order):
        left_entry = corpus_points_per_vert[0].get(i)
        center_entry = corpus_points_per_vert[1].get(i)
        right_entry = corpus_points_per_vert[2].get(i)
        if left_entry is None or center_entry is None or right_entry is None:
            prev_d_left_center = None
            prev_d_center_right = None
            continue
        loc_left, left = left_entry
        _loc_center, center = center_entry
        loc_right, right = right_entry
        d_lc = float(np.linalg.norm(left - center))
        d_cr = float(np.linalg.norm(center - right))
        if prev_d_left_center is not None and prev_d_center_right is not None:
            change_lc = abs(d_lc - prev_d_left_center)
            change_cr = abs(d_cr - prev_d_center_right)
            if change_lc > max(abs_dist_threshold, rel_dist_threshold * prev_d_left_center):
                severity = change_lc
                description = f"Left-center inter-lane distance changed by {change_lc:.2f}"
                relevant = {"vert": vert, "d_lc": d_lc, "prev_d_lc": prev_d_left_center}
                report_lines.append(LogicReport(subject_id, vert, Location(loc_left), severity, relevant, description))
            if change_cr > max(abs_dist_threshold, rel_dist_threshold * prev_d_center_right):
                severity = change_cr
                description = f"Center-right inter-lane distance changed by {change_cr:.2f}"
                relevant = {"vert": vert, "d_cr": d_cr, "prev_d_cr": prev_d_center_right}
                report_lines.append(LogicReport(subject_id, vert, Location(loc_right), severity, relevant, description))

        prev_d_left_center = d_lc
        prev_d_center_right = d_cr

    return report_lines


def _proc(subject: Path, ds_dir: Path, opt):
    subject_id = subject.name
    if not subject.is_dir():
        return

    # find all rawdata ct files
    img_paths = search_path(ds_dir / opt.rawdata / subject_id, f"{subject_id}*_ct.nii.gz")
    assert len(img_paths) > 0, f"No CT image found for subject {subject_id}"
    for img_path in img_paths:
        subject_ct_id = img_path.name.split(".")[0]
        #
        out_excel = opt.out_root.joinpath(opt.der_out_prefix + opt.der_poi_mainpred, ds_dir.name, f"{subject_ct_id}_poi_neighbor_angle_report.xlsx")
        if out_excel.exists() and not opt.overwrite:
            logger.print(f"{subject_ct_id}: POI report already exists, skipping: {out_excel}", Log_Type.WARNING)
            continue
        out_excel.parent.mkdir(parents=True, exist_ok=True)
        #

        # logger.print("Subject:", subject_ct_id)
        # img_path = search_path_single(ds_dir / RAWDATA / subject_id, f"{subject_id}*_ct.nii.gz")
        img_bidsf = BIDS_FILE(img_path, dataset=ds_dir)
        split_info = img_bidsf.info["split"] if "split" in img_bidsf.info else None
        vert_path = img_bidsf.get_changed_path(file_type="nii.gz", bids_format="msk", info={"seg": "vert"}, parent=opt.der_vert)
        # VERT MASK IS NOT MODIFIED, so just copy
        assert vert_path.exists(), f"{subject_ct_id}: Original vert mask does not exist: {vert_path}"
        subreg_path = img_bidsf.get_changed_path(file_type="nii.gz", bids_format="msk", info={"seg": "subreg"}, parent=opt.der_subreg)
        assert subreg_path.exists(), f"{subject_ct_id}: NO subreg mask exists: {subreg_path}"

        # load both masks
        subreg_nii = NII.load(subreg_path, seg=True).reorient_()
        subreg_nii.map_labels_({51: 49, 50: 49}, verbose=False)
        vert_nii = NII.load(vert_path, seg=True).reorient_()
        subreg_nii.assert_affine(other=vert_nii, verbose=logger, raise_error=True)

        poi_ref_path = img_bidsf.get_changed_path(
            file_type="json", bids_format="poi", info={"seg": "vert", "mod": "ct"}, parent=opt.der_poi_mainpred
        )
        if not poi_ref_path.exists():
            poi_ref_path = img_bidsf.get_changed_path(
                file_type="json",
                bids_format="poi",
                info={"seg": "vert", "mod": None, "source": "deterministic"},
                parent=opt.der_poi_mainpred,
            )
        poi_proj_ref_path = img_bidsf.get_changed_path(
            file_type="json", bids_format="poi", info={"seg": "vert", "mod": "ct"}, parent=opt.der_poi_mainpred + "_sproj"
        )
        if not poi_ref_path.exists():
            logger.print(f"{subject_ct_id}: Main POI prediction file does not exist: {poi_ref_path}", Log_Type.FAIL)
            continue
        poi_ref = POI.load(poi_ref_path).reorient_()
        #if poi_proj_ref_path.exists():
        #    poi_ref_proj = POI.load(poi_proj_ref_path).reorient_()
        #else:
        #    logger.print(f"{subject_ct_id}: Projection pred does not exist: {poi_proj_ref_path}", Log_Type.FAIL)
        #    continue

        if split_info is None:
            poi_4_rotation_p = search_path_single(
                ds_dir / opt.der_direction / subject_id, f"**/{subject_ct_id.split('_')[0]}*_seg-vert*poi.json"
            )
        else:
            poi_4_rotation_p = search_path_single(
                ds_dir / opt.der_direction / subject_id, f"**/{subject_ct_id.split('_')[0]}*split-{split_info}_seg-vert*poi.json"
            )
        if poi_4_rotation_p is not None and poi_4_rotation_p.exists():
            poi_4_rotation = POI.load(poi_4_rotation_p).reorient_()
            for v in poi_ref.keys_region():
                if v < opt.vertebra_from:
                    continue
                for loc in [
                    Location.Vertebra_Direction_Inferior,
                    Location.Vertebra_Direction_Posterior,
                    Location.Vertebra_Direction_Right,
                    Location.Vertebra_Corpus,
                ]:
                    assert (
                        v,
                        loc,
                    ) in poi_4_rotation, f"{subject_ct_id}: POI for rotation missing location {loc} in vertebra {v}, using paths {poi_ref_path} and {poi_4_rotation_p}"
                    if (v, loc) not in poi_ref:
                        poi_ref[v, loc] = poi_4_rotation[v, loc]

        report_lines = make_poi_report_corpus_lanes(
            poi_ref,
            None,#poi_ref_proj,
            subject_ct_id,
            vert_nii,
            subreg_nii,
            debug=False,
            #do_rotate_around_corpus=poi_4_rotation_p is not None,
            ignore_poi=opt.ignore_poi,
            vertebra_from=opt.vertebra_from,
        )

        if len(report_lines) > 0:
            logger.print(f"{subject_ct_id}: Saving {len(report_lines)} issues to {out_excel}", Log_Type.SAVE)
        save_logic_report(report_lines, out_excel)


if __name__ == "__main__":
    import cProfile

    opt = InferenceConfig.get_opt()
    if opt.cprofile_this:
        opt.num_threads = 1  # avoid multithreading issues with cprofile

    poi_mainpreds = [] + opt.der_poi_mainpred if isinstance(opt.der_poi_mainpred, list) else [opt.der_poi_mainpred]
    for poi_mainpred in poi_mainpreds:
        logger.print(f"Running POI report for main prediction: {poi_mainpred}", Log_Type.STAGE)
        opt.der_poi_mainpred = poi_mainpred

        for ds_dir in opt.root.iterdir():
            if not ds_dir.is_dir():
                continue
            if not ds_dir.name.startswith("dataset-"):
                continue

            if ds_dir.name not in opt.datasets:
                continue

            # has rawdata folder
            raw_dir = ds_dir.joinpath(opt.rawdata)
            assert raw_dir.exists(), f"No rawdata dir in dataset dir: {ds_dir}"

            logger.print("Processing dataset dir:", ds_dir.name, Log_Type.STAGE)

            subjects = list(raw_dir.iterdir())

            if not opt.cprofile_this:
                Parallel(n_jobs=opt.num_threads)(delayed(_proc)(subject, ds_dir, opt) for subject in subjects)
            else:
                for subject in subjects:
                    # if "e014" not in subject.name:
                    #    continue
                    with cProfile.Profile() as pr:
                        _proc(subject, ds_dir, opt)
                    pr.dump_stats(
                        f"/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/data_analysis/cprofiles/{ds_dir.name}_{subject.name}_poi_neighbor_angle_report.prof"
                    )
                    break

            # break
