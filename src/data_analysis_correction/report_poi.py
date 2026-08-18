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
from utils.vertebra_rotation import rotate_3darray, calc_orientation_from_poi, calc_orientation_from_poi_checked
from report_utils import (
    LogicReport,
    SPATIAL_LOGIC_CONSTRAINTS_DICT,
    SUBREGION_CONSTRAINT_DICT,
    poi_touches_subreg,
    save_logic_report,
    SUBREGION_SOFTCONSTRAINT_DICT,
    is_truncated,
)
from report_config import PipelineConfig as InferenceConfig
from tqdm import tqdm

logger = No_Logger(prefix="verse_cseg")


def make_single_vert_poi_report(
    c: COORDINATE,
    c_proj: COORDINATE,
    subject_id: str,
    vert: int,
    poi_loc: Location,
    vert_neighbor: int | None,
    other_poi: POI,
    other_poi_proj: POI,
    #
    orig_subreg_crop: NII,
    c_orig: COORDINATE,
    c_orig_proj: COORDINATE,
    is_first_or_last: bool,
    exempt_spatial: bool = False,
):
    report_lines: list[LogicReport] = []

    # check the SurfaceDistance of the POI
    # region SurfaceDistance
    surface_dist = np.linalg.norm(np.asarray(c) - np.asarray(c_proj))
    if surface_dist > 4:
        report_lines.append(
            LogicReport(
                subject_name=subject_id,
                vertebra=vert,
                location=poi_loc,
                severity=np.sqrt(float(surface_dist - 3.0)) - 0.9,
                relevant_data={"surface_distance_mm": surface_dist},
                description="Large surface distance",
            )
        )
    # endregion

    # check if projected poi touches allowed subregion
    # region SubregionConstraint
    allowed_subreg = SUBREGION_CONSTRAINT_DICT.get(poi_loc.value, None)
    touched_subreg = poi_touches_subreg(c_orig_proj, orig_subreg_crop)
    if touched_subreg is None:
        logger.print(
            f"{subject_id}-v={vert}-{poi_loc, poi_loc.value}: Projected POI coordinate {c_orig_proj} does not touch subreg mask with shape {orig_subreg_crop.shape}",
            Log_Type.WARNING,
        )
    if allowed_subreg is not None and touched_subreg not in allowed_subreg:
        allowed_soft_subreg = SUBREGION_SOFTCONSTRAINT_DICT.get(poi_loc.value, None)
        report_lines.append(
            LogicReport(
                subject_name=subject_id,
                vertebra=vert,
                location=poi_loc,
                severity=10.0 if allowed_soft_subreg is None or touched_subreg not in allowed_soft_subreg else 1.0,
                relevant_data={"touched_subregion": touched_subreg},
                description=f"Projected POI touches invalid subregion, allowed {allowed_subreg}",
            )
        )
    # endregion

    # check spatial logic constraints
    # region SpatialLogicConstraints
    spatial_lc_list = SPATIAL_LOGIC_CONSTRAINTS_DICT.get(poi_loc.value, [])
    if exempt_spatial:
        # Exempt from the ordering constraints only: the surface-distance and subregion checks
        # still apply. See `spatial_exemption` in report_config.py for which vertebrae qualify.
        spatial_lc_list = []
    for lc in spatial_lc_list:
        loc1 = poi_loc
        loc2 = lc.location2
        axis = lc.axis
        c1 = other_poi_proj[vert, loc1]
        axis_idx = other_poi_proj.get_axis(axis)

        if not isinstance(loc2, list):
            loc2 = [loc2]
        for loc2 in loc2:
            if (vert, loc2) not in other_poi_proj:
                continue
            c2 = other_poi_proj[vert, loc2]
            coord_diff = np.asarray(c1) - np.asarray(c2)
            # negate if different direction
            if axis != other_poi_proj.orientation[axis_idx]:
                coord_diff[axis_idx] = -coord_diff[axis_idx]
            if coord_diff[axis_idx] <= 0:
                report_lines.append(
                    LogicReport(
                        subject_name=subject_id,
                        vertebra=vert,
                        location=poi_loc,
                        severity=float(abs(coord_diff[axis_idx])) / 2.0,
                        relevant_data={
                            "axis": (axis, axis_idx),
                            "coord1": c1[axis_idx],
                            "loc2": loc2,
                            "coord2": c2[axis_idx],
                            "coord_diff": coord_diff.tolist(),
                        },
                        description=f"Spatial logic constraint violated: {lc.get_singular_str(loc1, axis, loc2)}",
                    )
                )
    # endregion

    # TODO: check relation between corpus points and corpus COM in L/R dimension, detect shift between neighboring vertebrae
    if vert_neighbor is not None and not is_first_or_last:
        # check relation to corpus in L/R dimension vs. relation of both corpus COMs
        # TODO
        pass

    # TODO: check if all three corpus lines at front have somewhat consistent direction vectors?
    # can we enforce this in the correction process?

    return report_lines


def make_vertebra_poi_report(
    poi_ref: POI,
    poi_ref_proj: POI | None,
    subject_id: str,
    vert: int,
    vert_msk: NII,
    subreg_msk: NII,
    is_first_or_last: bool,
    vert_neighbor: int | None = None,
    debug: bool = False,
    spatial_exemption: str = "truncated",
    truncation_margin: int = 0,
    do_rotate_around_corpus: bool = True,
    ignore_poi: list[Location] = [],
    first_v: bool = False,
) -> list[LogicReport]:
    """Report all logic violations for a single vertebra.

    Split out of `make_poi_report` so the correction scripts can re-score one vertebra
    after a candidate change (see `report_utils.guarded_apply`) without rerunning the
    whole subject.
    """
    report_lines: list[LogicReport] = []

    v_msk = vert_msk.extract_label(vert)

    # Which vertebrae are exempt from the spatial ordering constraints.
    #   "truncated"  - only those running into the edge of the scanned volume (default)
    #   "first_last" - the outermost vertebra of the scan, whether or not it is truncated
    #   "none"       - nobody is exempt
    if spatial_exemption == "none":
        exempt_spatial = False
    elif spatial_exemption == "first_last":
        exempt_spatial = is_first_or_last
    else:
        exempt_spatial = is_truncated(v_msk, margin=truncation_margin)

    v_poi = poi_ref
    if poi_ref_proj is None:
        surface_msk = v_msk  # .compute_surface_mask(connectivity=3, dilated_surface=False)
        v_poi_proj = surface_project_poi(v_poi.extract_region(vert), surface_msk, requires_filling=False)
    else:
        v_poi_proj = poi_ref_proj.extract_region(vert)

    crop = v_msk.compute_crop(dist=16)
    # crop = v_poi.
    v_msk_crop = v_msk.apply_crop(crop)
    if debug:
        v_msk_crop.save(
            f"/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/data_analysis/debug/{subject_id}-v{vert}_vmsk_crop.nii.gz"
        )

    # project everything
    # surf_crop = v_msk_crop.compute_surface_mask(connectivity=3, dilated_surface=False)

    subreg_crop = subreg_msk.apply_crop(crop) * v_msk_crop
    poi_crop = v_poi.apply_crop(crop)
    poi_proj_crop = v_poi_proj.apply_crop(crop)

    if do_rotate_around_corpus:
        R, corpus_com, rel_to_corpus, PIR_angle_degrees, frame_status = calc_orientation_from_poi_checked(poi_crop, vert)
        if frame_status != "ok":
            # A skewed frame silently flips every spatial check for this vertebra, so say so
            # rather than emitting dozens of bogus violations without explanation.
            logger.print(
                f"{subject_id}-v={vert}: vertebra direction frame {frame_status}"
                + ("; spatial checks use the global image axes" if rel_to_corpus is None else ""),
                Log_Type.WARNING,
            )
        if rel_to_corpus is None:
            do_rotate_around_corpus = False

    if do_rotate_around_corpus:
        # print("rel_to_corpus", rel_to_corpus)
        # print("R", R)
        # v_msk_crop = v_msk_crop.set_array(rotate_3darray(v_msk_crop.get_array(), R, center=corpus_com))
        # subreg_crop = subreg_crop.set_array(rotate_3darray(subreg_crop.get_array(), R, center=corpus_com))
        if debug:
            v_r_msk_crop = v_msk_crop.set_array(rotate_3darray(v_msk_crop.get_array(), R, center=corpus_com))
            v_r_msk_crop.save(
                f"/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/data_analysis/debug/{subject_id}-v{vert}_vmsk_crop_rotated.nii.gz"
            )

        def apply_rotation_fun(x, y, z) -> COORDINATE:
            coord = np.subtract(np.asarray((x, y, z)), corpus_com)
            rotated_coord = np.dot(coord, R)
            rotated_coord = np.add(rotated_coord, corpus_com)
            return tuple(rotated_coord)

        poi_r_crop = poi_crop.apply_all(apply_rotation_fun)
        poi_r_proj_crop = poi_proj_crop.apply_all(apply_rotation_fun)
        # poi_r_proj_crop = surface_project_poi(poi_r_proj_crop, surf_crop)
    else:
        poi_r_crop = poi_crop
        poi_r_proj_crop = poi_proj_crop
    # poi_proj_crop = surface_project_poi(poi_proj_crop, surf_crop)

    for poi_loc_val in poi_ref.keys_subregion():
        # if poi_loc_val != 82:
        #    continue
        poi_loc = Location(poi_loc_val)
        if poi_loc in ignore_poi:
            continue

        if (vert, poi_loc) not in poi_r_crop:
            continue
        c = poi_r_crop[vert, poi_loc]
        c_proj = poi_r_proj_crop[vert, poi_loc]
        print(poi_loc.value, poi_proj_crop[vert, poi_loc], c, c_proj) if first_v and debug else None

        new_reports = make_single_vert_poi_report(
            c,
            c_proj,
            subject_id,
            vert,
            poi_loc,
            vert_neighbor,
            poi_r_crop,
            poi_r_proj_crop,
            subreg_crop,
            poi_crop[vert, poi_loc],
            poi_proj_crop[vert, poi_loc],
            is_first_or_last,
            exempt_spatial=exempt_spatial,
        )
        if new_reports is not None:
            report_lines.extend(new_reports)

    return report_lines


def make_poi_report(
    poi_ref: POI,
    poi_ref_proj: POI | None,
    subject_id: str,
    vert_msk: NII,
    subreg_msk: NII,
    debug: bool = False,
    do_rotate_around_corpus: bool = True,
    ignore_poi: list[Location] = [],
    vertebra_from: int = 6,
    spatial_exemption: str = "truncated",
    truncation_margin: int = 0,
    # surface_msk: NII,
) -> list[LogicReport]:
    report_lines: list[LogicReport] = []

    first_v = True

    vert_in_order = sorted(poi_ref.keys_region(), key=lambda v: v_idx_order.index(v))
    vert_in_order = [v for v in vert_in_order if v >= vertebra_from - 1]

    for idx, vert in enumerate(tqdm(vert_in_order, desc="Processing vertebrae", unit="vertebra")):
        is_first_or_last = idx == 0 or idx == len(vert_in_order) - 1
        # if vert not in [23]:
        #    continue
        # if vert not in [10, 28]:
        #    continue
        vert_neighbor = vert_in_order[idx + 1] if idx + 1 < len(vert_in_order) else None

        report_lines.extend(
            make_vertebra_poi_report(
                poi_ref,
                poi_ref_proj,
                subject_id,
                vert,
                vert_msk,
                subreg_msk,
                is_first_or_last,
                vert_neighbor=vert_neighbor,
                debug=debug,
                do_rotate_around_corpus=do_rotate_around_corpus,
                ignore_poi=ignore_poi,
                first_v=first_v,
                spatial_exemption=spatial_exemption,
                truncation_margin=truncation_margin,
            )
        )

        first_v = False
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
        out_excel = opt.out_root.joinpath(opt.der_out_prefix + opt.der_poi_mainpred, ds_dir.name, f"{subject_ct_id}_poi_report.xlsx")
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
        if poi_proj_ref_path.exists():
            poi_ref_proj = POI.load(poi_proj_ref_path).reorient_()
        else:
            poi_ref_proj = None

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

        report_lines = make_poi_report(
            poi_ref,
            poi_ref_proj,
            subject_ct_id,
            vert_nii,
            subreg_nii,
            debug=False,
            do_rotate_around_corpus=poi_4_rotation_p is not None,
            ignore_poi=opt.ignore_poi,
            vertebra_from=opt.vertebra_from,
            spatial_exemption=opt.spatial_exemption,
            truncation_margin=opt.truncation_margin,
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
                        f"/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/data_analysis/cprofiles/{ds_dir.name}_{subject.name}_poi_report.prof"
                    )
                    break

            # break
