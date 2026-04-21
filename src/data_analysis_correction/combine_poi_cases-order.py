import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))
from pathlib import Path
from TPTBox import No_Logger, Log_Type, NII, BIDS_FILE, POI, Location, Vertebra_Instance
from joblib import Parallel, delayed
from panoptica import Metric
from utils.filepaths import search_path_single, search_path
from utils.vertebra_rotation import (
    calc_orientation_from_poi,
    move_poi_along_axis,
    find_extreme_point_along_axis,
    get_axis_direction_vector,
)
import numpy as np
from report_utils import SUBREGION_CONSTRAINT_DICT, load_agg_report_df, convert_agg_report_to_reported_bool_dict, is_poi_reported
import shutil
from utils.misc import surface_project_poi_vert_wise

logger = No_Logger(prefix="verse_cseg")

ROOT = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/")
RAWDATA = "rawdata"
DERIV_SUBREG = "derivatives_combined"  # "derivatives_subreg"
DERIV_VERT = "derivatives_combined"
#
DERIV_POI_PREDS = [
    "derivatives_poi_FirstIter/surface_cc3-bs32-v3",
    "derivatives_poi_FirstIter/surface_cc3-bs32-v3_flipped",
    "derivatives_poi_ForVerse/surface_cc3-bs32-v3",
    "derivatives_poi_ForVerse/surface_cc3-bs32-v3_flipped",
    "derivatives_poi_ForVerse/surface_cc3-bs32-v1",
    # "derivatives_poi_surface_neighbor_cc3-v1",
    # "derivatives_poi_surface_neighbor_cc3-v1_flipped",
    # "derivatives_poi_surface_cc3-v0",
    # "derivatives_poi_surface_cc3-v0_flipped",
    # "derivatives_poi_surface_cc3-bs32-v1",
    "derivatives_poi_ForVerse/surface_cc3-bs32-v1_flipped",
    # "derivatives_poi_surface_cc3-bs32-v4",
    # "derivatives_poi_surface_cc3-bs32-v4_flipped",
]
DERIV_DET = "derivatives_poi_deterministic"
#
DERIV_OUT = "derivatives_poi_automatic_correction-v4-onlygood"
# TODO use reports and don't use points that are marked there?

###
REPORT_ROOT = "/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/data_analysis/"
REPORT_PREFIX = "TEST_"
REPORT_AGG_NAME = "aggregated_poi_report.xlsx"


def _proc(
    subject: Path,
    ds_dir: Path,
    report_der2dict: dict,
    verbose=False,
):
    subject_id = subject.name
    if not subject.is_dir():
        return

    # find all rawdata ct files
    img_paths = search_path(ds_dir / RAWDATA / subject_id, f"{subject_id}*_ct.nii.gz")
    assert len(img_paths) > 0, f"No CT image found for subject {subject_id}"
    for img_path in img_paths:
        subject_ct_id = img_path.name.split(".")[0]
        # logger.print("Subject:", subject_ct_id)
        # img_path = search_path_single(ds_dir / RAWDATA / subject_id, f"{subject_id}*_ct.nii.gz")
        img_bidsf = BIDS_FILE(img_path, dataset=ds_dir)

        poi_out_path = img_bidsf.get_changed_path(
            file_type="json",
            bids_format="poi",
            info={"seg": "vert", "mod": "ct"},
            parent=DERIV_OUT,
        )
        poi_out_path_global = img_bidsf.get_changed_path(
            file_type="mrk.json",
            bids_format="poi",
            info={"seg": "vert", "mod": "ct"},
            parent=DERIV_OUT,
        )
        if poi_out_path.exists():
            logger.print(f"{subject_ct_id}: Output POI file already exists, skipping", Log_Type.WARNING)
            continue
        poi_out_path.parent.mkdir(parents=True, exist_ok=True)

        vert_path = img_bidsf.get_changed_path(file_type="nii.gz", bids_format="msk", info={"seg": "vert"}, parent=DERIV_VERT)
        # VERT MASK IS NOT MODIFIED, so just copy
        assert vert_path.exists(), f"{subject_ct_id}: Original vert mask does not exist: {vert_path}"
        subreg_path = img_bidsf.get_changed_path(file_type="nii.gz", bids_format="msk", info={"seg": "subreg"}, parent=DERIV_SUBREG)
        assert subreg_path.exists(), f"{subject_ct_id}: NO subreg mask exists: {subreg_path}"

        # load both masks
        subreg_nii = NII.load(subreg_path, seg=True)
        subreg_nii.map_labels_({51: 49, 50: 49}, verbose=False)
        vert_nii = NII.load(vert_path, seg=True)
        subreg_nii.assert_affine(other=vert_nii, verbose=logger, raise_error=True)

        surface_nii = vert_nii.compute_surface_mask(connectivity=3, dilated_surface=False)
        # load all predicted poi files
        poi_paths = {
            k: img_bidsf.get_changed_path(file_type="json", bids_format="poi", info={"seg": "vert", "mod": "ct"}, parent=k)
            for k in DERIV_POI_PREDS
        }
        poi_paths[DERIV_DET] = img_bidsf.get_changed_path(
            file_type="json", bids_format="poi", info={"seg": "vert", "source": "deterministic"}, parent=DERIV_DET
        )
        poi_dict: dict[str, POI] = {k: POI.load(v) if v.exists() else None for k, v in poi_paths.items()}  # type: ignore

        poi_proj_paths: dict[str, Path] = {
            k: img_bidsf.get_changed_path(file_type="json", bids_format="poi", info={"seg": "vert", "mod": "ct"}, parent=k + "_sproj")
            for k in poi_paths.keys()
        }
        # project everything
        poi_proj_dict: dict[str, POI] = {k: POI.load(v) if v.exists() else None for k, v in poi_proj_paths.items()}  # type: ignore

        #
        for k in list(poi_dict.keys()):
            if not isinstance(poi_dict[k], POI):
                logger.print(f"{subject_ct_id}: POI file does not exist: {k}", Log_Type.FAIL)
                poi_dict.pop(k)
            # else:
            # projected version
            #    poi_dict[k + "_proj"] = surface_project_poi(v, surface_nii)

        if len(poi_dict) == 0:
            logger.print(f"{subject_ct_id}: No POI predictions found, skipping", Log_Type.FAIL)
            continue

        # calc orientation of every vertebra to be used later
        vert_orientations = {}
        for idx, vert in enumerate(vert_nii.unique()):  # skip first and last
            if DERIV_DET in poi_dict and poi_dict[DERIV_DET] is not None and idx != 0 and idx != len(vert_nii.unique()) - 1:
                vert_orientations[vert] = calc_orientation_from_poi(poi_dict[DERIV_DET], vert)[2]
            else:
                vert_orientations[vert] = None

        remove_pois = []

        poi_source_priority_order = list(poi_dict.keys())
        primus = poi_dict[poi_source_priority_order[0]]

        for r, s, c in primus.items():
            is_first_or_last_v = r == vert_nii.unique()[0] or r == vert_nii.unique()[-1]
            vert_instance = Vertebra_Instance(r)
            s_location = Location(s)
            # throw all out if it's marked in the report as an issue for this subject/vert/location

            other_reported_err = [k for k in report_der2dict if is_poi_reported(report_der2dict, subject_ct_id, r, s)]  # type: ignore
            other_c = {k: np.asarray(v[r, s]) for k, v in poi_dict.items() if [r, s] in v and k not in other_reported_err}  # type: ignore

            # if all are reported, use current primus
            if len(other_c) == 0:
                other_c = {poi_source_priority_order[0]: np.asarray(c)}

            new_primus = list(other_c.keys())[0]
            poi_ref = poi_dict[new_primus]  # type: ignore
            c_new = np.asarray(other_c[new_primus])
            # pop the new_primus
            other_c.pop(new_primus)

            # for each reference poi, find corresponding pois in other predictions
            # then compute distances
            other_dist = {k: np.linalg.norm(c_new - pc) for k, pc in other_c.items()}
            other_dist = dict(sorted(other_dist.items(), key=lambda x: x[1]))
            other_dist_sorted_keys = other_dist.keys()

            ######################################################

            # remove points where the subregion is missing
            allowed_subreg = SUBREGION_CONSTRAINT_DICT.get(s_location.value, None)
            avilable_subregs = (subreg_nii * vert_nii.extract_label(r)).unique()
            if allowed_subreg is not None and all(sr not in avilable_subregs for sr in allowed_subreg):
                logger.print(
                    f"{subject_ct_id} V{vert_instance.value} {s_location.name}: No allowed subregions {allowed_subreg} available in subreg mask {avilable_subregs}, removing POI",
                    verbose=verbose,
                )
                remove_pois.append((r, s))
                continue

            ######################################################

            if s in [
                Location.Muscle_Inserts_Articulate_Process_Superior_Left.value,
                Location.Muscle_Inserts_Articulate_Process_Superior_Right.value,
            ]:
                # move towards up for a couple voxels
                # axis_idx = poi_ref.get_axis("S")
                shift = 3
                if not is_first_or_last_v and vert_orientations[r] is not None:
                    c_new = move_poi_along_axis(c_new, poi_ref, "S", vert_orientations[r], 3)
                # inversed = poi_ref.orientation[axis_idx] != "S"
                # shift = 3 * (1 if not inversed else -1)
                # c_new[axis_idx] = c_new[axis_idx] + shift
                logger.print(
                    f"{subject_ct_id} V{vert_instance.value} {s_location.name}: Superior Articulate Process POI shifted {shift} voxels along S axis",
                    verbose=verbose,
                )

            ######################################################

            if vert_instance in Vertebra_Instance.thoracic():
                if s == Location.Muscle_Inserts_Spinosus_Process.value:
                    # proc. spinosus. use most middle one in L/R dimension
                    # take average except for longest distance
                    ks = list(other_dist_sorted_keys)[:-1]
                    all_x = [pc for k, pc in other_c.items() if k in ks] + [c_new]
                    c_new = np.mean(all_x, axis=0)
                    # move a little back
                    if not is_first_or_last_v and vert_orientations[r] is not None:
                        c_new = move_poi_along_axis(c_new, poi_ref, "P", vert_orientations[r], 3)
                    logger.print(
                        f"{subject_ct_id} V{vert_instance.value} {s_location.name}: Spinosus POI adjusted to mean of closest predictions of {all_x}",
                        verbose=verbose,
                    )

                # thoracic: lower inferior facet POI
                if s in [
                    Location.Muscle_Inserts_Articulate_Process_Inferior_Left.value,
                    Location.Muscle_Inserts_Articulate_Process_Inferior_Right.value,
                ]:
                    # take average of neighbor predictions
                    ks = list(other_dist_sorted_keys)
                    all_x = [pc for k, pc in other_c.items() if k in ks if "neighbor" in k] + [c_new]
                    c_new = np.mean(all_x, axis=0)

                    if not is_first_or_last_v and vert_orientations[r] is not None:
                        # calculate center of mass of corresponding subregion in vert mask
                        vert_mask = vert_nii.extract_label(r)
                        subreg_label = 47 if s == Location.Muscle_Inserts_Articulate_Process_Inferior_Left.value else 48
                        subreg_mask = subreg_nii.extract_label(subreg_label)
                        combined_mask = vert_mask * subreg_mask
                        com = combined_mask.center_of_masses()[1]
                        axis_dir = get_axis_direction_vector(vert_orientations[r], poi_ref, "R")
                        # move c_new to coms coordinate in I axis, but keep other coordinates using the vert_orientation
                        # only move inwards to com not outwards
                        com_i = np.dot(com, axis_dir)
                        c_new_i = np.dot(c_new, axis_dir)
                        shift = com_i - c_new_i
                        if s == Location.Muscle_Inserts_Articulate_Process_Inferior_Left.value:
                            shift += 1  # add a little bias to the left side to prevent being too close to the edge.
                            shift = min(0, shift)
                        else:
                            shift -= 1  # add a little bias to the right side to prevent being too close to the edge.
                            shift = max(0, shift)
                        c_new = c_new + axis_dir * shift
                        c_new = move_poi_along_axis(c_new, poi_ref, "I", vert_orientations[r], 3)
                    logger.print(
                        f"{subject_ct_id} V{vert_instance.value} {s_location.name}: Spinosus POI adjusted to mean of closest predictions of {all_x}",
                        verbose=verbose,
                    )

            ######################################################

            # Corpus Anterior Edge points, so superior or inferior corpus points
            # ALL POIS: take most anterior point. then shift 3 voxels anterior then surface project again
            if s in [117, 101, 109, 119, 103, 111]:
                # anterior axis
                # take most anterior point
                all_a = {k: pc for k, pc in other_c.items() if "deterministic" not in k}
                all_a["ref"] = c_new
                # all_a_sorted = dict(sorted(all_a.items(), key=lambda x: x[1][axis_idx], reverse=inversed))
                # take most anterior
                # k_best = list(all_a_sorted.keys())[0]
                candidates = list(all_a.values())
                c_new = find_extreme_point_along_axis(candidates, poi_ref, "A", vert_orientations[r])
                logger.print(
                    f"{subject_ct_id} V{vert_instance.value} {s_location.name}: Anterior POI selected from {candidates}", verbose=verbose
                )
                # c_new = all_a_sorted[k_best]
                if not is_first_or_last_v and vert_orientations[r] is not None:
                    c_new = move_poi_along_axis(c_new, poi_ref, "A", vert_orientations[r], 3)

            ######################################################

            # lumbar: lower inferior facet POI
            if vert_instance in Vertebra_Instance.lumbar():
                if s in [
                    Location.Muscle_Inserts_Articulate_Process_Inferior_Left.value,
                    Location.Muscle_Inserts_Articulate_Process_Inferior_Right.value,
                ]:
                    # take lowest point of all predictions
                    ks = list(other_dist_sorted_keys)
                    all_x = [pc for k, pc in other_c.items() if k in ks and "deterministic" not in k] + [c_new]
                    candidates = all_x
                    c_new = find_extreme_point_along_axis(candidates, poi_ref, "I", vert_orientations[r])
                    # find axis index for inferior
                    # axis_idx = poi_ref.get_axis("I")
                    # inversed = poi_ref.orientation[axis_idx] != "I"
                    # all_x_sorted = sorted(all_x, key=lambda x: x[axis_idx], reverse=inversed)
                    # c_new = all_x_sorted[0]
                    logger.print(
                        f"{subject_ct_id} V{vert_instance.value} {s_location.name}: Lumbar Inferior Articulate POI adjusted to extremum of closest predictions of {all_x}",
                        verbose=verbose,
                    )
                    if not is_first_or_last_v and vert_orientations[r] is not None:
                        c_new = move_poi_along_axis(c_new, poi_ref, "I", vert_orientations[r], 2)

            ######################################################

            # costalis: most lateral point of all predictions
            if s in [82, 83]:  # left/right costal process
                dir = "L" if s == 83 else "R"
                # axis_idx = poi_ref.get_axis("L" if s == 83 else "R")
                # inversed = poi_ref.orientation[axis_idx] != ("L" if s == 83 else "R")
                all_x = [pc for k, pc in other_c.items()] + [c_new]
                # all_x_sorted = sorted(all_x, key=lambda x: x[axis_idx], reverse=inversed)
                # c_new = all_x_sorted[0]
                c_new = find_extreme_point_along_axis(all_x, poi_ref, dir, vert_orientations[r])
                logger.print(
                    f"{subject_ct_id} V{vert_instance.value} {s_location.name}: Costal Process POI adjusted to extremum of closest predictions of {all_x}",
                    verbose=verbose,
                )
                if not is_first_or_last_v and vert_orientations[r] is not None:
                    c_new = move_poi_along_axis(c_new, poi_ref, dir, vert_orientations[r], 2)

            ######################################################
            ######################################################

            poi_ref[r, s] = tuple(c_new)

        for rp in remove_pois:
            poi_ref.remove_(rp)
        if len(remove_pois) > 0:
            logger.print(f"Removed the points {remove_pois} from POI of {subject_ct_id}", Log_Type.STRANGE)

        poi_ref_proj = surface_project_poi_vert_wise(poi_ref, vert_nii, requires_filling=False)

        left_post_corpus_points = [112, 114, 110]
        right_post_corpus_points = [118, 122, 120]
        # post process corpus left/right direction
        for v in poi_ref_proj.keys_region():
            if vert_orientations[v] is None:
                continue
            for sbucket in [
                [117, 121, 118, 124, 119, 123, 120, 122],
                [103, 108, 101, 105, 102, 107, 104, 106],
                [111, 116, 109, 113, 110, 115, 112, 114],
            ]:
                dir = "L"
                axis_dir = get_axis_direction_vector(vert_orientations[v], poi_ref_proj, dir)
                # get average coordinate in dir axis direction
                avg_dir_coord = np.mean([np.dot(np.asarray(poi_ref_proj[v, si]), axis_dir) for si in sbucket if (v, si) in poi_ref_proj])
                for s in sbucket:
                    if (v, s) not in poi_ref_proj:
                        continue
                    # project point so that in dir axis direction matches avg_dir_coord
                    c = np.asarray(poi_ref_proj[v, s])
                    c_diff = avg_dir_coord - np.dot(c, axis_dir)
                    if s in left_post_corpus_points and c_diff > 0:
                        c_diff = 0
                    if s in right_post_corpus_points and c_diff < 0:
                        c_diff = 0
                    c_new = c + axis_dir * c_diff
                    poi_ref_proj[v, s] = tuple(c_new)
                #
        poi_ref_proj = surface_project_poi_vert_wise(poi_ref_proj, vert_nii, requires_filling=False)

        # Post processing some points
        neigh_t_b = {
            124: (117, 119),
            108: (101, 103),
            116: (109, 111),
        }
        # Corpus medial points, move towards half of superior direction between major and minor
        for v in poi_ref_proj.keys_region():
            for s in [124, 108, 116]:
                if (v, s) not in poi_ref_proj:
                    continue
                t, b = neigh_t_b[s]
                if (v, t) in poi_ref_proj and (v, b) in poi_ref_proj:
                    c_t = np.asarray(poi_ref_proj[v, t])
                    c_b = np.asarray(poi_ref_proj[v, b])
                    if vert_orientations[v] is None:
                        continue
                    axis_s = get_axis_direction_vector(vert_orientations[v], poi_ref_proj, "S")
                    superior_n = 0.5 * (np.dot(c_t, axis_s) + np.dot(c_b, axis_s))
                    superior_d = superior_n - np.dot(np.asarray(poi_ref_proj[v, s]), axis_s)
                    c_new = np.asarray(poi_ref_proj[v, s]) + axis_s * superior_d

                    # c_new = move_poi_along_axis(c_new, poi_ref, "S", vert_orientations[v], 2)
                    poi_ref_proj[v, s] = tuple(c_new)
                    # logger.print(
                    #    f"{subject_ct_id} V{v} {Location(s).name}: Corpus Medial POI adjusted to half between superior and inferior points shifted S by 2 voxels",
                    #    verbose=verbose,
                    # )

        #######################################
        #######################################
        poi_ref_proj = surface_project_poi_vert_wise(poi_ref_proj, vert_nii, requires_filling=False)
        surface_nii.save(poi_out_path.with_suffix(".nii.gz"))
        poi_ref_proj.save(poi_out_path)
        poi_ref_proj.to_global().save_mrk(poi_out_path_global, split_by_region=True)
    return True


if __name__ == "__main__":
    for ds_dir in ROOT.iterdir():
        if not ds_dir.is_dir():
            continue
        if not ds_dir.name.startswith("dataset-"):
            continue

        # if not ds_dir.name == "dataset-verse19training_1mmiso":
        #    continue

        # has rawdata folder
        raw_dir = ds_dir.joinpath(RAWDATA)
        assert raw_dir.exists(), f"No rawdata dir in dataset dir: {ds_dir}"

        logger.print("Processing dataset dir:", ds_dir.name, Log_Type.STAGE)

        subjects = sorted(list(raw_dir.iterdir()), key=lambda x: x.name)

        # load reports
        report_der2dict = {}
        for der_name in DERIV_POI_PREDS:
            report_df = load_agg_report_df(REPORT_ROOT, REPORT_PREFIX, der_name, REPORT_AGG_NAME)
            if report_df is not None:
                report_der2dict[der_name] = convert_agg_report_to_reported_bool_dict(report_df)

        Parallel(n_jobs=12)(delayed(_proc)(subject, ds_dir, report_der2dict) for subject in subjects)
        # for subject in subjects:
        #    _proc(subject, ds_dir, report_der2dict)
        # break

        # break
