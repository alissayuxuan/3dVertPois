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
import numpy as np
import shutil
from utils.misc import surface_project_poi, surface_project_poi_vert_wise

logger = No_Logger(prefix="verse_cseg")

ROOT = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/")
RAWDATA = "rawdata"
DERIV_SUBREG = "derivatives_combined"  # "derivatives_subreg"
DERIV_VERT = "derivatives_combined"

DERIV_POI_MAINPRED = "derivatives_poi_surface-neighbor-neighaug-project_gt_cc3-exclude6-v2"
DERIV_POI_PREDS = [
    "derivatives_poi_surface-neighbor-neighaug-project_gt_cc3-exclude6",
    "derivatives_poi_surface_project-gt_cc3-exclude6",
]
DERIV_DET = "derivatives_poi_deterministic"
DERIV_OUT = "derivatives_poi_automatic_correction"
# TODO use reports and don't use points that are marked there?


def _proc(subject: Path, ds_dir: Path):
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

        poi_ref_path = img_bidsf.get_changed_path(
            file_type="json", bids_format="poi", info={"seg": "vert", "mod": "ct"}, parent=DERIV_POI_MAINPRED
        )
        assert poi_ref_path.exists(), f"{subject_ct_id}: Main POI prediction file does not exist: {poi_ref_path}"
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
        poi_ref = POI.load(poi_ref_path)

        # load all predicted poi files
        poi_paths = {
            k: img_bidsf.get_changed_path(file_type="json", bids_format="poi", info={"seg": "vert", "mod": "ct"}, parent=k)
            for k in DERIV_POI_PREDS
        }
        poi_paths[DERIV_DET] = img_bidsf.get_changed_path(
            file_type="json", bids_format="poi", info={"seg": "vert", "source": "deterministic"}, parent=DERIV_DET
        )
        poi_dict: dict[str, POI] = {k: POI.load(v) if v.exists() else None for k, v in poi_paths.items()}

        # project everything
        poi_ref_proj_path = img_bidsf.get_changed_path(
            file_type="json", bids_format="poi", info={"seg": "vert", "mod": "ct"}, parent=DERIV_POI_MAINPRED + "_sproj"
        )
        if poi_ref_proj_path.exists():
            poi_ref_proj = POI.load(poi_ref_proj_path)
        else:
            poi_ref_proj = surface_project_poi_vert_wise(poi_ref, surface_nii)
            poi_ref_proj.save(poi_ref_proj_path)

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

        for r, s, c in poi_ref.items():
            c_new = np.asarray(c)
            vert_instance = Vertebra_Instance(r)
            s_location = Location(s)
            # for each reference poi, find corresponding pois in other predictions
            # then compute distances
            other_c = {k: np.asarray(v[r, s]) for k, v in poi_dict.items()}
            other_dist = {k: np.linalg.norm(c_new - pc) for k, pc in other_c.items()}
            other_dist = dict(sorted(other_dist.items(), key=lambda x: x[1]))
            other_dist_sorted_keys = other_dist.keys()
            # compare main poi_ref to all others,
            c_proj = poi_ref_proj[r, s]

            # TODO: remove all candidate points that have an issue based on the report thingy
            # if none survive, keep original

            ##
            # TODO do we require rotation here? makes it complex
            ##
            if vert_instance in Vertebra_Instance.thoracic():
                if s == Location.Muscle_Inserts_Spinosus_Process.value:
                    # proc. spinosus. use most middle one in L/R dimension
                    # take average except for longest distance?
                    ks = list(other_dist_sorted_keys)
                    all_x = [pc for k, pc in other_c.items() if k in ks] + [c_new]
                    c_new = np.mean(all_x, axis=0)
                    logger.print(
                        f"{subject_ct_id} V{vert_instance.value} {s_location.name}: Spinosus POI adjusted to mean of closest predictions of {all_x}"
                    )

            # Corpus Anterior Edge points, so superior or inferior corpus points
            if s in [117, 101, 109, 119, 103, 111]:
                # anterior axis
                axis_idx = poi_ref.get_axis("A")
                inversed = poi_ref.orientation[axis_idx] != "A"
                # take most anterior point
                all_a = {k: pc for k, pc in other_c.items()}
                all_a["ref"] = c_new
                all_a_sorted = dict(sorted(all_a.items(), key=lambda x: x[1][axis_idx], reverse=inversed))
                # take most anterior
                k_best = list(all_a_sorted.keys())[0]
                logger.print(f"{subject_ct_id} V{vert_instance.value} {s_location.name}: Anterior POI selected from {k_best}")
                c_new = all_a_sorted[k_best]

                # move 3 voxel anteriorly
                shift = 3 * (1 if not inversed else -1)
                c_new[axis_idx] = c_new[axis_idx] + shift
            # ALL POIS: take most anterior point. then shift 3 voxels anterior then surface project again
            # TODO thoracic: lower inferior facet POI
            # take average of neighbor predictions, then move along the vector to the superior POI
            # TODO: lumbar: lower inferior facet POI
            # take lowest point of all predictions

            # TODO: remove points where the subregion is missing

            # TODO: costalis: most lateral point of all predictions
            poi_ref[r, s] = tuple(c_new)
        poi_ref_proj = surface_project_poi_vert_wise(poi_ref, surface_nii)
        surface_nii.save(poi_out_path.with_suffix(".nii.gz"))
        poi_ref_proj.save(poi_out_path)
        poi_ref_proj.to_global().save_mrk(poi_out_path_global, split_by_region=True)


if __name__ == "__main__":
    for ds_dir in ROOT.iterdir():
        if not ds_dir.is_dir():
            continue
        if not ds_dir.name.startswith("dataset-"):
            continue

        if not ds_dir.name == "dataset-verse20validation_1mmiso":
            continue

        # has rawdata folder
        raw_dir = ds_dir.joinpath(RAWDATA)
        assert raw_dir.exists(), f"No rawdata dir in dataset dir: {ds_dir}"

        logger.print("Processing dataset dir:", ds_dir.name, Log_Type.STAGE)

        subjects = list(raw_dir.iterdir())

        # Parallel(n_jobs=8)(delayed(_proc)(subject, ds_dir) for subject in subjects)
        for subject in subjects:
            _proc(subject, ds_dir)
            break

        # break
