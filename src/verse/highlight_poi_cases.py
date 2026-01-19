import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))
from pathlib import Path
from TPTBox import No_Logger, Log_Type, NII, BIDS_FILE, POI
from joblib import Parallel, delayed
from panoptica import Metric
from utils.filepaths import search_path_single, search_path
import numpy as np
import shutil
from spineps.phase_post import assign_missing_cc
from utils.misc import surface_project_poi

logger = No_Logger(prefix="verse_cseg")

ROOT = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/")
RAWDATA = "rawdata"
DERIV_SUBREG = "derivatives_combined"  # "derivatives_subreg"
DERIV_VERT = "derivatives_combined"

DERIV_POI_MAINPRED = "derivatives_inference_poi_subreg-project_gt-no_freeze-surface-cc3-exclude6_proj"
DERIV_POI_PREDS = [
    "derivatives_poi_deterministic",
]


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

        surface_nii = vert_nii.compute_surface_mask(connectivity=1, dilated_surface=False)

        poi_ref_path = img_bidsf.get_changed_path(
            file_type="json", bids_format="poi", info={"seg": "vert", "mod": "ct"}, parent=DERIV_POI_MAINPRED
        )
        assert poi_ref_path.exists(), f"{subject_ct_id}: Main POI prediction file does not exist: {poi_ref_path}"
        poi_ref = POI.load(poi_ref_path)

        # load all predicted poi files
        poi_paths = {
            k: img_bidsf.get_changed_path(file_type="json", bids_format="poi", info={"seg": "vert", "mod": "ct"}, parent=k)
            for k in DERIV_POI_PREDS
        }
        poi_dict: dict[str, POI] = {k: POI.load(v) if v.exists() else None for k, v in poi_paths.items()}

        # project everything
        poi_ref_proj = surface_project_poi(poi_ref, surface_nii)

        for k, v in poi_dict.items():
            if not isinstance(v, POI):
                logger.print(f"{subject_ct_id}: POI file does not exist: {k}", Log_Type.FAIL)
                poi_dict.pop(k)
            else:
                # projected version
                poi_dict[k + "_proj"] = surface_project_poi(v, surface_nii)

        if len(poi_dict) == 0:
            logger.print(f"{subject_ct_id}: No POI predictions found, skipping", Log_Type.FAIL)
            continue

        for r, s, c in poi_ref.items():
            # for each reference poi, find corresponding pois in other predictions
            # then compute distances
            other_c = {k: v[r, s] for k, v in poi_dict.items()}
            other_dist = {k: np.linalg.norm(np.asarray(c) - np.asarray(pc)) for k, pc in other_c.items()}
            # compare main poi_ref to all others,
            c_proj = poi_ref_proj[r, s]
            # surface distance of main poi
            surface_dist = np.linalg.norm(np.asarray(c) - np.asarray(c_proj))
            if surface_dist > 4:
                logger.print(f"{subject_ct_id, r, s}: Large surface distance for poi: {surface_dist:.2f} mm", Log_Type.WARNING)
            # distance to nearest poi and also if that poi is projected
            for k, dist in other_dist.items():
                if dist > 6:
                    logger.print(
                        f"{subject_ct_id, r, s}: POI - Distance to {k}: {dist:.2f} mm",
                        Log_Type.WARNING,
                    )

        # TODO logic pois that are above or besides other pois


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

        Parallel(n_jobs=8)(delayed(_proc)(subject, ds_dir) for subject in subjects)
        # for subject in subjects:
        #    _proc(subject, ds_dir)
        #    break

        # break
