import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

from TPTBox import NII, BIDS_FILE, BIDS_Global_info, No_Logger, Log_Type, Location, POI, calc_poi_from_subreg_vert

# from utils.filepaths import filepath_dataset
import numpy as np
from joblib import Parallel, delayed
from TPTBox.core.bids_constants import sequence_splitting_keys
from utils.misc import surface_project_poi

import os


logger = No_Logger(prefix="verse_surface_project")
SKIPPED_SUBJECTS = []


# zoom = (0.8, 0.8, 0.8)
def _proc(name, subject, der_out: str):
    # if "25" not in name:
    #    return
    try:
        q = subject.new_query()
        families = q.loop_dict(key_addendum=["space"])
        for f in families:
            fid = f.family_id

            if ["poi_seg-vert", "msk_seg-vert"] not in f:
                logger.print(fid, f.get_key_len())
                continue
            poi_ref: BIDS_FILE = f["poi_seg-vert"][0]
            vert_ref: BIDS_FILE = f["msk_seg-vert"][0]

            #####
            # outputs
            out_det = poi_ref.get_changed_path(
                file_type="json",
                bids_format="poi",
                parent=der_out,
                make_parent=True,
            )
            out_det_global = poi_ref.get_changed_path(
                file_type="mrk.json",
                bids_format="poi",
                parent=der_out,
                info={"space": "global"},
                make_parent=True,
            )
            # out_det = Path(SAVE_PATH) / name / "poi_predicted.json"
            # out_det.parent.mkdir(parents=True, exist_ok=True)
            #####
            if out_det.exists():
                logger.print("Outputs already exist")
                continue

            poi = poi_ref.open_poi()
            vert_nii = vert_ref.open_nii()
            for v in vert_nii.unique():
                if v == 0:
                    continue
                vpoi = poi.extract_region(v)
                surface_nii = vert_nii.extract_label(v).compute_surface_mask(connectivity=1, dilated_surface=False)
                det_poi = surface_project_poi(vpoi, surface_nii=surface_nii)
                poi_s = [i for i in poi.keys_subregion() if i in vpoi.keys_subregion()]
                for s in poi_s:
                    if s == 0:
                        continue
                    poi[(v, s)] = det_poi[(v, s)]
            # det_poi = det_poi.round(3)
            poi.save(out_det)
            poi.to_global().save_mrk(out_det_global)

    except Exception as e:
        logger.print(f"[SKIP] Error at {name}: {e}", Log_Type.FAIL)
        SKIPPED_SUBJECTS.append(name)
        raise e


if __name__ == "__main__":
    ds_names = [
        "dataset-verse19training_1mmiso",
        "dataset-verse20training_1mmiso",
        "dataset-verse19validation_1mmiso",
        "dataset-verse20validation_1mmiso",
        "dataset-verse19test_1mmiso",
        "dataset-verse20test_1mmiso",
    ]

    DER_MSK = "derivatives_combined"
    DER_IN = "derivatives_poi_surface_project-gt_cc3-exclude6"  # "derivatives_poi_deterministic"
    der_out = DER_IN + "_sproj"

    for ds_name in ds_names:

        ds_path = f"/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/{ds_name}/"
        bgi = BIDS_Global_info(
            datasets=[ds_path],
            parents=[DER_IN, DER_MSK],
        )

        # Parallel(n_jobs=10, backend="threading")(
        #    delayed(_proc)(name, subject, der_out) for name, subject in bgi.enumerate_subjects(sort=True)
        # )
        for name, subject in bgi.enumerate_subjects(sort=True):
            _proc(name, subject, der_out)
        # break

        if len(SKIPPED_SUBJECTS) > 0:
            print("Skipped subjects:")
            for s in SKIPPED_SUBJECTS:
                print(s)

        # break
