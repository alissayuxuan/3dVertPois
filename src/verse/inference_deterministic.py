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

import os


logger = No_Logger(prefix="verse_deterministic_inference")
SKIPPED_SUBJECTS = []


# zoom = (0.8, 0.8, 0.8)
def _proc(name, subject, der_out: str):
    # if "25" not in name:
    #    return
    try:
        q = subject.new_query()
        families = q.loop_dict(key_addendum=["source"])
        for f in families:
            fid = f.family_id

            logger.print(fid, f.get_key_len())

            if ["msk_seg-subreg", "msk_seg-vert"] not in f:
                continue
            sem_ref = f["msk_seg-subreg"][0]
            vert_ref: BIDS_FILE = f["msk_seg-vert"][0]

            #####
            # outputs
            out_det = vert_ref.get_changed_path(
                file_type="json",
                bids_format="poi",
                parent=der_out,
                info={"source": "deterministic"},
                make_parent=True,
            )
            out_det_global = vert_ref.get_changed_path(
                file_type="mrk.json",
                bids_format="poi",
                parent=der_out,
                info={"space": "global", "source": "deterministic"},
                make_parent=True,
            )
            # out_det = Path(SAVE_PATH) / name / "poi_predicted.json"
            # out_det.parent.mkdir(parents=True, exist_ok=True)
            #####
            if out_det.exists():
                logger.print("Outputs already exist")
                continue

            vert_msk = vert_ref.open_nii()
            sem_msk = sem_ref.open_nii()

            vert_ids = [v for v in vert_msk.unique() if v >= 6]

            try:
                det_poi = calc_poi_from_subreg_vert(
                    vert=vert_msk,
                    subreg=sem_msk,
                    _vert_ids=vert_ids,
                    subreg_id=[
                        # Location.Vertebra_Full,
                        # Location.Arcus_Vertebrae,
                        # Location.Spinosus_Process,
                        # Location.Costal_Process_Left,
                        # Location.Costal_Process_Right,
                        # Location.Superior_Articular_Left,
                        # Location.Superior_Articular_Right,
                        # Location.Inferior_Articular_Left,
                        # Location.Inferior_Articular_Right,
                        # Location.Vertebra_Corpus_border, CT only
                        # Location.Vertebra_Corpus,
                        # Location.Vertebra_Disc,
                        Location.Muscle_Inserts_Spinosus_Process,
                        Location.Muscle_Inserts_Transverse_Process_Left,
                        Location.Muscle_Inserts_Transverse_Process_Right,
                        Location.Muscle_Inserts_Vertebral_Body_Left,
                        Location.Muscle_Inserts_Vertebral_Body_Right,
                        Location.Muscle_Inserts_Articulate_Process_Inferior_Left,
                        Location.Muscle_Inserts_Articulate_Process_Inferior_Right,
                        Location.Muscle_Inserts_Articulate_Process_Superior_Left,
                        Location.Muscle_Inserts_Articulate_Process_Superior_Right,
                        Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median,
                        Location.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Median,
                        Location.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Median,
                        Location.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Median,
                        Location.Additional_Vertebral_Body_Middle_Superior_Median,
                        Location.Additional_Vertebral_Body_Posterior_Central_Median,
                        Location.Additional_Vertebral_Body_Middle_Inferior_Median,
                        Location.Additional_Vertebral_Body_Anterior_Central_Median,
                        Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left,
                        Location.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Left,
                        Location.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Left,
                        Location.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Left,
                        Location.Additional_Vertebral_Body_Middle_Superior_Left,
                        Location.Additional_Vertebral_Body_Posterior_Central_Left,
                        Location.Additional_Vertebral_Body_Middle_Inferior_Left,
                        Location.Additional_Vertebral_Body_Anterior_Central_Left,
                        Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right,
                        Location.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Right,
                        Location.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Right,
                        Location.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Right,
                        Location.Additional_Vertebral_Body_Middle_Superior_Right,
                        Location.Additional_Vertebral_Body_Posterior_Central_Right,
                        Location.Additional_Vertebral_Body_Middle_Inferior_Right,
                        Location.Additional_Vertebral_Body_Anterior_Central_Right,
                        Location.Ligament_Attachment_Point_Flava_Superior_Median,
                        Location.Ligament_Attachment_Point_Flava_Inferior_Median,
                        Location.Vertebra_Direction_Posterior,
                        Location.Vertebra_Direction_Inferior,
                        Location.Vertebra_Direction_Right,
                    ],
                )
            except Exception as e:
                logger.print(f"[SKIP] Error at {name}: {e}", Log_Type.FAIL)
                # raise e
                SKIPPED_SUBJECTS.append(name)
                return  # Direkt zurück → dieses Subjekt wird geskippt
            det_poi = det_poi.round(2)
            det_poi.save(out_det)

            det_poi.to_global().save_mrk(out_det_global, split_by_region=True)

            # det_poi_nii = det_poi.make_point_cloud_nii()[1]
            # det_poi_nii.save(out_det.parent.joinpath("nifty.nii.gz"))
            # break
        # break
    except Exception as e:
        logger.print(f"[SKIP] Error at {name}: {e}", Log_Type.FAIL)
        # raise e
        SKIPPED_SUBJECTS.append(name)


if __name__ == "__main__":
    ds_names = [
        # "dataset-verse19training_1mmiso",
        # "dataset-verse20training_1mmiso",
        # "dataset-verse19validation_1mmiso",
        # "dataset-verse20validation_1mmiso",
        # "dataset-verse19test_1mmiso",
        "dataset-verse20test_1mmiso",
    ]

    for ds_name in ds_names:

        ds_path = f"/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/{ds_name}/"
        bgi = BIDS_Global_info(
            datasets=[ds_path],
            parents=["derivatives_combined"],
        )

        der_out = "derivatives_poi_deterministic"

        Parallel(n_jobs=10, backend="threading")(
            delayed(_proc)(name, subject, der_out) for name, subject in bgi.enumerate_subjects(sort=True)
        )
        # for name, subject in bgi.enumerate_subjects(sort=True):
        # if "601" not in name:
        #    continue
        #    _proc(name, subject, der_out)
        #    break

        if len(SKIPPED_SUBJECTS) > 0:
            print("Skipped subjects:")
            for s in SKIPPED_SUBJECTS:
                print(s)

        # break
