import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

from TPTBox import NII, BIDS_FILE, BIDS_Global_info, No_Logger, Log_Type, Location, POI, calc_poi_from_subreg_vert
#from utils.filepaths import filepath_dataset
import numpy as np
from joblib import Parallel, delayed
from TPTBox.core.bids_constants import sequence_splitting_keys

import os


logger = No_Logger()

# zoom = (0.8, 0.8, 0.8)

der_seg = "derivatives"

DATASET = "dataset/data_preprocessing/dataset-verse19"
SAVE_PATH = "predictions/verse19-deterministic"

SKIPPED_SUBJECTS = []


bids_ds = BIDS_Global_info(
    datasets=[DATASET],#[filepath_dataset()],
    parents=["rawdata", der_seg],
    sequence_splitting_keys=[*sequence_splitting_keys, "seq"],
)


def _proc(name, subject):
    # if "25" not in name:
    #    return
    try:
        q = subject.new_query()
        families = q.loop_dict(key_addendum=["source"])
        for f in families:
            fid = f.family_id

            logger.print(fid, f.get_key_len())

            if ["ct", "msk_seg-subreg", "msk_seg-vert"] not in f:
                continue

            ct_ref = f["ct"][0]
            sem_ref = f["msk_seg-subreg"][0]
            vert_ref = f["msk_seg-vert"][0]

            #####
            # outputs
            #out_det = ct_ref.get_changed_path(file_type="json", bids_format="poi", parent=der_seg, info={"source": "deterministic"})
            #out_det_global = ct_ref.get_changed_path(file_type="json", bids_format="poi", parent=der_seg, info={"source": "global"})
            out_det = Path(SAVE_PATH) / name / "poi_predicted.json"
            out_det.parent.mkdir(parents=True, exist_ok=True)
            #####
            if out_det.exists():
                logger.print("Outputs already exist")
                continue

            ct_nii = ct_ref.open_nii()
            logger.print(ct_nii)
            vert_msk = vert_ref.open_nii()
            sem_msk = sem_ref.open_nii()

            ct_nii.assert_affine(other=vert_msk)
            print("DEBUG:", name)

            try:
                det_poi = calc_poi_from_subreg_vert(
                    vert=vert_msk,
                    subreg=sem_msk,
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
                print(f"[SKIP] Fehler bei {name}: {e}")
                SKIPPED_SUBJECTS.append(name)
                return  # Direkt zurück → dieses Subjekt wird geskippt    
            det_poi = det_poi.round(2)
            det_poi.save(out_det)
            
            #det_poi.to_global().save_mrk(out_det_global)

            # det_poi_nii = det_poi.make_point_cloud_nii()[1]
            # det_poi_nii.save(out_det.parent.joinpath("nifty.nii.gz"))
            # break
        # break
    except Exception as e:
        print(f"[SKIP] Fehler bei {name}: {e}")
        SKIPPED_SUBJECTS.append(name)

# for name, subject in bids_ds.enumerate_subjects(sort=True):
Parallel(n_jobs=6)(delayed(_proc)(name, subject) for name, subject in bids_ds.enumerate_subjects(sort=True))

if len(SKIPPED_SUBJECTS) > 0:
    print("Skipped subjects:")
    for s in SKIPPED_SUBJECTS:
        print(s)