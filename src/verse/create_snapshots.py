import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

from TPTBox import NII, BIDS_FILE, BIDS_Global_info, No_Logger, Log_Type, Location, POI, calc_poi_from_subreg_vert
from TPTBox.spine.snapshot2D import create_snapshot, Snapshot_Frame
from TPTBox.mesh3D.snapshot3D import make_snapshot3D_parallel

# from utils.filepaths import filepath_dataset
import numpy as np
from joblib import Parallel, delayed


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

            if ["ct", "msk_seg-subreg", "msk_seg-vert"] not in f:
                continue
            sem_ref = f["msk_seg-subreg"][0]
            vert_ref: BIDS_FILE = f["msk_seg-vert"][0]
            ct_ref: BIDS_FILE = f["ct"][0]

            #####
            # outputs
            out_snp = vert_ref.get_changed_path(
                file_type="png",
                bids_format="snp",
                parent=der_out,
                make_parent=False,
            )
            out_snp = out_snp.parent.parent.joinpath(out_snp.name)
            out_snp.parent.mkdir(parents=True, exist_ok=True)
            out_snp3d_vert = vert_ref.get_changed_path(
                file_type="png",
                bids_format="snp",
                info={"source": "3D", "seg": "vert"},
                parent=der_out,
                make_parent=False,
            )
            out_snp3d_vert = out_snp3d_vert.parent.parent.joinpath(out_snp3d_vert.name)
            out_snp3d_sem = vert_ref.get_changed_path(
                file_type="png",
                bids_format="snp",
                info={"source": "3D", "seg": "subreg"},
                parent=der_out,
                make_parent=False,
            )
            out_snp3d_sem = out_snp3d_sem.parent.parent.joinpath(out_snp3d_sem.name)
            #####
            if out_snp.exists():
                logger.print("Outputs already exist")
                continue

            vert_msk = vert_ref.open_nii()
            sem_msk = sem_ref.open_nii()
            ct_nii = ct_ref.open_nii()

            poi = calc_poi_from_subreg_vert(vert_msk, sem_msk)

            frames = [
                Snapshot_Frame(
                    image=ct_nii,
                    segmentation=vert_msk,
                    centroids=poi,
                    crop_img=True,
                ),
                Snapshot_Frame(
                    image=ct_nii,
                    segmentation=sem_msk,
                    centroids=poi,
                    crop_img=True,
                ),
            ]

            try:
                create_snapshot(out_snp, frames)

                make_snapshot3D_parallel([vert_msk, sem_msk], output_paths=[out_snp3d_vert, out_snp3d_sem], view=["A", "L"])
            except Exception as e:
                logger.print(f"[SKIP] Error at {name}: {e}", Log_Type.FAIL)
                # raise e
                SKIPPED_SUBJECTS.append(name)
                return  # Direkt zurück → dieses Subjekt wird geskippt
    except Exception as e:
        logger.print(f"[SKIP] Error at {name}: {e}", Log_Type.FAIL)
        # raise e
        SKIPPED_SUBJECTS.append(name)


if __name__ == "__main__":
    ds_names = [
        "dataset-verse19training_1mmiso",
        "dataset-verse20training_1mmiso",
        "dataset-verse19validation_1mmiso",
        "dataset-verse20validation_1mmiso",
        "dataset-verse19test_1mmiso",
        "dataset-verse20test_1mmiso",
    ]

    for ds_name in ds_names:

        ds_path = f"/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/{ds_name}/"
        DER_IN = "derivatives_combined"
        bgi = BIDS_Global_info(
            datasets=[ds_path],
            parents=["rawdata", DER_IN],
        )

        der_out = "snaps_" + DER_IN

        # for name, subject in bids_ds.enumerate_subjects(sort=True):
        Parallel(n_jobs=3, backend="threading")(
            delayed(_proc)(name, subject, der_out) for name, subject in bgi.enumerate_subjects(sort=True)
        )
        # for name, subject in bgi.enumerate_subjects(sort=True):
        #    _proc(name, subject, der_out)
        #    break

        if len(SKIPPED_SUBJECTS) > 0:
            print("Skipped subjects:")
            for s in SKIPPED_SUBJECTS:
                print(s)

        # break
