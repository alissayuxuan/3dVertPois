from pathlib import Path
from shutil import copy
from TPTBox import No_Logger, Log_Type, NII
from joblib import Parallel, delayed

logger = No_Logger(prefix="verse_copy_der")


def _proc(subject, from_ori, to_subject_dir, COPY_FORMATS):
    if not subject.is_dir():
        return

    logger.print("Subject:", subject.name)

    subject_id = subject.name
    from_ori = from_ori_ds if "sub-gl" not in subject_id else gl_from
    from_subject_dir = from_ori.joinpath(subject_id)
    if not from_subject_dir.exists():
        logger.print(f"Source subject dir does not exist: {from_subject_dir}", Log_Type.WARNING)
        return

    to_subject_dir = ds_dir / f"derivatives_subreg"
    to_subject_dir.mkdir(exist_ok=True)
    to_subject_dir = to_subject_dir / subject_id
    #
    to_subject_dir.mkdir(exist_ok=True)
    #
    for file in from_subject_dir.iterdir():
        if file.suffix in COPY_FORMATS and ("seg-subreg_" in file.name or "seg-vert_" in file.name):
            end_p = to_subject_dir / file.name
            if not end_p.exists():
                logger.print(f"Copying file: {file} -> {end_p}")
                NII.load(file, seg=True).rescale_().save(end_p)
            # copy(file, to_subject_dir / file.name)
            # break


if __name__ == "__main__":
    root = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/")

    v19_from = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-verse19/derivatives")
    v20_from = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-verse20/derivatives")
    gl_from = Path("/DATA/NAS/datasets_processed/CT_spine/CT_TRAINING_ORG2/dataset-gl/derivatives/")

    COPY_FORMATS = [".nii.gz", ".gz"]  # [".json", ".csv", ".nii.gz", ".gz"]

    for ds_dir in root.iterdir():
        if not ds_dir.is_dir():
            continue

        # has rawdata folder
        raw_dir = ds_dir.joinpath("rawdata")
        if not raw_dir.exists():
            logger.print(f"No rawdata dir in dataset dir: {ds_dir}", Log_Type.WARNING)
            continue

        logger.print("Processing dataset dir:", ds_dir.name, Log_Type.STAGE)

        is_v19 = "verse19" in ds_dir.name
        from_ori_ds = v19_from if is_v19 else v20_from

        # search through rawdata for subjects and then copy
        subjects = [d for d in raw_dir.iterdir() if d.is_dir() and "sub-" in d.name]
        Parallel(n_jobs=8)(delayed(_proc)(subject, from_ori_ds, ds_dir, COPY_FORMATS) for subject in subjects)
        # break
        # break
