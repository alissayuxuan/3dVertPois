import os
import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))
from shutil import copy
from TPTBox import No_Logger, Log_Type, NII
from utils.filepaths import search_path


logger = No_Logger(prefix="verse_copy_der")

if __name__ == "__main__":
    root = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/")

    v19_from = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-verse19/derivatives")
    v20_from = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-verse20/derivatives")
    gl_from = Path("/DATA/NAS/datasets_processed/CT_spine/CT_TRAINING_ORG2/dataset-gl/derivatives/")

    iso_suffix = "_1mmiso"

    for ds_dir in root.iterdir():
        ds_name = ds_dir.name
        if not ds_dir.is_dir():
            continue

        if iso_suffix in ds_name:
            continue

        ds_out_name = f"{ds_name}{iso_suffix}"

        for par in ["rawdata", "derivatives"]:
            par_dir = ds_dir.joinpath(par)
            assert par_dir.exists(), f"No {par} dir in dataset dir: {ds_dir}"

            x = 0
            niftis_p = search_path(par_dir, f"**/*.nii.gz", suppress=True)
            for nifti_path in niftis_p:
                out_p = str(nifti_path).replace(ds_name, ds_out_name)
                out_p = Path(out_p)
                if out_p.exists():
                    logger.print(f"Exists: {out_p}")
                    continue
                out_p.parent.mkdir(parents=True, exist_ok=True)
                logger.print(f"Processing: {nifti_path} -> {out_p}")
                nii = NII.load(nifti_path, seg="_seg-" in nifti_path.name).rescale_()
                nii.save(out_p)
