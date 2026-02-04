import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))
from pathlib import Path
from TPTBox import No_Logger, Log_Type, NII, BIDS_FILE
from joblib import Parallel, delayed
from panoptica import Metric
from utils.filepaths import search_path_single, search_path
import shutil
from spineps.phase_post import assign_missing_cc

logger = No_Logger(prefix="verse_cseg")

ROOT = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/")
RAWDATA = "rawdata"
DERIV_ORIGINAL_SUBREG = "segmentation_origins/derivatives_subreg"  # "derivatives_subreg"
DERIV_SUBREG_NEW = "segmentation_origins/derivatives-template_padd"
DERIV_ORIGINAL = "derivatives"
DERIV_C_ONE = "segmentation_origins/derivatives-template_padd"

DERIV_OUT = "TEST_derivatives_combined"


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

        vert_out = img_bidsf.get_changed_path(
            file_type="nii.gz", bids_format="msk", info={"seg": "vert"}, parent=DERIV_OUT, make_parent=True
        )
        subreg_out = img_bidsf.get_changed_path(
            file_type="nii.gz", bids_format="msk", info={"seg": "subreg"}, parent=DERIV_OUT, make_parent=True
        )
        if vert_out.exists() and subreg_out.exists():
            continue

        vert_path = img_bidsf.get_changed_path(file_type="nii.gz", bids_format="msk", info={"seg": "vert"}, parent=DERIV_ORIGINAL)
        # VERT MASK IS NOT MODIFIED, so just copy
        assert vert_path.exists(), f"{subject_ct_id}: Original vert mask does not exist: {vert_path}"

        subreg_ori_path = img_bidsf.get_changed_path(
            file_type="nii.gz", bids_format="msk", info={"seg": "subreg"}, parent=DERIV_ORIGINAL_SUBREG
        )
        if not subreg_ori_path.exists():
            subreg_ori_path = img_bidsf.get_changed_path(
                file_type="nii.gz",
                bids_format="notFORtraining",
                info={"seg": "subreg"},
                parent=DERIV_ORIGINAL_SUBREG,
                non_strict_mode=True,
            )
        subreg_new_path = img_bidsf.get_changed_path(file_type="nii.gz", bids_format="msk", info={"seg": "subreg"}, parent=DERIV_SUBREG_NEW)

        if not subreg_ori_path.exists() and not subreg_new_path.exists():
            logger.print(f"{subject_ct_id}: NO subreg mask exists: {subreg_ori_path}", Log_Type.FAIL)
            continue

        # always use new subregion mask if exists
        subreg_path = subreg_new_path if not subreg_ori_path.exists() else subreg_ori_path

        take_new_for_vert = []

        img_nii = NII.load(img_path, seg=False)

        # load both masks
        subreg_nii = NII.load(subreg_path, seg=True).resample_from_to_(img_nii, verbose=False)
        subreg_nii.map_labels_({51: 49, 50: 49}, verbose=False)
        vert_nii = NII.load(vert_path, seg=True).resample_from_to_(img_nii, verbose=False)
        vert_arr = vert_nii.get_seg_array()
        # subreg_nii.resample_from_to_(vert_nii)
        subreg_nii.assert_affine(other=vert_nii, verbose=logger, raise_error=True)

        if len(take_new_for_vert) > 0:
            subreg_n_nii = NII.load(subreg_new_path, seg=True).resample_from_to_(img_nii, verbose=False)
            subreg_n_nii.map_labels_({51: 49, 50: 49}, verbose=False)
            subreg_n_nii.assert_affine(other=vert_nii, verbose=logger, raise_error=True)
            subreg_n_arr = subreg_n_nii.get_seg_array()
            subreg_n_arr = vert_nii.extract_label(take_new_for_vert) * subreg_n_arr
            subreg_nii[subreg_n_arr != 0] = subreg_n_arr[subreg_n_arr != 0]

        # then add anything in vert mask to nearest subreg region
        subreg_arr_cleaned, _, deletion_map = assign_missing_cc(
            target_arr=subreg_nii.get_seg_array(),
            reference_arr=vert_arr.copy(),
            proc_assign_missing_dilate_first=True,
        )
        # delete everything in subreg that does not exist in vert
        subreg_arr_cleaned[vert_arr == 0] = 0
        # delete everything in vert that does not exist in subreg
        vert_arr[subreg_arr_cleaned == 0] = 0

        vert_nii.set_array_(vert_arr)
        subreg_nii.set_array_(subreg_arr_cleaned)

        # Replace C1 with the seg-c1 mask
        segcone_path = img_bidsf.get_changed_path(
            file_type="nii.gz", bids_format="msk", info={"seg": "c1", "mod": "ct"}, parent=DERIV_C_ONE
        )
        if segcone_path.exists():
            segcone_nii = NII.load(segcone_path, seg=True)
            segcone_nii.assert_affine(other=vert_nii, verbose=logger, raise_error=True)
            segcone_arr = segcone_nii.get_seg_array()
            # set all C1 regions in subreg to label 1
            subreg_arr_cleaned[segcone_arr > 0] = segcone_arr[segcone_arr > 0]
            subreg_nii.set_array_(subreg_arr_cleaned)
        else:
            logger.print(f"{subject_ct_id}: No C1 seg found at {segcone_path}, skipping C1 replacement", Log_Type.UNDERLINE)

        # Save new subreg mask
        dsc = Metric.DSC(vert_arr > 0, subreg_arr_cleaned > 0)
        if not dsc >= 0.99:
            logger.print(f"{subject_ct_id}: After combining, Mask mismatch {dsc} between vert and subreg\n", Log_Type.FAIL)
        else:
            subreg_nii.save(subreg_out)
            vert_nii.save(vert_out)


if __name__ == "__main__":
    for ds_dir in ROOT.iterdir():
        if not ds_dir.is_dir():
            continue
        # filter
        if not ds_dir.name.startswith("dataset-"):
            continue
        # if not ds_dir.name == "dataset-verse20test_1mmiso":
        #    continue
        if not ds_dir.name == "dataset-verse19training_1mmiso":
            continue

        # has rawdata folder
        raw_dir = ds_dir.joinpath(RAWDATA)
        assert raw_dir.exists(), f"No rawdata dir in dataset dir: {ds_dir}"

        logger.print("Processing dataset dir:", ds_dir.name, Log_Type.STAGE)

        subjects = list(raw_dir.iterdir())

        Parallel(n_jobs=10, backend="threading")(delayed(_proc)(subject, ds_dir) for subject in subjects)
        # for subject in subjects:
        # if "e766" not in subject.name:
        #    continue
        # _proc(subject, ds_dir)
        # break

        # break
