import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))
from pathlib import Path
from TPTBox import No_Logger, Log_Type, NII, BIDS_FILE
from joblib import Parallel, delayed
from panoptica import Metric
from known_missing_labels import KNOWN_MISSING_LABELS
from utils.filepaths import search_path_single, search_path

logger = No_Logger(prefix="verse_sanity")

FROM_LABEL_ON = 6

CHECK_AFFINE = True
#
CHECK_ZOOM = True
CHECK_VERT_LABELS = True
#
CHECK_SUBREG_EXISTANCE = True
CHECK_SUBREG_LABELS = True
CHECK_BINARY_DSC = True
#
CHECK_BINARY_DSC_TO_ORIGINAL = True
#
CHECK_POI_EXISTANCE = True
CHECK_POI_AFFINE = True


ROOT = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/")
RAWDATA = "rawdata"
DERIV_SUBREG = "derivatives_combined"  # "derivatives_subreg"
DERIV_VERT = "derivatives_combined"
DERIV_ORIGINAL = "derivatives"
DERIV_POI = "derivatives_poi_deterministic"

DERIV_DEBUG = "derivatives_sanity_checks"


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
        subreg_path = img_bidsf.get_changed_path(file_type="nii.gz", bids_format="msk", info={"seg": "subreg"}, parent=DERIV_SUBREG)

        # vert_path = search_path_single(ds_dir / DERIV_SUBREG / subject_id, f"{subject_id}*_seg-vert*_msk.nii.gz")
        # subreg_path = search_path_single(ds_dir / DERIV_SUBREG / subject_id, f"{subject_id}*_seg-subreg*_msk.nii.gz")

        if not vert_path.exists():
            logger.print(f"{subject_ct_id}: Vert mask does not exist {vert_path}\n", Log_Type.FAIL)
            continue

        if CHECK_AFFINE:
            img_nii = NII.load(img_path, seg=False)
            vert_nii = NII.load(vert_path, seg=True)
            # check affine
            vert_nii.assert_affine(other=img_nii, verbose=logger, raise_error=False, text=f"{subject_ct_id}")
            vert_label_c = [v for v in vert_nii.unique() if v >= FROM_LABEL_ON]

            if CHECK_VERT_LABELS:
                vert_labels = vert_nii.unique()
                if not all([(l in range(1, 27)) or (l == 28) for l in vert_labels]):
                    logger.print(f"{subject_ct_id}: Unexpected vert labels found: {vert_labels}\n", Log_Type.FAIL)
                if len(vert_labels) <= 3:
                    logger.print(f"{subject_ct_id}: Very few vert labels found: {vert_labels}\n", Log_Type.FAIL)

            if CHECK_ZOOM:
                # check zooms
                if not sum(img_nii.zoom) == 3:
                    logger.print(f"{subject_ct_id}: Image zooms are not isotropic: {img_nii.zoom}\n", Log_Type.FAIL)
                if not img_nii.zoom == vert_nii.zoom:
                    logger.print(
                        f"{subject_ct_id}: Vert mask zooms do not match image zooms: {vert_nii.zoom} vs {img_nii.zoom}\n", Log_Type.FAIL
                    )

        if CHECK_SUBREG_EXISTANCE:
            if not subreg_path.exists():
                logger.print(f"{subject_ct_id}: Subreg mask does not exist {subreg_path}\n", Log_Type.FAIL)
                continue

            if CHECK_AFFINE:
                subreg_nii = NII.load(subreg_path, seg=True)
                # check affine
                vert_nii.assert_affine(other=subreg_nii, verbose=logger, raise_error=False, text=f"{subject_ct_id}")

                if CHECK_SUBREG_LABELS:
                    for v in vert_label_c:
                        s_labels = subreg_nii * vert_nii.extract_label(v)
                        s_labels = s_labels.volumes()
                        s_labels_exist = [sl for sl, sv in s_labels.items() if sv >= 50]
                        missing_label = []
                        for sl in [41, 42, 49]:
                            known_issue = (
                                True
                                if subject_ct_id in KNOWN_MISSING_LABELS
                                and v in KNOWN_MISSING_LABELS[subject_ct_id]
                                and sl in KNOWN_MISSING_LABELS[subject_ct_id][v]
                                else False
                            )
                            if sl not in s_labels_exist and not known_issue:
                                missing_label.append(sl)
                        if len(missing_label) > 0:
                            logger.print(f"{subject_ct_id}: Subreg labels {missing_label} missing for vert {v}\n", Log_Type.FAIL)
                        if 9 - len(s_labels_exist) >= 3:
                            logger.print(f"{subject_ct_id}: Very few subreg labels found for vert {v}: {s_labels_exist}\n", Log_Type.FAIL)

                if CHECK_BINARY_DSC:
                    # check that subreg matches vert perfectly
                    vert_arr = vert_nii.get_seg_array()
                    subreg_arr = subreg_nii.get_seg_array()
                    dsc = Metric.DSC(vert_arr > 0, subreg_arr > 0)
                    if not dsc == 1.0:
                        logger.print(f"{subject_ct_id}: Mask mismatch vert and subreg {dsc}\n", Log_Type.FAIL)
                        vert_nii.clamp(0, 1).get_segmentation_difference_to(
                            subreg_nii.clamp(0, 1),
                            ignore_background_tp=True,
                        ).save(
                            img_bidsf.get_changed_path(
                                file_type="nii.gz",
                                bids_format="msk",
                                info={"seg": "vert_subreg_dsc-mismatch"},
                                parent=DERIV_DEBUG,
                                make_parent=True,
                            )
                        )

        if CHECK_BINARY_DSC_TO_ORIGINAL:
            vert_orig_path = img_bidsf.get_changed_path(file_type="nii.gz", bids_format="msk", info={"seg": "vert"}, parent=DERIV_ORIGINAL)
            assert vert_orig_path.exists(), f"Original vert mask does not exist: {vert_orig_path}"
            vert_nii = NII.load(vert_path, seg=True)
            vert_arr = vert_nii.get_seg_array()
            vert_orig_nii = NII.load(vert_orig_path, seg=True)

            if not vert_orig_nii.assert_affine(
                other=vert_nii, verbose=logger, raise_error=False, text=f"{subject_ct_id} vert vs vert original"
            ):
                vert_orig_nii.resample_from_to_(vert_nii, verbose=False)
            vert_orig_arr = vert_orig_nii.get_seg_array()
            dsc = Metric.DSC(vert_arr > 0, vert_orig_arr > 0)
            if not dsc == 1.0:
                logger.print(f"{subject_ct_id}: Mask mismatch vert and vert original {dsc}\n", Log_Type.FAIL)
                vert_nii.clamp(0, 1).get_segmentation_difference_to(
                    vert_orig_nii.clamp(0, 1),
                    ignore_background_tp=True,
                ).save(
                    img_bidsf.get_changed_path(
                        file_type="nii.gz",
                        bids_format="msk",
                        info={"seg": "vert_vert_orig_dsc-mismatch"},
                        parent=DERIV_DEBUG,
                        make_parent=True,
                    )
                )


if __name__ == "__main__":
    for ds_dir in ROOT.iterdir():
        if not ds_dir.is_dir():
            continue
        if not ds_dir.name.startswith("dataset-"):
            continue
        # if not ds_dir.name == "dataset-verse20test_1mmiso":
        #    continue

        # has rawdata folder
        raw_dir = ds_dir.joinpath(RAWDATA)
        assert raw_dir.exists(), f"No rawdata dir in dataset dir: {ds_dir}"

        subjects = list(raw_dir.iterdir())
        logger.print("Processing dataset dir:", ds_dir.name, "n_subjects=", len(subjects), Log_Type.STAGE)

        Parallel(n_jobs=10)(delayed(_proc)(subject, ds_dir) for subject in subjects)
        # for subject in subjects:
        # if "gl195" not in subject.name:
        #    continue
        #    _proc(subject, ds_dir)
        # break
