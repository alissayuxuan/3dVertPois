from TPTBox import NII

sub_ds = "dataset-verse20test_1mmiso"
subject_id = "gl279"


ddir = f"/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/{sub_ds}/derivatives_combined/sub-{subject_id}/"


subreg = NII.load(ddir + f"sub-{subject_id}_dir-ax_seg-subreg_msk.nii.gz", seg=True)
vert = NII.load(ddir + f"sub-{subject_id}_dir-ax_seg-vert_msk.nii.gz", seg=True)

label = 24
subreg_ = vert.extract_label(label) * subreg

subreg_.save(ddir + f"sub-{subject_id}_dir-iso_seg-subreg_label-{label}_msk.nii.gz")
