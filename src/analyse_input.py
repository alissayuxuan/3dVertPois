"""
import os
import numpy as np
from TPTBox import NII
from TPTBox.core.poi import POI

subreg_path_WS05 = "dataset/data_preprocessing/cutout-folder/cutouts_exclude/WS-05/22/subreg.nii.gz"
vertseg_path_WS05 = "dataset/data_preprocessing/cutout-folder/cutouts_exclude/WS-05/22/vertseg.nii.gz"

subreg_path_WS06 = "dataset/data_preprocessing/cutout-folder/cutouts_exclude/WS-06/22/subreg.nii.gz"
vertseg_path_WS06 = "dataset/data_preprocessing/cutout-folder/cutouts_exclude/WS-06/22/vertseg.nii.gz"

subreg_WS05 = NII.load(subreg_path_WS05, seg=True)
vertseg_WS05 = NII.load(vertseg_path_WS05, seg=True)
subreg_WS06 = NII.load(subreg_path_WS06, seg=True)
vertseg_WS06 = NII.load(vertseg_path_WS06, seg=True)

print("Subregion WS-05 affine:\n", subreg_WS05.affine)
print("Vertseg WS-05 affine:\n", vertseg_WS05.affine)
print("Subregion WS-06 affine:\n", subreg_WS06.affine)
print("Vertseg WS-06 affine:\n", vertseg_WS06.affine)
"""
"""
subreg_path_WS05 = "dataset/data_preprocessing/dataset-folder/derivatives/WS-05/ses-20221109/sub-WS-05_ses-20221109_seq-1_seg-subreg_msk.nii.gz"
subreg_path_WS06 = "dataset/data_preprocessing/dataset-folder/derivatives/WS-06/ses-20221109/sub-WS-06_ses-20221109_seq-1_seg-subreg_msk.nii.gz"
subreg_path_WS07 = "dataset/data_preprocessing/dataset-folder/derivatives/WS-07/ses-20221109/sub-WS-07_ses-20221109_seq-1_seg-subreg_msk.nii.gz"
subreg_path_WS08 = "dataset/data_preprocessing/dataset-folder/derivatives/WS-08/ses-20221109/sub-WS-08_ses-20221109_seq-1_seg-subreg_msk.nii.gz"
subreg_path_WS09 = "dataset/data_preprocessing/dataset-folder/derivatives/WS-09/ses-20221109/sub-WS-09_ses-20221109_seq-1_seg-subreg_msk.nii.gz"
subreg_path_WS15 = "dataset/data_preprocessing/dataset-folder/derivatives/WS-15/ses-20221111/sub-WS-15_ses-20221111_seq-1_seg-subreg_msk.nii.gz"
subreg_path_WS17 = "dataset/data_preprocessing/dataset-folder/derivatives/WS-17/ses-20220912/sub-WS-17_ses-20220912_seq-1_seg-subreg_msk.nii.gz"
subreg_path_WS18 = "dataset/data_preprocessing/dataset-folder/derivatives/WS-18/ses-20221114/sub-WS-18_ses-20221114_seq-1_seg-subreg_msk.nii.gz"


subreg_WS05 = NII.load(subreg_path_WS05, seg=True)
subreg_WS06 = NII.load(subreg_path_WS06, seg=True)

subreg_WS05 = subreg_WS05.resample_from_to(subreg_WS06)

subreg_WS07 = NII.load(subreg_path_WS07, seg=True)
subreg_WS08 = NII.load(subreg_path_WS08, seg=True)
subreg_WS09 = NII.load(subreg_path_WS09, seg=True)
subreg_WS15 = NII.load(subreg_path_WS15, seg=True)
subreg_WS17 = NII.load(subreg_path_WS17, seg=True)
subreg_WS18 = NII.load(subreg_path_WS18, seg=True)



print("Subregion WS-05 affine:\n", subreg_WS05.affine)
print("Subregion WS-06 affine:\n", subreg_WS06.affine)
print("Subregion WS-07 affine:\n", subreg_WS07.affine)
print("Subregion WS-08 affine:\n", subreg_WS08.affine)
print("Subregion WS-09 affine:\n", subreg_WS09.affine)
print("Subregion WS-15 affine:\n", subreg_WS15.affine)
print("Subregion WS-17 affine:\n", subreg_WS17.affine)
print("Subregion WS-18 affine:\n", subreg_WS18.affine)
"""


import os
import numpy as np
from TPTBox import NII
from TPTBox.core.poi import POI


base_dir = "dataset/data_preprocessing/cutout-folder/cutouts_exclude"
subreg_filename = "subreg.nii.gz"  
vertseg_filename = "vertseg.nii.gz"  

def is_affine_normal(affine, atol=1e-3):
    core = affine[:3, :3]
    expected = np.diag(np.sign(np.diag(core)))  # e.g. [-1, 1, 1]
    return np.allclose(core, expected, atol=atol) and np.allclose(affine[3], [0, 0, 0, 1])

abnormal_affines = []

for subject in sorted(os.listdir(base_dir)):
    subject_path = os.path.join(base_dir, subject)
    if not os.path.isdir(subject_path):
        continue

    for vertebra_id in range(1, 25):
        vertebra_path = os.path.join(subject_path, str(vertebra_id))
        subreg_path = os.path.join(vertebra_path, subreg_filename)
        verseg_path = os.path.join(vertebra_path, vertseg_filename)


        if not os.path.exists(subreg_path) or not os.path.exists(verseg_path):
            continue

        try:
            """subreg = NII.load(subreg_path, seg=True)
            vertseg = NII.load(verseg_path, seg=True)

            # Vergleich: Haben subreg und vertseg dieselbe affine Matrix?
            match = subreg.assert_affine(vertseg, raise_error=False, verbose=True)

            if not match:
                print(f"❌ Affine/Metadata stimmt nicht überein bei: \nsubreg: {subreg.affine}, \nvertseg: {vertseg.affine}")
            
            """
            img = NII.load(verseg_path, seg=True)

            affine = img.affine
            if not is_affine_normal(affine):
                abnormal_affines.append({
                    "subject": subject,
                    "vertebra": vertebra_id,
                    "affine": affine
                })
        except Exception as e:
            print(f"⚠️ Fehler bei {subject} Wirbel {vertebra_id}: {e}")

# Ergebnis anzeigen
print("\n📋 Abnormale Affine-Matrizen:")
for entry in abnormal_affines:
    print(f"\n🧠 Subjekt: {entry['subject']}  |  Wirbel: {entry['vertebra']:02d}")
    print(entry['affine'])
