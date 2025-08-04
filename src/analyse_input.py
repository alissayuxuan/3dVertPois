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
from TPTBox import NII, BIDS_Global_info
from TPTBox.core.poi import POI

import pandas as pd
import ast


base_dir = "dataset/data_preprocessing/cutout-folder/cutouts_exclude"
subreg_filename = "subreg.nii.gz"  
vertseg_filename = "vertseg.nii.gz"  

def get_subreg(container):
    subreg_query = container.new_query(flatten=True)
    subreg_query.filter_format("msk")
    subreg_query.filter_filetype("nii.gz")  # only nifti files
    subreg_query.filter("seg", "subreg")
    subreg_candidate = subreg_query.candidates[0]
    return str(subreg_candidate.file["nii.gz"])

def get_vertseg(container):
    vertseg_query = container.new_query(flatten=True)
    vertseg_query.filter_format("msk")
    vertseg_query.filter_filetype("nii.gz")  # only nifti files
    vertseg_query.filter("seg", "vert")
    vertseg_candidate = vertseg_query.candidates[0]
    return str(vertseg_candidate.file["nii.gz"])

def get_poi(container):
    poi_query = container.new_query(flatten=True)
    poi_query.filter_format("poi")    
    poi_candidate = poi_query.candidates[0]
    return str(poi_candidate.file["json"])

def is_affine_normal(affine, atol=1e-3):
    core = affine[:3, :3]
    expected = np.diag(np.sign(np.diag(core)))  # e.g. [-1, 1, 1]
    return np.allclose(core, expected, atol=atol) and np.allclose(affine[3], [0, 0, 0, 1])

def analyse_affine():
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

def analyse_orientation():
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
                vertseg = NII.load(verseg_path, seg=True)
                subreg = NII.load(subreg_path, seg=True)

                if vertseg.orientation != subreg.orientation:
                    print(f"❌ Subjekt: {subject} vertseg.orientation ({vertseg.orientation})!= subreg.orientation ({subreg.orientation}) ")
                else:
                    print(f"Subjekt {subject}: vertseg: {vertseg.orientation}, subreg: {subreg.orientation}")


            except Exception as e:
                print(f"⚠️ Fehler bei {subject} Wirbel {vertebra_id}: {e}")

def analyse_original_orientation():
    bgi = BIDS_Global_info(
        datasets=["/home/student/alissa/3dVertPois/src/dataset/data_preprocessing/dataset-folder"],
        parents=["derivatives"],
    )

    for sub, container in bgi.enumerate_subjects():
        vert_msk_path = get_vertseg(container)
        subreg_msk_path = get_subreg(container)
        gt_poi_path = get_poi(container)

        vert_msk = NII.load(vert_msk_path, seg=True)
        subreg_msk = NII.load(subreg_msk_path, seg=True)
        gt_poi = POI.load(gt_poi_path)

        print(f"Subjekt {sub}: vertseg: {vert_msk.orientation}, subreg: {subreg_msk.orientation}")  

def analyse_cutout_orientation():
    base_dir = "dataset/data_preprocessing/cutout-folder/cutouts_exclude"

    for subject in sorted(os.listdir(base_dir)):
        subject_path = os.path.join(base_dir, subject)
        if not os.path.isdir(subject_path):
            continue
        
        for vertebra_id in range(1, 25):
            vertebra_path = os.path.join(subject_path, str(vertebra_id))
            subreg_path = os.path.join(vertebra_path, subreg_filename)
            verseg_path = os.path.join(vertebra_path, vertseg_filename)
            

def add_outliers_to_master_df(master_df_path, outlier_paths, save_path=None):
    """
    Fügt alle POI-Outlier aus den gegebenen CSV-Dateien zur bad_poi_list in master_df hinzu.
    
    Args:
        master_df_path (str): Pfad zur master_df CSV-Datei.
        outlier_paths (list[str]): Liste von Pfaden zu Outlier-CSV-Dateien.
        save_path (str, optional): Wohin die aktualisierte master_df gespeichert werden soll. 
                                   Falls None, wird nichts gespeichert, nur zurückgegeben.
        
    Returns:
        pd.DataFrame: Die aktualisierte master_df.
    """
    # Lade master_df
    master_df = pd.read_csv(master_df_path)

    # Wandle alle bad_poi_list-Einträge von Strings in echte Listen um
    master_df["bad_poi_list"] = master_df["bad_poi_list"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

    # Verarbeite jede Outlier-Datei
    for path in outlier_paths:
        outlier_df = pd.read_csv(path)

        for _, row in outlier_df.iterrows():
            subj = row["subject"]
            vert = row["vertebra"]
            poi = int(row["poi_idx"])

            # Finde den passenden Eintrag in master_df
            mask = (master_df["subject"] == subj) & (master_df["vertebra"] == vert)
            if not mask.any():
                print(f"⚠️ Kein Match gefunden für Subject {subj}, Vertebra {vert}")
                continue
            """
            # Füge POI hinzu, wenn noch nicht vorhanden
            current_list = master_df.loc[mask, "bad_poi_list"].values[0]
            print(f"Verarbeite Subject {subj}, Vertebra {vert}, POI {poi}: {current_list}")
            if not isinstance(current_list, list):
                current_list = ast.literal_eval(current_list) if isinstance(current_list, str) else [current_list]

            if poi not in current_list:
                updated_list = current_list + [poi]
                master_df.loc[mask, "bad_poi_list"] = [updated_list]
            """
            print(f"Verarbeite Subject {subj}, Vertebra {vert}, POI {poi}:",
                master_df.loc[mask, "bad_poi_list"].values)

            # Wende die Aktualisierung direkt auf alle zutreffenden Zeilen an
            master_df.loc[mask, "bad_poi_list"] = master_df.loc[mask, "bad_poi_list"].apply(
                lambda l: l + [poi] if poi not in l else l
            )

    # Optional speichern
    if save_path:
        master_df.to_csv(save_path, index=False)
        print(f"✅ Aktualisierte master_df gespeichert unter: {save_path}")

    return master_df


if __name__ == "__main__":
    #analyse_original_orientation()
    master_df_path = "dataset/data_preprocessing/cutout-folder/cutouts-bad_poi/master_df.csv"
    outlier_paths = [
        "experiments/experiment_evaluation/k_fold/fold_1/test/outliers_error_higher_10.csv",
        "experiments/experiment_evaluation/k_fold/fold_2/test/outliers_error_higher_10.csv",
        "experiments/experiment_evaluation/k_fold/fold_3/test/outliers_error_higher_10.csv",
        "experiments/experiment_evaluation/k_fold/fold_4/test/outliers_error_higher_10.csv",
        "experiments/experiment_evaluation/k_fold/fold_5/test/outliers_error_higher_10.csv",
        "experiments/experiment_evaluation/k_fold/fold_6/test/outliers_error_higher_10.csv"
        ]
    save_path = "dataset/data_preprocessing/cutout-folder/cutouts-bad_poi/updated_master_df.csv"

    updated_master_df = add_outliers_to_master_df(master_df_path, outlier_paths, save_path)
    print(updated_master_df.head())