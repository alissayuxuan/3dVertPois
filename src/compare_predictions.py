import os
import pandas as pd
import numpy as np
import json
from TPTBox.core.poi import POI

import itertools 
from functools import reduce

  

def distance(a, b):
    return np.linalg.norm(a - b)

def get_poi_metadata(poi_file):
    """"returns metadata of the POI file as a dictionary"""

    meta_data = {
        "orientation": poi_file.orientation,
        "zoom": poi_file.zoom,
        "shape": poi_file.shape,
        "origin": poi_file.origin,
        "rotation": poi_file.rotation,
    }
    return meta_data

def compare_meta_data(meta_a, meta_b, tolerance=1e-6, shape_tolerance=1):
    """Compares two metadata dictionaries with numerical tolerance for floating point values and returns differences."""
    diffs = {}
    for key in meta_a.keys():
        val_a = meta_a[key]
        val_b = meta_b[key]
        
        # Special handling for shape: allow 1 voxel difference
        if key == "shape":
            if isinstance(val_a, (list, tuple, np.ndarray)) and isinstance(val_b, (list, tuple, np.ndarray)):
                if not np.allclose(val_a, val_b, atol=shape_tolerance, rtol=0):
                    diffs[key] = (val_a, val_b)
            continue

        # Handle numpy arrays
        if isinstance(val_a, np.ndarray) or isinstance(val_b, np.ndarray):
            # Check if array contains numeric values
            if np.issubdtype(val_a.dtype, np.number) and np.issubdtype(val_b.dtype, np.number):
                if not np.allclose(val_a, val_b, atol=tolerance, rtol=0):
                    diffs[key] = (val_a, val_b)
            else:
                # Non-numeric arrays: exact comparison
                if not np.array_equal(val_a, val_b):
                    diffs[key] = (val_a, val_b)
        
        # Handle tuples/lists
        elif isinstance(val_a, (tuple, list)) and isinstance(val_b, (tuple, list)):
            # Check if all elements are numeric
            if all(isinstance(x, (int, float, np.number)) for x in val_a) and \
               all(isinstance(x, (int, float, np.number)) for x in val_b):
                if not np.allclose(val_a, val_b, atol=tolerance, rtol=0):
                    diffs[key] = (val_a, val_b)
            else:
                # Non-numeric tuples/lists: exact comparison
                if val_a != val_b:
                    diffs[key] = (val_a, val_b)

        # Handle numeric scalars
        elif isinstance(val_a, (int, float, np.number)) and \
             isinstance(val_b, (int, float, np.number)):
            if not np.isclose(val_a, val_b, atol=tolerance, rtol=0):
                diffs[key] = (val_a, val_b)
        
        # Handle non-numeric values (strings, etc.)
        else:
            if val_a != val_b:
                diffs[key] = (val_a, val_b)
    
    return diffs


def create_poi_df(data_path, method_name, load_full_spine=True):
    """
    Returns:
    --------
    poi_df : pd.DataFrame
        DataFrame with columns: subject, vertebra, poi_idx, coords_{method_name}
    meta_df : pd.DataFrame
        DataFrame with columns: subject, meta_{method_name}
    """

    poi_dict = {
        "subject": [],
        "vertebra": [],
        "poi_idx": [],
        f"coords_{method_name}": []
    }

    meta_dict = {
        "subject": [],
        f"meta_{method_name}": []
    }

    if load_full_spine:
        subjects_list = [
            name for name in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, name))
        ]

        for subject in subjects_list:
            poi_path = os.path.join(data_path, subject, "poi_predicted.json")

            if not os.path.exists(poi_path):
                print(f"POI file not found for subject {subject} at {poi_path}, skipping.")
                continue

            poi = POI.load(poi_path)

            # save metadata
            meta = get_poi_metadata(poi)
            meta_dict["subject"].append(subject)
            meta_dict[f"meta_{method_name}"].append(meta)

            # save POI coordinates
            for vert, poi_id, coords in poi.items(): 
                poi_dict["subject"].append(subject)
                poi_dict["vertebra"].append(vert)
                poi_dict["poi_idx"].append(poi_id)
                poi_dict[f"coords_{method_name}"].append(np.array(coords))
    else:
        # New logic: individual POI files per vertebra
        # Get all JSON files matching pattern: <subject>_<vertebra>_pred.json
        poi_files = [
            f for f in os.listdir(data_path)
            if f.endswith("_pred.json")
        ]

        # Group files by subject to save metadata once per subject
        subject_files = {}
        for poi_file in poi_files:
            # Parse filename: <subject>_<vertebra>_pred.json
            parts = poi_file.replace("_pred.json", "").split("_")
            if len(parts) >= 2:
                vertebra = parts[-1]
                subject = "_".join(parts[:-1])
                
                if subject not in subject_files:
                    subject_files[subject] = []
                subject_files[subject].append((poi_file, vertebra))

        # Process each subject
        for subject, files in subject_files.items():
            metadata_saved = False
            
            for poi_file, vertebra in files:
                poi_path = os.path.join(data_path, poi_file)

                if not os.path.exists(poi_path):
                    print(f"POI file not found at {poi_path}, skipping.")
                    continue

                poi = POI.load(poi_path)

                # Save metadata once per subject (from first file)
                if not metadata_saved:
                    meta = get_poi_metadata(poi)
                    meta_dict["subject"].append(subject)
                    meta_dict[f"meta_{method_name}"].append(meta)
                    metadata_saved = True

                # Save POI coordinates
                for vert, poi_id, coords in poi.items():
                    poi_dict["subject"].append(subject)
                    poi_dict["vertebra"].append(vert)
                    poi_dict["poi_idx"].append(poi_id)
                    poi_dict[f"coords_{method_name}"].append(np.array(coords))


    poi_df = pd.DataFrame(poi_dict)
    meta_df = pd.DataFrame(meta_dict)
    return poi_df, meta_df

def join_poi_dfs(path_method_list, load_full_spine=True):

    df_list = []
    meta_df_list = []
    method_names = [method_name for (_, method_name) in path_method_list]

    for (data_path, method_name) in path_method_list:
        poi_df, meta_df = create_poi_df(data_path, method_name, load_full_spine=load_full_spine)
        
        df_list.append(poi_df)
        meta_df_list.append(meta_df)

    # merge metadata dataframes
    meta_merged = reduce(
        lambda left, right: left.merge(right, on="subject", how="outer"),
        meta_df_list
    )

    # compare metadata between methods for each subject
    incompatible_subjects = []
    incompatible_subjects_info = {}
    
    print("\n=== Metadaten-Vergleich ===")
    
    for _, row in meta_merged.iterrows():
        subject = row["subject"]
        
        # Verwende erste Methode als Referenz
        ref_method = method_names[0]
        ref_meta = row[f"meta_{ref_method}"]
        
        if pd.isna(ref_meta):
            print(f"\nSubject {subject}: no meta data available for {ref_method}")
            incompatible_subjects.append(subject)
            incompatible_subjects_info[subject] = {
                "reason": f"Missing metadata for {ref_method}",
                "differences": {}
            }
            continue
        
        subject_compatible = True
        all_differences = {}
        
        # compare meta data with other methods
        for compare_method in method_names[1:]:
            compare_meta = row[f"meta_{compare_method}"]
            
            if pd.isna(compare_meta):
                print(f"  {compare_method}: Missing metadata")
                subject_compatible = False
                all_differences[compare_method] = "Missing metadata"
                continue
            
            diffs = compare_meta_data(ref_meta, compare_meta)
            
            if len(diffs) != 0:
                for key, (val_ref, val_compare) in diffs.items():
                    print(f"    - {key}: {val_ref} vs {val_compare}")
                subject_compatible = False
                all_differences[compare_method] = diffs
        
        if not subject_compatible:
            incompatible_subjects.append(subject)
            incompatible_subjects_info[subject] = {
                "reference_method": ref_method,
                "reference_metadata": ref_meta,
                "differences": all_differences
            }


    df_merged = reduce(
        lambda left, right: left.merge(
            right, on=["subject", "vertebra", "poi_idx"], how="outer"
        ),
        df_list
    )

    df_merged = df_merged.dropna()

    # remove incompatible subjects from dataframe
    if incompatible_subjects:        
        df_merged = df_merged[~df_merged['subject'].isin(incompatible_subjects)]
    
    return df_merged, incompatible_subjects_info

# calculates the distance between two columns in a dataframe row
def distance(row, col_a, col_b):
    a = row[col_a]
    b = row[col_b]
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.linalg.norm(a - b)
    return None

def compute_center(row, method_name_list):
    coords = []
    for m in method_name_list:
        c = row.get(f"coords_{m}")

        if isinstance(c, np.ndarray):
            coords.append(c)
    if len(coords) == 0:
        return None
    return np.mean(coords, axis=0)


# calculates center point and distances to center for each method
def calc_center_and_distances(df, method_name_list):
    df = df.copy()

    df["center_point"] = df.apply(lambda row: compute_center(row, method_name_list), axis=1)

    # add distances of each method to center point
    for m in method_name_list:
        col_coords = f"coords_{m}"
        dist_center_col = f"dist_center_{m}"

        df[dist_center_col] = df.apply(lambda row: distance(row, col_coords, "center_point"), axis=1)

    return df


# calculates pairwise distances between methods
def calc_distance_between_methods(df, method_name_list):
    method_pairs = itertools.combinations(method_name_list, 2)

    for m1, m2 in method_pairs:
        col1 = f"coords_{m1}"
        col2 = f"coords_{m2}"
        dist_col = f"dist_{m1}_{m2}"

        df[dist_col] = df.apply(lambda row: distance(row, col1, col2), axis=1)

    return df


def calc_distance_to_average(df, method_det, method_list_dl):
    # calculate the average coordinates of method_list_dl (deep learning models)
    df["avg_point_dl"] = df.apply(lambda row: compute_center(row, method_list_dl), axis=1)

    # calculate distance from method_det to avg_point_dl
    col_det = f"coords_{method_det}"
    out_col = f"dist_{method_det}_avg_dl"

    df[out_col] = df.apply(lambda row: distance(row, col_det, "avg_point_dl"), axis=1)

    return df


# computes statistics (mean, max) for distance columns in the dataframe (optional grouping) 
def compute_distance_statistics(df, group_cols=None):

    distance_columns = [col for col in df.columns if col.startswith("dist")]

    if group_cols is None:
        stats = df[distance_columns].agg(["mean", "max"])
        return stats

    stats = df.groupby(group_cols)[distance_columns].agg(["mean", "max"])
    return stats

def analyze_all_entries(df):
    return compute_distance_statistics(df, group_cols=None)

def analyze_by_subject(df):
    return compute_distance_statistics(df, group_cols=["subject"])

def analyze_by_poi(df):
    return compute_distance_statistics(df, group_cols=["poi_idx"])

def analyze_by_vertebra(df):
    return compute_distance_statistics(df, group_cols=["vertebra"])


def save_outliers(df, threshold):

    dist_cols = [col for col in df.columns if col.startswith("dist")]

    mask = (df[dist_cols] > threshold).any(axis=1)

    outliers_df = df[mask].copy()

    return outliers_df



def local_to_global_poi(data_path):
    subjects_list = [
        name for name in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, name))
    ]

    for subject in subjects_list:
        poi_path = os.path.join(data_path, subject, "poi_predicted.json")

        if not os.path.exists(poi_path):
            print(f"POI file not found for subject {subject} at {poi_path}, skipping.")
            continue

        poi = POI.load(poi_path)
        poi_global_path = poi_path.replace("predicted.json", "predicted_global.json")
        poi.to_global().save_mrk(poi_global_path)


    



if __name__ == "__main__":
    #path_method_list = [
    #    ("predictions/verse19-deterministic", "deter"),
    #    ("predictions/verse19-inferenced/subreg-project_gt-no_freeze-SADenseNet-NoVertPatchTransformer-excel_outliers_exclude", "dl-NoVert"),
    #    ("predictions/verse19-inferenced/subreg-project_gt-no_freeze-SADenseNet-standard_architecture-excel_outliers_exclude", "dl-standard"),
    #]

    path_method_list = [
        #("predictions/compare_eval_inference/eval/subreg_standard_excel_outliers_exclude_version0_epoch60_test-batch_size1/prediction_files-no_proj", "eval-1"),
        #("predictions/compare_eval_inference/eval/subreg_standard_excel_outliers_exclude_version0_epoch60_test/prediction_files-no_proj", "eval"),
        
        #("predictions/compare_eval_inference/eval/subreg_standard_excel_outliers_exclude_version0_epoch60_test/prediction_files-no_proj", "eval"),
        #("predictions/compare_eval_inference/inference/subreg-standard-excel_outliers_exclude-version0_epoch60-test/cutouts-preproccessed", "inference")
    ]
    save_dir = "predictions/compare_eval_inference"

    #local_to_global_poi("predictions/verse19-deterministic")

    
    df, incompatible_subjects_info = join_poi_dfs(path_method_list, load_full_spine=False)
    
    with open(os.path.join(save_dir, 'incompatible_subjects.json'), 'w') as f:
        json.dump(incompatible_subjects_info, f, indent=2, default=str)


    method_name_list = [m for (_, m) in path_method_list]
    #df = calc_center_and_distances(df, method_name_list)
    df = calc_distance_between_methods(df, method_name_list)

    #df = calc_distance_to_average(df, "deter", ["dl-NoVert", "dl-standard"])

    print(df)

    os.makedirs(save_dir, exist_ok=True)

    df.to_csv(os.path.join(save_dir, "comparison_of_predictions.csv"), index=False)

    # analyze
    analyze_all_entries(df).to_csv(os.path.join(save_dir, "stats_all_entries.csv"))
    analyze_by_subject(df).to_csv(os.path.join(save_dir,"stats_by_subject.csv"))
    analyze_by_poi(df).to_csv(os.path.join(save_dir, "stats_by_poi.csv"))
    analyze_by_vertebra(df).to_csv(os.path.join(save_dir, "stats_by_vertebra.csv"))

    # save outliers
    save_outliers(df, threshold=20).to_csv(os.path.join(save_dir, "outliers_distance_above_20mm.csv"))
    

