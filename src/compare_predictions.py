import os
import pandas as pd
import numpy as np
from TPTBox.core.poi import POI

import itertools 
from functools import reduce
  

def distance(a, b):
    return np.linalg.norm(a - b)

def create_poi_df(data_path, method_name):
    """
    param poi_path: path to dictionary with POIs 
    return: 
        {
            'vertebra': <string>,
            'poi': <string>,
            'coords': np.array([x, y, z])
        }
    """


    poi_dict = {
        "subject": [],
        "vertebra": [],
        "poi_idx": [],
        f"coords_{method_name}": []
    }

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

        for vert, poi_id, coords in poi.items(): 
            poi_dict["subject"].append(subject)
            poi_dict["vertebra"].append(vert)
            poi_dict["poi_idx"].append(poi_id)
            poi_dict[f"coords_{method_name}"].append(np.array(coords))

    poi_df = pd.DataFrame(poi_dict)
    return poi_df

def join_poi_dfs(path_method_list):
    """
    """
    df_list = []
    for (data_path, method_name) in path_method_list:
        poi_df = create_poi_df(data_path, method_name)
        df_list.append(poi_df)


    df_merged = reduce(
        lambda left, right: left.merge(
            right, on=["subject", "vertebra", "poi_idx"], how="outer"
        ),
        df_list
    )

    df_merged = df_merged.dropna()

    return df_merged

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




    



if __name__ == "__main__":
    path_method_list = [
        ("predictions/verse19-deterministic", "deter"),
        ("predictions/verse19-inferenced-LAS/subreg-project_gt-no_freeze-SADenseNet-NoVertPatchTransformer-excel_outliers_exclude/", "dl-NoVert"),
        ("predictions/verse19-inferenced-LAS/subreg-project_gt-no_freeze-SADenseNet-standard_architecture-excel_outliers_exclude/", "dl-standard"),
    ]
    df = join_poi_dfs(path_method_list)

    method_name_list = [m for (_, m) in path_method_list]
    df = calc_center_and_distances(df, method_name_list)
    df = calc_distance_between_methods(df, method_name_list)

    df = calc_distance_to_average(df, "deter", ["dl-NoVert", "dl-standard"])

    print(df)

    df.to_csv("predictions/comparison_of_predictions.csv", index=False)

    # analyze
    analyze_all_entries(df).to_csv("predictions/stats_all_entries.csv")
    analyze_by_subject(df).to_csv("predictions/stats_by_subject.csv")
    analyze_by_poi(df).to_csv("predictions/stats_by_poi.csv")
    analyze_by_vertebra(df).to_csv("predictions/stats_by_vertebra.csv")

    # save outliers
    save_outliers(df, threshold=100).to_csv("predictions/outliers_distance_above_100mm.csv")

