"""Script to update master_df with POI exclusions from Excel and outlier CSV files."""

import argparse
import ast
import os
import pandas as pd


def load_exclusion_dict(excel_path):
    """Load Excel file and create lookup dictionary for exclusions"""
    if not os.path.exists(excel_path):
        return {}
    
    df = pd.read_excel(excel_path)
    exclude_dict = {}

    for _, row in df.iterrows():
        subject = row['subject']
        label = int(row['label'])  

        for col in df.columns[2:]:  # columns: 'subject' and 'label'
            val = str(row[col]).strip().lower()
            if val == 'x':
                try:
                    poi_id = int(col.strip().split()[0])  # e.g. '124 \n(VertBodAntCenR)' → 124
                except ValueError:
                    continue  # if no valid POI ID can be extracted

                if subject not in exclude_dict:
                    exclude_dict[subject] = []
                exclude_dict[subject].append((label, poi_id))
    
    return exclude_dict


def get_bad_poi_list(subject_id: str, vert: int, exclude_dict: dict[str, list[tuple[int, int]]]) -> list[int]:
    """
    Args:
        subject_id: Subject ID, e.g., 'WS-13'
        vert_id: Vertebra ID, e.g., 
        exclude_dict: Dict mapping subject_id -> list of (vert_id, poi_id)

    Returns:
        A list of global POI IDs
    """
    if exclude_dict is None:
        return []
    bad_pois = exclude_dict.get(subject_id, [])
    filtered_pois = [poi_id for vert_id, poi_id in bad_pois if vert_id == vert]
    return filtered_pois


def add_excel_exclusions_to_master_df(master_df_path, excel_exclude_path, save_path=None):
    """
    Adds all POI exclusions from the Excel file to the bad_poi_list in master_df.
    
    Args:
        master_df_path (str): Path to master_df.
        excel_exclude_path (str): Path to excel file with POI exclusions.
        save_path (str, optional): Where updated master_df should be saved. 
                                   If none, nothing is saved, only returned.
        
    Returns:
        pd.DataFrame: updated master_df
    """
    master_df = pd.read_csv(master_df_path)
    
    exclusion_dict = load_exclusion_dict(excel_exclude_path)
    
    # convert all bad_poi_list entries from strings to lists
    master_df["bad_poi_list"] = master_df["bad_poi_list"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() != '[]' else []
    )
    
    # add new POIs from Excel to existing bad_poi_list
    for idx, row in master_df.iterrows():
        subject = row['subject']
        vertebra = row['vertebra']
        current_bad_pois = row['bad_poi_list']
        
        new_bad_pois = get_bad_poi_list(f"sub-{subject}", vertebra, exclusion_dict)
        
        # add new POIs to existing list (without duplicates)
        if new_bad_pois:
            updated_bad_pois = list(set(current_bad_pois + new_bad_pois))
            master_df.at[idx, 'bad_poi_list'] = updated_bad_pois
            print(f"Subject {subject}, Vertebra {vertebra}: Added {new_bad_pois}")
    
    if save_path:
        master_df.to_csv(save_path, index=False)

    return master_df


def add_outliers_to_master_df(master_df_path, outlier_paths, save_path=None):
    """
    Adds all POI outliers from the given CSV files to the bad_poi_list in master_df.
    
    Args:
        master_df_path (str): Path to master_df.
        outlier_paths (list[str]): Lists of paths to outlier CSV files.
        save_path (str, optional): Where updated master_df should be saved. 
                                   If none, nothing is saved, only returned.
        
    Returns:
        pd.DataFrame: Updated master_df.
    """
    master_df = pd.read_csv(master_df_path)

    # convert all bad_poi_list entries from strings to lists
    master_df["bad_poi_list"] = master_df["bad_poi_list"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() != '[]' else []
    )

    for path in outlier_paths:
        outlier_df = pd.read_csv(path)

        for _, row in outlier_df.iterrows():
            subj = row["subject"]
            vert = row["vertebra"]
            poi = int(row["poi_idx"])

            mask = (master_df["subject"] == subj) & (master_df["vertebra"] == vert)
            if not mask.any():
                print(f"⚠️ No match found for Subject {subj}, Vertebra {vert}")
                continue

            master_df.loc[mask, "bad_poi_list"] = master_df.loc[mask, "bad_poi_list"].apply(
                lambda l: l + [poi] if poi not in l else l
            )

    if save_path:
        master_df.to_csv(save_path, index=False)

    return master_df


def add_excel_and_outliers_to_master_df(master_df_path, excel_exclude_path=None, outlier_paths=None, save_path=None):
    """
    Adds POI exclusions from an Excel file and outliers from CSV files to the master_df.
    
    Args:
        master_df_path (str): Path to master_df.
        excel_exclude_path (str, optional): Path to Excel file with POI exclusions.
        outlier_paths (list[str], optional): List of paths to outlier CSV files.
        save_path (str, optional): Where updated master_df should be saved. 
                                   If none, nothing is saved, only returned.
        
    Returns:
        pd.DataFrame: Updated master_df with both Excel exclusions and outliers.
    """
    master_df = pd.read_csv(master_df_path)
    
    if excel_exclude_path and os.path.exists(excel_exclude_path):
        master_df = add_excel_exclusions_to_master_df(
            master_df_path=master_df_path, 
            excel_exclude_path=excel_exclude_path,
            save_path=save_path  
        )
    
        if outlier_paths:
            master_df = add_outliers_to_master_df(
                master_df_path=save_path, 
                outlier_paths=outlier_paths,
                save_path=save_path 
            )
    
    elif outlier_paths:
        master_df = add_outliers_to_master_df(
            master_df_path=master_df_path, 
            outlier_paths=outlier_paths,
            save_path=save_path 
        )

    return master_df

if __name__ == "__main__":

    mode = 'both'  # Default mode

    master_df_path = 'cutout-folder/cutouts-all_pois/master_df.csv'
    excel_exclude_path = 'exclude/correction_excel'
    outlier_paths = [
        "experiments/experiment_evaluation/k_fold/fold_1/test/outliers_error_higher_10.csv",
        "experiments/experiment_evaluation/k_fold/fold_2/test/outliers_error_higher_10.csv",
        "experiments/experiment_evaluation/k_fold/fold_3/test/outliers_error_higher_10.csv",
        "experiments/experiment_evaluation/k_fold/fold_4/test/outliers_error_higher_10.csv",
        "experiments/experiment_evaluation/k_fold/fold_5/test/outliers_error_higher_10.csv",
        "experiments/experiment_evaluation/k_fold/fold_6/test/outliers_error_higher_10.csv"
        ]
    save_path_excel = 'cutout-folder/cutouts-all_pois/master_df-excel_exclude.csv'
    save_path_outliers = 'cutout-folder/cutouts-all_pois/master_df-outliers_exclude.csv'
    save_path_both = 'cutout-folder/cutouts-all_pois/master_df-excel_outliers_exclude.csv'



    if mode == 'excel':
        add_excel_exclusions_to_master_df(
            master_df_path=master_df_path,
            excel_exclude_path=excel_exclude_path,
            save_path=save_path_excel
        )
    
    elif mode == 'outliers':
        add_outliers_to_master_df(
            master_df_path=master_df_path,
            outlier_paths=outlier_paths,
            save_path=save_path_outliers
        )
    
    elif mode == 'both':
        add_excel_and_outliers_to_master_df(
            master_df_path=master_df_path,
            excel_exclude_path=excel_exclude_path,
            outlier_paths=outlier_paths,
            save_path=save_path_both
        )
    
    else:
        print("unknown mode, please specify 'excel', 'outliers', or 'both'")