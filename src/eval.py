#Alissa
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent)) # Add project root to Python path


import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
#from BIDS import POI
from TPTBox.core.poi import POI
from TPTBox import NII  # For loading NIfTI files

import shutil  # For file operations

from modules.PoiModule import PoiPredictionModule
from src.modules.PoiDataModules import JointDataModule, POIDataModule
from utils.misc import surface_project_coords


def load_data_module_from_config(config_path, joint=False, alternative_poi_ending=None):
    # Load the configuration file
    with open(config_path, "r") as f:
        config = json.load(f)

    # Instantiate the DataModule with the loaded configurations
    config["batch_size"] = 1
    if alternative_poi_ending is not None:
        config["poi_file_ending"] = alternative_poi_ending
    if joint:
        return JointDataModule(**config)
    else:
        return POIDataModule(**config)


def load_model_from_checkpoint(checkpoint_path):
    # Load the model from the checkpoint
    model = PoiPredictionModule.load_from_checkpoint(checkpoint_path)
    return model


def np_to_ctd(
    t,
    vertebra,
    origin,
    rotation,
    idx_list=None,
    shape=(128, 128, 96),
    zoom=(1, 1, 1),
    offset=(0, 0, 0),
    orientation=None,  # <- Neu: orientation als Argument
):
    ctd = {}
    for i, coords in enumerate(t):
        coords = np.array(coords).astype(float) - np.array(offset).astype(float)
        coords = (coords[0], coords[1], coords[2])
        if idx_list is None:
            ctd[vertebra, i] = coords
        elif i < len(idx_list):
            ctd[vertebra, idx_list[i]] = coords

    ###
    if orientation is None:
        raise ValueError("You must provide the orientation of the input POI.")

    ctd = POI(
        centroids=ctd,
        orientation=orientation,#("L", "A", "S"),
        zoom=zoom,
        shape=shape,
        origin=origin,
        rotation=rotation,
    )

    ctd.reorient_(axcodes_to=("L", "A", "S"), verbose=False).rescale_((1, 1, 1), verbose=False)

    return ctd


def create_prediction_poi_files(
    data_module_save_path,
    checkpoint_path,
    poi_file_ending,
    split="val",
    joint=False,
    save_in_dir=False,
    save_path=None,
    return_paths=False,
    project=True,
):
    # Create the POI files for the refined predictions
    if return_paths:
        poi_paths_dict = {}
    if not save_in_dir and save_path is None:
        raise ValueError("Either save_in_dir or save_path must be set")

    # Assert that the poi_file_ending is a json file
    if not poi_file_ending.endswith(".json"):
        raise ValueError("The poi_file_ending must be a json file")

    data_module = load_data_module_from_config(data_module_save_path, joint=joint)
    data_module.setup()

    # Load the checkpoint
    poi_module = PoiPredictionModule.load_from_checkpoint(checkpoint_path)

    # Set the model to evaluation mode
    poi_module.eval()

    if split == "val":
        val_dl = data_module.val_dataloader()
    elif split == "test":
        val_dl = data_module.test_dataloader()
    else:
        raise ValueError(f"Invalid split: {split}")

    for batch in val_dl:
        # Bring all tensors to device
        batch = {
            k: v.to(poi_module.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        batch = poi_module(batch)

        subject_batch = batch["subject"]
        vertebra_batch = batch["vertebra"]
        refined_preds_batch = batch["refined_preds"]
        if project:
            refined_preds_projected_batch, _ = surface_project_coords(
                refined_preds_batch, batch["surface"]
            )
        target_indices_batch = batch["target_indices"]
        offset_batch = batch["offset"]
        poi_path_batch = batch["poi_path"]

        loss_mask_batch = batch["loss_mask"] #Alissa

        coarse_preds_batch = batch["coarse_preds"]

        subreg = NII.load(batch["subreg_path"][0], seg=True)
        vertseg = NII.load(batch["msk_path"][0], seg=True)

        match = subreg.assert_affine(vertseg, raise_error=False, verbose=True)

        if not match:
            print(f"❌ Affine/Metadata stimmt nicht überein bei: \nsubreg: {subreg.affine}, \nvertseg: {vertseg.affine}")
        

        # Detach all tensors
        vertebra_batch = vertebra_batch.detach().cpu().numpy()
        refined_preds_batch = refined_preds_batch.detach().cpu().numpy()
        if project:
            refined_preds_projected_batch = (
                refined_preds_projected_batch.detach().cpu().numpy()
            )
        target_indices_batch = target_indices_batch.detach().cpu().numpy()
        offset_batch = offset_batch.detach().cpu().numpy()

        loss_mask_batch = loss_mask_batch.detach().cpu().numpy() #Alissa

        pred_batch = refined_preds_projected_batch if project else refined_preds_batch

        for sub, vert, preds, indices, poi_path, offset, mask in zip( #Alissa: mask
            subject_batch,
            vertebra_batch,
            pred_batch,
            target_indices_batch,
            poi_path_batch,
            offset_batch,
            loss_mask_batch, #Alissa
        ):
            
            #Alissa: Filter nur gültige POIs (mask == True)
            #mask = loss_mask_batch.astype(bool)
            preds = preds[mask]
            indices = indices[mask]

            # Open the old POI file to get the origin and rotation
            ctd = POI.load(poi_path)
            origin = ctd.origin
            rotation = ctd.rotation
            shape = ctd.shape
            zoom = ctd.zoom
            orientation = ctd.orientation ### Alissa

            # Create the new POI file
            ctd = np_to_ctd(
                preds,
                vert,
                origin,
                rotation,
                idx_list=indices,
                shape=shape,
                zoom=zoom,
                offset=offset,
                orientation=orientation ### Alissa
            )

            if save_in_dir:
                ctd_save_path = poi_path.replace(
                    data_module.poi_file_ending, poi_file_ending
                )
                # Make sure we do not overwrite the original POI file
                if ctd_save_path == poi_path:
                    # Print warning
                    print(
                        f"Warning: The save path {ctd_save_path} is the same as the original POI path. The new file will be saved with the ending '_pred.json'"
                    )
                    ctd_save_path = poi_path.replace(".json", "_pred.json")
                    

            else:
                # Make sure the save path exists
                os.makedirs(save_path, exist_ok=True)
                ctd_save_path = os.path.join(
                    save_path, str(sub) + "_" + str(vert) + "_" + poi_file_ending
                )

            ctd.save(ctd_save_path, verbose=False)
            
            

            # === (1) Speichere globale Prediction-POIs ===
            if not os.path.exists(ctd_save_path):
                print(f"⚠️ Prediction file not found: {ctd_save_path}")
                continue
            ctd_global_save_path = ctd_save_path.replace("_pred.json", "_pred_global.json")
            POI.load(ctd_save_path).to_global().save_mrk(ctd_global_save_path)

            # === (2) Speichere GT-POI-Datei ===
            gt_poi = POI.load(poi_path).extract_region(vert)
            gt_save_path = ctd_save_path.replace("_pred.json", "_gt.json")
            gt_poi.save(gt_save_path)

            # === (3) Speichere globale GT-POIs ===
            gt_global_save_path = gt_save_path.replace("_gt.json", "_gt_global.json")
            gt_poi.to_global().save_mrk(gt_global_save_path)

            # === (4) Kopiere Segmentationsmaske ===
            seg_vert_path = poi_path.replace("poi.json", "vertseg.nii.gz")
            seg_save_path = ctd_save_path.replace("_pred.json", "_seg.nii.gz")
            if os.path.exists(seg_vert_path):
                shutil.copy(seg_vert_path, seg_save_path)
            else:
                print(f"⚠️ Segmentation file not found: {seg_vert_path}")




            if return_paths:
                poi_paths_dict[sub, vert] = {
                    "gt": poi_path,
                    "pred": ctd_save_path,
                    "seg_vert": poi_path.replace("poi.json", "vertseg.nii.gz"),
                }

    if return_paths:
        return poi_paths_dict

def create_self_training_pois(
    data_module_save_path,
    checkpoint_path,
    poi_file_ending,
    split="val",
    joint=False,
    thre=3.0,
):
    new_bad_pois = {
        "subject": [],
        "vertebra": [],
        "bad_poi_list": [],
    }

    # Assert that the poi_file_ending is a json file
    if not poi_file_ending.endswith(".json"):
        raise ValueError("The poi_file_ending must be a json file")

    data_module = load_data_module_from_config(data_module_save_path, joint=joint)
    data_module.setup()

    # Load the checkpoint
    poi_module = PoiPredictionModule.load_from_checkpoint(checkpoint_path)

    # Set the model to evaluation mode
    poi_module.eval()

    if split == "val":
        val_dl = data_module.val_dataloader()
    elif split == "test":
        val_dl = data_module.test_dataloader()
    else:
        raise ValueError(f"Invalid split: {split}")

    for batch in val_dl:
        # Bring all tensors to device
        batch = {
            k: v.to(poi_module.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        batch = poi_module(batch)

        # Get target, projected refined_preds, target_indices, loss_mask, offset
        # and poi_path
        subject_batch = batch["subject"]
        vertebra_batch = batch["vertebra"]
        refined_preds_batch = batch["refined_preds"]
        refined_preds_projected_batch, _ = surface_project_coords(
            refined_preds_batch, batch["surface"]
        )
        target_indices_batch = batch["target_indices"]
        offset_batch = batch["offset"]
        poi_path_batch = batch["poi_path"]
        loss_mask_batch = batch["loss_mask"]
        target_batch = batch["target"]
        target_projected_batch, _ = surface_project_coords(
            target_batch, batch["surface"]
        )

        # Detach all tensors
        vertebra_batch = vertebra_batch.detach().cpu().numpy()
        refined_preds_batch = refined_preds_batch.detach().cpu().numpy()
        refined_preds_projected_batch = (
            refined_preds_projected_batch.detach().cpu().numpy()
        )
        target_indices_batch = target_indices_batch.detach().cpu().numpy()
        offset_batch = offset_batch.detach().cpu().numpy()
        loss_mask_batch = loss_mask_batch.detach().cpu().numpy()
        target_projected_batch = target_projected_batch.detach().cpu().numpy()

        for sub, vert, preds, indices, poi_path, offset, loss_mask, target in zip(
            subject_batch,
            vertebra_batch,
            refined_preds_projected_batch,
            target_indices_batch,
            poi_path_batch,
            offset_batch,
            loss_mask_batch,
            target_projected_batch,
        ):
            # Open the old POI file to get the origin and rotation
            ctd = POI.load(poi_path)
            origin = ctd.origin
            rotation = ctd.rotation
            shape = ctd.shape
            zoom = ctd.zoom
            orientation = ctd.orientation ### Alissa

            # Where the loss mask is false (i.e. a bad gt), we use the predicted POI
            # as the new POI
            pred_mask = np.logical_not(loss_mask)

            # If the distance between predicted and target is larger than the threshold,
            # we mark the POI as bad
            bad_poi_idx = np.where(np.linalg.norm(preds - target, axis=1) > thre)[0]
            bad_poi_indices = indices[bad_poi_idx]

            # Create the target, use preds where pred_mask and else use target
            new_target = np.where(pred_mask[:, None], preds, target)

            # Create the new POI file
            ctd = np_to_ctd(
                new_target,
                vert,
                origin,
                rotation,
                idx_list=indices,
                shape=shape,
                zoom=zoom,
                offset=offset,
                orientation=orientation ### Alissa
            )

            ctd_save_path = poi_path.replace(
                data_module.poi_file_ending, poi_file_ending
            )
            # Make sure we do not overwrite the original POI file
            if ctd_save_path == poi_path:
                # Print warning
                print(
                    f"Warning: The save path {ctd_save_path} is "
                    "the same as the original  POI path. "
                    "The new file will be saved with the ending '_pred.json'"
                )
                ctd_save_path = poi_path.replace(".json", "_pred.json")

            ctd.save(ctd_save_path, verbose=False)

            new_bad_pois["subject"].append(sub)
            new_bad_pois["vertebra"].append(vert)
            new_bad_pois["bad_poi_list"].append(bad_poi_indices)

    return new_bad_pois

def run_predictions(
    data_module_save_path,
    checkpoint_path,
    split="val",
    joint=False,
    alternative_poi_ending=None,
):
    # Change the ending of the POI files if necessary
    data_module = load_data_module_from_config(
        data_module_save_path,
        joint=joint,
        alternative_poi_ending=alternative_poi_ending,
    )
    data_module.setup()

    # Load the checkpoint
    poi_module = PoiPredictionModule.load_from_checkpoint(checkpoint_path)

    # Set the model to evaluation mode
    poi_module.eval()

    if split == "val":
        val_dl = data_module.val_dataloader()
    elif split == "test":
        val_dl = data_module.test_dataloader()
    else:
        raise ValueError(f"Invalid split: {split}")

    project_gt = poi_module.hparams.refinement_config["params"]["project_gt"]

    pred_dict = {
        "subject": [],
        "vertebra": [],
        "poi_idx": [],
        "target": [],
        "coarse": [],
        "refined": [],
        "coarse_proj": [],
        "refined_proj": [],
        "coarse_proj_dist": [],
        "refined_proj_dist": [],
        "loss_mask": [],
    }

    for batch in val_dl:
        # Bring all torch tensors to device
        batch = {
            k: v.to(poi_module.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        batch = poi_module(batch)

        # Get target, coarse preds, refined preds, subject, vertebra and target indices
        subject_batch = batch["subject"]
        vertebra_batch = batch["vertebra"]

        target_batch = batch["target"]
        target_indices_batch = batch["target_indices"]
        loss_mask_batch = batch["loss_mask"]
        coarse_preds_batch = batch["coarse_preds"]
        refined_preds_batch = batch["refined_preds"]

        if project_gt:
            target_batch, _ = surface_project_coords(target_batch, batch["surface"])

        coarse_preds_projected_batch, coarse_pred_proj_distances_batch = (
            surface_project_coords(coarse_preds_batch, batch["surface"])
        )
        refined_preds_projected_batch, refined_preds_proj_distances_batch = (
            surface_project_coords(refined_preds_batch, batch["surface"])
        )

        # Detach all tensors and convert to numpy
        vertebra_batch = vertebra_batch.detach().cpu().numpy()
        target_batch = target_batch.detach().cpu().numpy()
        target_indices_batch = target_indices_batch.detach().cpu().numpy()
        loss_mask_batch = loss_mask_batch.detach().cpu().numpy()
        coarse_preds_batch = coarse_preds_batch.detach().cpu().numpy()
        refined_preds_batch = refined_preds_batch.detach().cpu().numpy()
        coarse_preds_projected_batch = (
            coarse_preds_projected_batch.detach().cpu().numpy()
        )
        coarse_pred_proj_distances_batch = (
            coarse_pred_proj_distances_batch.detach().cpu().numpy()
        )
        refined_preds_projected_batch = (
            refined_preds_projected_batch.detach().cpu().numpy()
        )
        refined_preds_proj_distances_batch = (
            refined_preds_proj_distances_batch.detach().cpu().numpy()
        )

        keys = [
            "target",
            "target_indices",
            "coarse_preds",
            "refined_preds",
            "coarse_preds_projected",
            "coarse_pred_proj_distances",
            "refined_preds_projected",
            "refined_preds_proj_distances",
            "loss_mask",
            "subject",
            "vertebra",
        ]

        for values in zip(
            target_batch,
            target_indices_batch,
            coarse_preds_batch,
            refined_preds_batch,
            coarse_preds_projected_batch,
            coarse_pred_proj_distances_batch,
            refined_preds_projected_batch,
            refined_preds_proj_distances_batch,
            loss_mask_batch,
            subject_batch,
            vertebra_batch,
        ):
            data_dict = dict(zip(keys, values))
            # Iterate over all POIs to collect POI-wise information
            for poi_idx, t, c, r, c_proj, r_proj, c_proj_dist, r_proj_dist, l in zip(
                data_dict["target_indices"],
                data_dict["target"],
                data_dict["coarse_preds"],
                data_dict["refined_preds"],
                data_dict["coarse_preds_projected"],
                data_dict["refined_preds_projected"],
                data_dict["coarse_pred_proj_distances"],
                data_dict["refined_preds_proj_distances"],
                data_dict["loss_mask"],
            ):
                pred_dict["subject"].append(data_dict["subject"])
                pred_dict["vertebra"].append(data_dict["vertebra"])
                pred_dict["poi_idx"].append(poi_idx)
                pred_dict["target"].append(t)
                pred_dict["coarse"].append(c)
                pred_dict["refined"].append(r)
                pred_dict["coarse_proj"].append(c_proj)
                pred_dict["refined_proj"].append(r_proj)
                pred_dict["coarse_proj_dist"].append(c_proj_dist)
                pred_dict["refined_proj_dist"].append(r_proj_dist)
                pred_dict["loss_mask"].append(l)

    return pred_dict

def create_prediction_df(
    data_module_save_path,
    checkpoint_path,
    split="val",
    joint=False,
    alternative_poi_ending=None,
):
    pred_dict = run_predictions(
        data_module_save_path, checkpoint_path, split, joint, alternative_poi_ending
    )
    # Calculate distances between target and predicted POIs
    pred_dict["coarse_error"] = [
        np.linalg.norm(np.array(t) - np.array(c))
        for t, c in zip(pred_dict["target"], pred_dict["coarse"])
    ]
    pred_dict["refined_error"] = [
        np.linalg.norm(np.array(t) - np.array(r))
        for t, r in zip(pred_dict["target"], pred_dict["refined"])
    ]

    # Calculate distances between target and projected POIs
    pred_dict["coarse_proj_error"] = [
        np.linalg.norm(np.array(t) - np.array(c))
        for t, c in zip(pred_dict["target"], pred_dict["coarse_proj"])
    ]
    pred_dict["refined_proj_error"] = [
        np.linalg.norm(np.array(t) - np.array(r))
        for t, r in zip(pred_dict["target"], pred_dict["refined_proj"])
    ]

    # Create DataFrame
    df = pd.DataFrame(pred_dict)
    return df

def calculate_metrics(errors, threshold=2.0):
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    mse = np.mean(errors**2)
    accuracy = np.mean(errors < threshold)
    max_error = np.max(errors)
    return mean_error, median_error, mse, accuracy, max_error

def compute_overall_metrics(df):
    # Create an empty DataFrame to hold the metrics
    metrics_df = pd.DataFrame(
        columns=["Mean Error", "Median Error", "MSE", "Accuracy", "Max Error"]
    )

    # Calculate metrics for each error type
    for error_type in [
        "coarse_error",
        "refined_error",
        "coarse_proj_error",
        "refined_proj_error",
    ]:
        metrics_df.loc[error_type] = calculate_metrics(df[error_type])

    return metrics_df

def compute_poi_wise_metrics(df):
    # Group by poi_idx and calculate metrics for refined_proj_error
    grouped = df.groupby("poi_idx")["refined_proj_error"]
    metrics_df = grouped.apply(lambda x: calculate_metrics(x)).apply(pd.Series)
    metrics_df.columns = ["Mean Error", "Median Error", "MSE", "Accuracy", "Max Error"]

    return metrics_df

def compute_vert_wise_metrics(df):
    # Group by vertebra and calculate metrics for refined_proj_error
    grouped = df.groupby("vertebra")["refined_proj_error"]
    metrics_df = grouped.apply(lambda x: calculate_metrics(x)).apply(pd.Series)
    metrics_df.columns = ["Mean Error", "Median Error", "MSE", "Accuracy", "Max Error"]

    return metrics_df

def compute_sub_wise_metrics(df):
    # Group by vertebra and calculate metrics for refined_proj_error
    grouped = df.groupby("subject")["refined_proj_error"]
    metrics_df = grouped.apply(lambda x: calculate_metrics(x)).apply(pd.Series)
    metrics_df.columns = ["Mean Error", "Median Error", "MSE", "Accuracy", "Max Error"]

    return metrics_df

def filter_high_error_pois(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Gibt ein neues DataFrame mit subject, vertebra und poi_idx zurück,
    für die der refined_proj_error größer als thre_hold ist.

    Parameter:
    df (pd.DataFrame): Eingabedatenframe mit Fehlerwerten.
    threshold (float): Schwellwert für refined_proj_error.

    Rückgabe:
    pd.DataFrame: Gefiltertes DataFrame mit den Spalten subject, vertebra und poi_idx.
    """
    filtered_df = df[df['refined_proj_error'] > threshold]
    return filtered_df[['subject', 'vertebra', 'poi_idx', 'refined_proj_error']].reset_index(drop=True)


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_module_save_path",
        type=str,
        help="Path to the saved DataModule configuration",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, help="Path to the saved checkpoint"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split to evaluate on (val/test)",
    )
    parser.add_argument(
        "--save_path", type=str, help="Path to save the evaluation results"
    )
    parser.add_argument(
        "--joint", action="store_true", help="Whether to use the JointDataModule"
    )


    args = parser.parse_args()
    

    os.makedirs(args.save_path, exist_ok=True)

    
    ### Create DataFrame with prediction information 
    prediction_df = create_prediction_df(
        args.data_module_save_path, args.checkpoint_path, args.split, args.joint
    )
    prediction_df = prediction_df[prediction_df['loss_mask'] == True]

    prediction_df.to_csv(os.path.join(args.save_path, "results.csv"))
    print("Prediction DataFrame saved")

    ### Compute overall metrics 
    metrics_df = compute_overall_metrics(prediction_df)
    metrics_df.to_csv(os.path.join(args.save_path, "overall_metrics.csv"))
    print("Overal metrics saved")

    ### Compute POI-wise metrics 

    #prediction_df = pd.read_csv("experiments/experiment_evaluation/gruber/surface/excel_excluded_pois/no_freeze/val/version_2_epoch_55/results.csv")
    #save_path = "experiments/experiment_evaluation/gruber/surface/excel_excluded_pois/no_freeze/val/version_2_epoch_55/"
    poi_metrics_df = compute_poi_wise_metrics(prediction_df)
    poi_metrics_df.to_csv(os.path.join(args.save_path, "poi_metrics.csv"))
    print("POI-wise metrics saved")

    ### Compute vertebra-wise metrics 
    vert_metrics_df = compute_vert_wise_metrics(prediction_df)
    vert_metrics_df.to_csv(os.path.join(args.save_path, "vertebra_metrics.csv"))
    print("Vertebra-wise metrics saved")

    ### Compute subject-wise metrics
    sub_metrics_df = compute_sub_wise_metrics(prediction_df)
    sub_metrics_df.to_csv(os.path.join(args.save_path, "subject_metrics.csv"))
    print("Subject-wise metrics saved")


    ### Find Outliers
    outlier_df = filter_high_error_pois(prediction_df, 10)
    outlier_df.to_csv(os.path.join(args.save_path, "outliers_error_higher_10.csv"))
    print("Outliers (error > 10) saved")
    
    
    ### Create Prediction files 
    prediction_files_path = os.path.join(args.save_path, "prediction_files")
    os.makedirs(prediction_files_path, exist_ok=True)

    # Generate predictions and get paths
    poi_paths_dict = create_prediction_poi_files(
        data_module_save_path=args.data_module_save_path,
        checkpoint_path=args.checkpoint_path,
        poi_file_ending="_pred.json",
        split=args.split,
        joint=args.joint,
        save_path=prediction_files_path,
        return_paths=True, 
        project=True  
    )

    """
    # Copy ground truth POIs to output directory for comparison
    for (sub, vert), paths in poi_paths_dict.items():
        gt_poi = POI.load(paths["gt"])
        gt_poi = gt_poi.extract_region(vert)  # Extract region for the specific vertebra
        gt_save_path = os.path.join(prediction_files_path, f"{sub}_{vert}_gt.json")
        gt_poi.save(gt_save_path)

        # Save global ground truth POI
        gt_global_save_path = gt_save_path.replace("_gt.json", "_gt_global.json")
        gt_poi.to_global().save_mrk(gt_global_save_path)

        seg_path = paths["seg_vert"]
        seg_save_path = os.path.join(prediction_files_path, f"{sub}_{vert}_seg.nii.gz")
        shutil.copy(seg_path, seg_save_path)
    """

    print(f"Saved predictions and ground truths to: {prediction_files_path}")