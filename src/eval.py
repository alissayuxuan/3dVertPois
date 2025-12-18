# Alissa
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Add project root to Python path


import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# from BIDS import POI
from TPTBox.core.poi import POI
from TPTBox import NII  # For loading NIfTI files

import shutil  # For file operations

from modules.PoiModule import PoiPredictionModule
from src.modules.PoiDataModules import POIDataModule
from utils.misc import surface_project_coords


def load_data_module_from_config(config_path, alternative_poi_ending=None):
    # Load the configuration file
    with open(config_path, "r") as f:
        config = json.load(f)

    # Instantiate the DataModule with the loaded configurations
    config["batch_size"] = 1
    if alternative_poi_ending is not None:
        config["poi_file_ending"] = alternative_poi_ending
    else:
        return POIDataModule(**config)


def load_model_from_checkpoint(checkpoint_path):
    # Load the model from the checkpoint
    model = PoiPredictionModule.load_from_checkpoint(checkpoint_path)
    return model


def combine_centroids(data_list):
    # Extract the first dictionary for comparison
    first_entry = data_list[0]

    # Define the expected values for comparison
    expected_subject = first_entry["subject"]
    expected_shape = first_entry["original_shape"]
    expected_zoom = first_entry["original_zoom"]
    expected_orientation = first_entry["original_orientation"]
    expected_rotation = first_entry["original_rotation"]  # ALISSA
    expected_origin = first_entry["original_origin"]  # ALISSA

    # Initialize a defaultdict for combining centroids
    combined_centroids = {}

    # Iterate through each entry in the list
    for entry in data_list:
        # Assert that subject, shape, zoom, and orientation match the expected values
        assert entry["subject"] == expected_subject, "Subjects do not match."
        assert entry["original_shape"] == expected_shape, "Original shapes do not match."
        assert entry["original_zoom"] == expected_zoom, "Original zooms do not match."
        assert entry["original_orientation"] == expected_orientation, "Original orientations do not match."

        assert np.allclose(entry["original_rotation"], expected_rotation, rtol=1e-10), "Original rotations do not match."

        # Combine the centroids
        for v_idx, p_idx, c in entry["centroids"].items():
            combined_centroids[v_idx, p_idx] = c

    # Convert combined_centroids to a regular dict
    combined_centroids = dict(combined_centroids)

    # Return the common attributes and the combined centroids
    poi_file = POI(
        centroids=combined_centroids,
        orientation=expected_orientation,
        zoom=expected_zoom,
        shape=expected_shape,
        rotation=expected_rotation,  # ALISSA
        origin=expected_origin,  # ALISSA
    )

    return expected_subject, poi_file


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
        orientation=orientation,
        zoom=zoom,
        shape=shape,
        origin=origin,
        rotation=rotation,
    )

    # ctd.reorient_(axcodes_to=("L", "A", "S"), verbose=False).rescale_((1, 1, 1), verbose=False)

    return ctd


def create_prediction_poi_files(
    data_module_save_path,
    checkpoint_path,
    poi_file_ending,
    split="val",
    save_in_dir=False,
    save_path=None,
    return_paths=False,
    project=False,
    save_gt_proj=False,
):

    print(f"save_gt_proj: {save_gt_proj}")

    # Create the POI files for the refined predictions
    if return_paths:
        poi_paths_dict = {}
    if not save_in_dir and save_path is None:
        raise ValueError("Either save_in_dir or save_path must be set")

    # Assert that the poi_file_ending is a json file
    if not poi_file_ending.endswith(".json"):
        raise ValueError("The poi_file_ending must be a json file")

    data_module = load_data_module_from_config(data_module_save_path)
    data_module.setup()

    # Load the checkpoint
    poi_module = PoiPredictionModule.load_from_checkpoint(checkpoint_path)

    # Set the model to evaluation mode
    poi_module.eval()

    if split == "val":
        val_dl = data_module.val_dataloader()
    elif split == "test":
        val_dl = data_module.test_dataloader()
    elif split == "train":
        val_dl = data_module.train_noaug_dataloader()
    else:
        raise ValueError(f"Invalid split: {split}")

    for batch in val_dl:
        # Bring all tensors to device
        batch = {k: v.to(poi_module.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        first_batch = True

        if first_batch:
            # === DEBUG CODE ===
            print("\n=== EVAL PIPELINE DEBUG ===")
            print(f"Subject: {batch['subject']}, Vertebra: {batch['vertebra'].item()}")
            print(f"Input shape: {batch['input'].shape}")
            print(f"Input mean: {batch['input'].mean().item():.6f}")
            print(f"Input std: {batch['input'].std().item():.6f}")
            print(f"Input min: {batch['input'].min().item():.6f}")
            print(f"Input max: {batch['input'].max().item():.6f}")
            print(f"Input sum: {batch['input'].sum().item():.6f}")
            if "surface" in batch:
                print(f"Surface sum: {batch['surface'].sum().item():.6f}")
            if "offset" in batch:
                print(f"Offset: {batch['offset']}")
            print("=========================\n")
            # === END DEBUG ===
            first_batch = False

        batch = poi_module(batch)

        subject_batch = batch["subject"]
        vertebra_batch = batch["vertebra"]

        target_batch = batch["target"]
        target_indices_batch = batch["target_indices"]
        loss_mask_batch = batch["loss_mask"]

        coarse_preds_batch = batch["coarse_preds"]
        refined_preds_batch = batch["refined_preds"]

        if save_gt_proj:
            target_batch, _ = surface_project_coords(target_batch, batch["surface"])

        if project:
            refined_preds_projected_batch, _ = surface_project_coords(refined_preds_batch, batch["surface"])

        offset_batch = batch["offset"]
        poi_path_batch = batch["poi_path"]

        subreg = NII.load(batch["subreg_path"][0], seg=True)
        vertseg = NII.load(batch["msk_path"][0], seg=True)

        match = subreg.assert_affine(vertseg, raise_error=False, verbose=True)

        if not match:
            print(f"❌ Affine/Metadata stimmt nicht überein bei: \nsubreg: {subreg.affine}, \nvertseg: {vertseg.affine}")

        # Detach all tensors
        vertebra_batch = vertebra_batch.detach().cpu().numpy()
        refined_preds_batch = refined_preds_batch.detach().cpu().numpy()

        target_batch = target_batch.detach().cpu().numpy()

        if project:
            refined_preds_projected_batch = refined_preds_projected_batch.detach().cpu().numpy()

        target_indices_batch = target_indices_batch.detach().cpu().numpy()
        offset_batch = offset_batch.detach().cpu().numpy()

        loss_mask_batch = loss_mask_batch.detach().cpu().numpy()  # Alissa

        pred_batch = refined_preds_projected_batch if project else refined_preds_batch

        for sub, vert, preds, targets, indices, poi_path, offset, mask in zip(  # Alissa: mask
            subject_batch,
            vertebra_batch,
            pred_batch,
            target_batch,  # Alissa: save GT proj
            target_indices_batch,
            poi_path_batch,
            offset_batch,
            loss_mask_batch,
        ):

            # Alissa: Filter nur gültige POIs (mask == True)

            preds = preds[mask]
            indices = indices[mask]
            targets = targets[mask]

            # Open the old POI file to get the origin and rotation
            ctd = POI.load(poi_path)

            print(f"subject {sub}, vertebra {vert}")

            origin = ctd.origin
            rotation = ctd.rotation
            shape = ctd.shape
            zoom = ctd.zoom
            orientation = ctd.orientation  ### Alissa

            # Create the new POI file
            ctd = np_to_ctd(
                preds, vert, origin, rotation, idx_list=indices, shape=shape, zoom=zoom, offset=offset, orientation=orientation  ### Alissa
            )

            if save_gt_proj:
                # Create GT POI file with projected targets
                gt_proj_ctd = np_to_ctd(
                    targets,
                    vert,
                    origin,
                    rotation,
                    idx_list=indices,
                    shape=shape,
                    zoom=zoom,
                    offset=offset,
                    orientation=orientation,  ### Alissa
                )
                # Save GT projected POI file
                save_path_gt_proj = os.path.join(save_path, "gt_projections")
                os.makedirs(save_path_gt_proj, exist_ok=True)
                gt_proj_save_path = os.path.join(save_path_gt_proj, str(sub) + "_" + str(vert) + "_" + "gt_proj.json")
                gt_proj_ctd.save(gt_proj_save_path, verbose=False)

                gt_proj_global_save_path = gt_proj_save_path.replace("gt_proj.json", "gt_proj_global.json")
                gt_proj = POI.load(gt_proj_save_path).extract_region(vert)
                gt_proj.to_global().save_mrk(gt_proj_global_save_path)

            if save_in_dir:
                ctd_save_path = poi_path.replace(data_module.poi_file_ending, poi_file_ending)
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
                ctd_save_path = os.path.join(save_path, str(sub) + "_" + str(vert) + "_" + poi_file_ending)

            ctd.save(ctd_save_path, verbose=False)

            print("ctd zoom: ", ctd.zoom)

            # === (1) Speichere globale Prediction-POIs ===
            if not os.path.exists(ctd_save_path):
                print(f"⚠️ Prediction file not found: {ctd_save_path}")
                continue
            ctd_global_save_path = ctd_save_path.replace("_pred.json", "_pred_global.json")
            POI.load(ctd_save_path).to_global().save_mrk(ctd_global_save_path)

            # === (2) Speichere GT-POI-Datei ===

            gt_poi = POI.load(poi_path).extract_region(vert)

            print("gt_poi zoom: ", gt_poi.zoom)

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

                vertseg_test = NII.load(seg_vert_path, seg=True)
                print("seg zoom: ", vertseg_test.zoom)
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


def run_predictions(
    data_module_save_path,
    checkpoint_path,
    split="val",
    alternative_poi_ending=None,
    neighbor=False,
):
    # Change the ending of the POI files if necessary
    data_module = load_data_module_from_config(
        data_module_save_path,
        alternative_poi_ending=alternative_poi_ending,
    )
    data_module.setup()
    zoom = getattr(data_module, "zoom", (1, 1, 1))

    print(f"ZOOM: {zoom}")

    # Load the checkpoint
    poi_module = PoiPredictionModule.load_from_checkpoint(checkpoint_path)

    # Set the model to evaluation mode
    poi_module.eval()

    if split == "val":
        val_dl = data_module.val_dataloader()
    elif split == "test":
        val_dl = data_module.test_dataloader()
    elif split == "train":
        val_dl = data_module.train_noaug_dataloader()
    else:
        raise ValueError(f"Invalid split: {split}")

    project_gt = poi_module.hparams.refinement_config["params"]["project_gt"]
    print(f"project_gt: {project_gt}")

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
        "zoom": [],
    }

    for batch in val_dl:
        # Bring all torch tensors to device
        batch = {k: v.to(poi_module.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch = poi_module(batch)

        # Get target, coarse preds, refined preds, subject, vertebra and target indices
        subject_batch = batch["subject"]
        vertebra_batch = batch["vertebra"]

        target_batch = batch["target"]
        target_indices_batch = batch["target_indices"]
        loss_mask_batch = batch["loss_mask"]
        coarse_preds_batch = batch["coarse_preds"]
        refined_preds_batch = batch["refined_preds"]

        if neighbor:
            n_pois_per_vert_batch = batch["n_pois_per_vertebra"]

        debug = False
        # print(subject_batch, vertebra_batch, target_indices_batch)
        if "WS-16" in subject_batch and 23 in vertebra_batch:
            debug = True

        if project_gt:
            target_batch, _ = surface_project_coords(target_batch, batch["surface"], debug=False)

        coarse_preds_projected_batch, coarse_pred_proj_distances_batch = surface_project_coords(coarse_preds_batch, batch["surface"])
        refined_preds_projected_batch, refined_preds_proj_distances_batch = surface_project_coords(refined_preds_batch, batch["surface"])

        # Detach all tensors and convert to numpy
        vertebra_batch = vertebra_batch.detach().cpu().numpy()

        if neighbor:
            n_pois_per_vert_batch = n_pois_per_vert_batch.detach().cpu().numpy()  # new

        target_batch = target_batch.detach().cpu().numpy()
        target_indices_batch = target_indices_batch.detach().cpu().numpy()
        loss_mask_batch = loss_mask_batch.detach().cpu().numpy()
        coarse_preds_batch = coarse_preds_batch.detach().cpu().numpy()
        refined_preds_batch = refined_preds_batch.detach().cpu().numpy()
        coarse_preds_projected_batch = coarse_preds_projected_batch.detach().cpu().numpy()
        coarse_pred_proj_distances_batch = coarse_pred_proj_distances_batch.detach().cpu().numpy()
        refined_preds_projected_batch = refined_preds_projected_batch.detach().cpu().numpy()
        refined_preds_proj_distances_batch = refined_preds_proj_distances_batch.detach().cpu().numpy()

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

        batch_list = [
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
        ]

        if neighbor:
            keys.append("n_pois_per_vert")
            batch_list.append(n_pois_per_vert_batch)

        for values in zip(*batch_list):
            data_dict = dict(zip(keys, values))

            if neighbor:
                n_pois_per_vert = data_dict["n_pois_per_vert"]

                indices = data_dict["target_indices"][:n_pois_per_vert]
                targets = data_dict["target"][:n_pois_per_vert]
                coarse = data_dict["coarse_preds"][:n_pois_per_vert]
                refined = data_dict["refined_preds"][:n_pois_per_vert]
                coarse_proj = data_dict["coarse_preds_projected"][:n_pois_per_vert]
                refined_proj = data_dict["refined_preds_projected"][:n_pois_per_vert]
                coarse_proj_dist = data_dict["coarse_pred_proj_distances"][:n_pois_per_vert]
                refined_proj_dist = data_dict["refined_preds_proj_distances"][:n_pois_per_vert]
                loss_mask = data_dict["loss_mask"][:n_pois_per_vert]

            else:
                indices = data_dict["target_indices"]
                targets = data_dict["target"]
                coarse = data_dict["coarse_preds"]
                refined = data_dict["refined_preds"]
                coarse_proj = data_dict["coarse_preds_projected"]
                refined_proj = data_dict["refined_preds_projected"]
                coarse_proj_dist = data_dict["coarse_pred_proj_distances"]
                refined_proj_dist = data_dict["refined_preds_proj_distances"]
                loss_mask = data_dict["loss_mask"]

            # Iterate over all POIs to collect POI-wise information
            for poi_idx, t, c, r, c_proj, r_proj, c_proj_dist, r_proj_dist, l in zip(
                indices,  # data_dict["target_indices"],
                targets,  # data_dict["target"],
                coarse,  # data_dict["coarse_preds"],
                refined,  # data_dict["refined_preds"],
                coarse_proj,  # data_dict["coarse_preds_projected"],
                refined_proj,  # data_dict["refined_preds_projected"],
                coarse_proj_dist,  # data_dict["coarse_pred_proj_distances"],
                refined_proj_dist,  # data_dict["refined_preds_proj_distances"],
                loss_mask,  # data_dict["loss_mask"],
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
                pred_dict["zoom"].append(zoom)

    return pred_dict


def create_prediction_df(data_module_save_path, checkpoint_path, split="val", alternative_poi_ending=None, neighbor=False):
    pred_dict = run_predictions(data_module_save_path, checkpoint_path, split, alternative_poi_ending, neighbor)
    # Calculate distances between target and predicted POIs (in mm)
    pred_dict["coarse_error"] = [
        np.linalg.norm((np.array(t) - np.array(c)) * np.array(zoom))
        for t, c, zoom in zip(pred_dict["target"], pred_dict["coarse"], pred_dict["zoom"])
    ]
    pred_dict["refined_error"] = [
        np.linalg.norm((np.array(t) - np.array(r)) * np.array(zoom))
        for t, r, zoom in zip(pred_dict["target"], pred_dict["refined"], pred_dict["zoom"])
    ]

    # Calculate distances between target and projected POIs
    pred_dict["coarse_proj_error"] = [
        np.linalg.norm((np.array(t) - np.array(c)) * np.array(zoom))
        for t, c, zoom in zip(pred_dict["target"], pred_dict["coarse_proj"], pred_dict["zoom"])
    ]
    pred_dict["refined_proj_error"] = [
        np.linalg.norm((np.array(t) - np.array(r)) * np.array(zoom))
        for t, r, zoom in zip(pred_dict["target"], pred_dict["refined_proj"], pred_dict["zoom"])
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
    metrics_df = pd.DataFrame(columns=["Mean Error", "Median Error", "MSE", "Accuracy", "Max Error"])

    # Calculate metrics for each error type
    for error_type in [
        "coarse_error",
        "refined_error",
        "coarse_proj_error",
        "refined_proj_error",
    ]:
        metrics_df.loc[error_type] = calculate_metrics(df[error_type])

    return metrics_df

    return metrics_df


def compute_poi_wise_metrics(df, project_pred=False):
    # Group by poi_idx and calculate metrics for refined_proj_error
    if project_pred:
        grouped = df.groupby("poi_idx")["refined_proj_error"]
    else:
        grouped = df.groupby("poi_idx")["refined_error"]
    metrics_df = grouped.apply(lambda x: calculate_metrics(x)).apply(pd.Series)
    metrics_df.columns = ["Mean Error", "Median Error", "MSE", "Accuracy", "Max Error"]

    return metrics_df


def compute_vert_wise_metrics(df, project_pred=False):
    # Group by vertebra and calculate metrics for refined_proj_error
    if project_pred:
        grouped = df.groupby("vertebra")["refined_proj_error"]
    else:
        grouped = df.groupby("vertebra")["refined_error"]
    metrics_df = grouped.apply(lambda x: calculate_metrics(x)).apply(pd.Series)
    metrics_df.columns = ["Mean Error", "Median Error", "MSE", "Accuracy", "Max Error"]

    return metrics_df


def compute_sub_wise_metrics(df, project_pred=False):
    # Group by vertebra and calculate metrics for refined_proj_error
    if project_pred:
        grouped = df.groupby("subject")["refined_proj_error"]
    else:
        grouped = df.groupby("subject")["refined_error"]
    metrics_df = grouped.apply(lambda x: calculate_metrics(x)).apply(pd.Series)
    metrics_df.columns = ["Mean Error", "Median Error", "MSE", "Accuracy", "Max Error"]

    return metrics_df


def filter_high_refined_error_pois(df, threshold, project_pred=False):
    """
    filters all subjects with vertebra and poi_idx, where refined_proj_error > threshold
    """
    if project_pred:
        filtered_df = df[df["refined_proj_error"] > threshold]
        return filtered_df[["subject", "vertebra", "poi_idx", "refined_proj_error"]].reset_index(drop=True)

    else:
        filtered_df = df[df["refined_error"] > threshold]
        return filtered_df[["subject", "vertebra", "poi_idx", "refined_error"]].reset_index(drop=True)


def create_neighbor_prediction_poi_files(
    data_module_save_path,
    checkpoint_path,
    poi_file_ending,
    split="val",
    save_in_dir=False,
    save_path=None,
    return_paths=False,
    project=False,
):
    # Create the POI files for the refined predictions
    if return_paths:
        poi_paths_dict = {}
    if not save_in_dir and save_path is None:
        raise ValueError("Either save_in_dir or save_path must be set")

    # Assert that the poi_file_ending is a json file
    if not poi_file_ending.endswith(".json"):
        raise ValueError("The poi_file_ending must be a json file")

    data_module = load_data_module_from_config(data_module_save_path)
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
        batch = {k: v.to(poi_module.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch = poi_module(batch)

        # ALISSA: save img of heatmaps and global features -> für ppt
        # save_heatmaps(batch, out_dir="heatmaps")
        # save_feature_maps(batch, out_dir="features")

        subject_batch = batch["subject"]
        vertebra_batch = batch["vertebra"]
        refined_preds_batch = batch["refined_preds"]
        if project:
            refined_preds_projected_batch, _ = surface_project_coords(refined_preds_batch, batch["surface"])
        target_indices_batch = batch["target_indices"]
        offset_batch = batch["offset"]
        poi_path_batch = batch["poi_path"]

        loss_mask_batch = batch["loss_mask"]  # Alissa

        coarse_preds_batch = batch["coarse_preds"]

        n_pois_per_vert_batch = batch["n_pois_per_vertebra"]

        subreg = NII.load(batch["subreg_path"][0], seg=True)
        vertseg = NII.load(batch["msk_path"][0], seg=True)

        match = subreg.assert_affine(vertseg, raise_error=False, verbose=True)

        if not match:
            print(f"❌ Affine/Metadata stimmt nicht überein bei: \nsubreg: {subreg.affine}, \nvertseg: {vertseg.affine}")

        # Detach all tensors
        vertebra_batch = vertebra_batch.detach().cpu().numpy()
        refined_preds_batch = refined_preds_batch.detach().cpu().numpy()
        if project:
            refined_preds_projected_batch = refined_preds_projected_batch.detach().cpu().numpy()
        target_indices_batch = target_indices_batch.detach().cpu().numpy()
        offset_batch = offset_batch.detach().cpu().numpy()

        loss_mask_batch = loss_mask_batch.detach().cpu().numpy()  # Alissa

        pred_batch = refined_preds_projected_batch if project else refined_preds_batch

        n_pois_per_vert_batch = n_pois_per_vert_batch.detach().cpu().numpy()

        for sub, vert, preds, indices, poi_path, offset, mask, n_pois_per_vert in zip(  # Alissa: mask
            subject_batch,
            vertebra_batch,
            pred_batch,
            target_indices_batch,
            poi_path_batch,
            offset_batch,
            loss_mask_batch,  # Alissa
            n_pois_per_vert_batch,  # Alissa new
        ):

            # Open the old POI file to get the origin, rotation, ...
            ctd = POI.load(poi_path)

            origin = ctd.origin
            rotation = ctd.rotation
            shape = ctd.shape
            zoom = ctd.zoom
            orientation = ctd.orientation

            # current vertebra
            print(f"n_pois_per_vert: {n_pois_per_vert}")
            current_preds = preds[:n_pois_per_vert]  #
            current_indices = indices[:n_pois_per_vert]  #
            current_mask = mask[:n_pois_per_vert]  #

            current_preds = current_preds[current_mask]
            current_indices = current_indices[current_mask]

            # Create the new POI file
            current_ctd = np_to_ctd(
                current_preds,
                vert,
                origin,
                rotation,
                idx_list=current_indices,
                shape=shape,
                zoom=zoom,
                offset=offset,
                orientation=orientation,  ### Alissa
            )
            partial_centroids = []
            partial_centroids.append(
                {
                    "subject": sub,
                    "original_shape": shape,
                    "original_zoom": zoom,
                    "original_orientation": orientation,
                    "original_rotation": rotation,  # ALISSA
                    "original_origin": origin,  # ALISSA
                    "centroids": current_ctd.centroids,
                }
            )
            start_idx = n_pois_per_vert

            # top neighbor
            if vert > 1:
                top_vert = vert - 1
                top_preds = preds[start_idx : start_idx + n_pois_per_vert]  #
                top_indices = indices[start_idx : start_idx + n_pois_per_vert]  #
                top_mask = mask[start_idx : start_idx + n_pois_per_vert]

                top_preds = top_preds[top_mask]
                top_indices = top_indices[top_mask]

                top_ctd = np_to_ctd(
                    top_preds,
                    top_vert,
                    origin,
                    rotation,
                    idx_list=top_indices,
                    shape=shape,
                    zoom=zoom,
                    offset=offset,
                    orientation=orientation,  ### Alissa
                )

                partial_centroids.append(
                    {
                        "subject": sub,
                        "original_shape": shape,
                        "original_zoom": zoom,
                        "original_orientation": orientation,
                        "original_rotation": rotation,  # ALISSA
                        "original_origin": origin,  # ALISSA
                        "centroids": top_ctd.centroids,
                    }
                )

            # bottom neighbor
            start_idx += n_pois_per_vert

            if vert < 24:
                bottom_vert = vert + 1
                bottom_preds = preds[start_idx:]  #
                bottom_indices = indices[start_idx:]  #
                bottom_mask = mask[start_idx:]

                bottom_preds = bottom_preds[bottom_mask]
                bottom_indices = bottom_indices[bottom_mask]

                bottom_ctd = np_to_ctd(
                    bottom_preds,
                    bottom_vert,
                    origin,
                    rotation,
                    idx_list=bottom_indices,
                    shape=shape,
                    zoom=zoom,
                    offset=offset,
                    orientation=orientation,  ### Alissa
                )

                partial_centroids.append(
                    {
                        "subject": sub,
                        "original_shape": shape,
                        "original_zoom": zoom,
                        "original_orientation": orientation,
                        "original_rotation": rotation,  # ALISSA
                        "original_origin": origin,  # ALISSA
                        "centroids": bottom_ctd.centroids,
                    }
                )

            sub, pois = combine_centroids(partial_centroids)

            if save_in_dir:
                ctd_save_path = poi_path.replace(data_module.poi_file_ending, poi_file_ending)
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
                ctd_save_path = os.path.join(save_path, str(sub) + "_" + str(vert) + "_" + poi_file_ending)

            pois.save(ctd_save_path, verbose=False)

            # save pred as global
            if not os.path.exists(ctd_save_path):
                print(f"⚠️ Prediction file not found: {ctd_save_path}")
                continue
            ctd_global_save_path = ctd_save_path.replace("_pred.json", "_pred_global.json")
            POI.load(ctd_save_path).to_global().save_mrk(ctd_global_save_path)

            # save GT files
            all_vert = [vert]
            if vert > 1:
                all_vert.append(vert - 1)
            if vert < 24:
                all_vert.append(vert + 1)

            gt_poi = POI.load(poi_path).extract_region(*all_vert)
            gt_save_path = ctd_save_path.replace("_pred.json", "_gt.json")
            gt_poi.save(gt_save_path)

            gt_global_save_path = gt_save_path.replace("_gt.json", "_gt_global.json")
            gt_poi.to_global().save_mrk(gt_global_save_path)

            # save segmentation masks
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


def load_and_filter_csv(df):
    """
    Lädt CSV-Datei und filtert poi_idx 41-50 heraus
    """
    # CSV laden
    df

    # poi_idx 41-50 ausschließen
    excluded_poi_idx = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    df_filtered = df[~df["poi_idx"].isin(excluded_poi_idx)]

    print(f"Original Anzahl Zeilen: {len(df)}")
    print(f"Gefilterte Anzahl Zeilen: {len(df_filtered)}")
    print(f"Entfernte Zeilen: {len(df) - len(df_filtered)}")

    return df_filtered


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_module_save_path",
        type=str,
        help="Path to the saved DataModule configuration",
        default="",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the saved checkpoint",
        default="/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/src/hendrik/trainings/include_pois-cc3-exclude6/subreg-project_gt-no_freeze-surface-cc3-exclude6/version_1/checkpoints/sad-pt-epoch=122-fine_mean_distance_val=1.58.ckpt",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split to evaluate on (val/test)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to save the evaluation results",
        default="/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/src/ablation_study/hendrik/val_dummy/subreg-project_gt-no_freeze-surface/",
    )
    parser.add_argument("--neighbor", action="store_true", help="Whether neighbor predictions were made")

    parser.add_argument("--project", action="store_true", help="Whether the final predictions should be projected to the surface.")

    parser.add_argument("--save_gt_proj", action="store_true", help="Whether to save projected GT POI files.")

    args = parser.parse_args()
    args.project = True  # REMOVE
    args.save_gt_proj = True  # REMOVE

    os.makedirs(args.save_path, exist_ok=True)

    if args.data_module_save_path == "":
        t = Path(args.checkpoint_path)
        i = 0
        while i < 3:
            if t.joinpath("data_module_params.json").exists():
                args.data_module_save_path = str(t.joinpath("data_module_params.json"))
                break
            t = t.parent
            i += 1
            # Path(args.checkpoint_path).parent.parent.joinpath("data_module_params.json"))
        # args.data_module_save_path = str(Path(args.checkpoint_path).parent.parent.joinpath("data_module_params.json"))

    ### Create DataFrame with prediction information
    prediction_df = create_prediction_df(
        data_module_save_path=args.data_module_save_path, checkpoint_path=args.checkpoint_path, split=args.split, neighbor=args.neighbor
    )
    prediction_df = prediction_df[prediction_df["loss_mask"] == True]

    prediction_df.to_csv(os.path.join(args.save_path, "results.csv"))
    print("Prediction DataFrame saved")

    # prediction_df = load_and_filter_csv(prediction_df)
    # prediction_df = load_and_filter_csv(prediction_df)

    ### Compute overall metrics
    metrics_df = compute_overall_metrics(prediction_df)
    metrics_df.to_csv(os.path.join(args.save_path, "overall_metrics.csv"))
    print("Overal metrics saved")

    ### Compute POI-wise metrics
    poi_metrics_df = compute_poi_wise_metrics(prediction_df, args.project)
    poi_metrics_df.to_csv(os.path.join(args.save_path, "poi_metrics.csv"))
    print("POI-wise metrics saved")

    ### Compute vertebra-wise metrics
    vert_metrics_df = compute_vert_wise_metrics(prediction_df, args.project)
    vert_metrics_df.to_csv(os.path.join(args.save_path, "vertebra_metrics.csv"))
    print("Vertebra-wise metrics saved")

    ### Compute subject-wise metrics
    sub_metrics_df = compute_sub_wise_metrics(prediction_df, args.project)
    sub_metrics_df.to_csv(os.path.join(args.save_path, "subject_metrics.csv"))
    print("Subject-wise metrics saved")

    ### Find Outliers
    outlier_df = filter_high_refined_error_pois(prediction_df, 10, args.project)
    outlier_df.to_csv(os.path.join(args.save_path, "outliers_refined_error_higher_10.csv"))
    print("Outliers (refined_error > 10) saved")

    """
    save_path = "experiments/experiment_evaluation/k_fold/fold_6/val"

    results_df = pd.read_csv("experiments/experiment_evaluation/k_fold/fold_6/val/results.csv")

    outlier_df = filter_high_refined_error_pois(results_df, 6, False)
    outlier_df.to_csv(os.path.join(save_path, "outliers_refined_error_higher_6.csv"))
    print("Outliers (refined_error > 6) saved")

    outlier_proj_df = filter_high_refined_error_pois(results_df, 6, True)
    outlier_proj_df.to_csv(os.path.join(save_path, "outliers_refined_proj_error_higher_6.csv"))
    print("Outliers (refined_proj_error > 6) saved")
    """

    ### Create Prediction files
    prediction_files_path_proj = os.path.join(args.save_path, "prediction_files")
    prediction_files_path = os.path.join(args.save_path, "prediction_files-no_proj")
    os.makedirs(prediction_files_path, exist_ok=True)
    os.makedirs(prediction_files_path_proj, exist_ok=True)

    if args.neighbor:
        # Generate predictions and get paths
        poi_paths_dict = create_neighbor_prediction_poi_files(
            data_module_save_path=args.data_module_save_path,
            checkpoint_path=args.checkpoint_path,
            poi_file_ending="pred.json",
            split=args.split,
            save_path=prediction_files_path,
            return_paths=True,
            project=args.project,
        )
    else:
        # Generate predictions and get paths
        poi_paths_dict = create_prediction_poi_files(
            data_module_save_path=args.data_module_save_path,
            checkpoint_path=args.checkpoint_path,
            poi_file_ending="pred.json",
            split=args.split,
            save_path=prediction_files_path,
            return_paths=True,
            project=False,
            save_gt_proj=args.save_gt_proj,
        )
        poi_paths_dict = create_prediction_poi_files(
            data_module_save_path=args.data_module_save_path,
            checkpoint_path=args.checkpoint_path,
            poi_file_ending="_pred.json",
            split=args.split,
            save_path=prediction_files_path_proj,
            return_paths=True,
            project=True,
            save_gt_proj=args.save_gt_proj,
        )
    print(f"Saved predictions and ground truths to: {prediction_files_path}")
