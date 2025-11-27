"""Inference Pipeline:
Given a path to a vert and subreg segmentation mask, model and data module, this pipeline will:
1. Load the vert and subreg mask
2. Create vertebra-wise cutouts and a master_df in a temporary directory
3. Reorient and rescale the cutouts to (1,1,1) mm resolution
4. Pad the cutouts to a fixed size
5. Arrange the cutouts into a batch
6. Pass the batch through the model
7. Extract the predicted landmarks from the model output
8. Revert the predicted landmarks to the original space (remove padding, rescale, reorient, remove margin, add offset)
9. Save the predicted landmarks to a BIDS POI file (.json)
10. Delete the temporary directory
"""

import ast
import json
import os
import shutil  # For file operations


import pandas as pd
import torch
import numpy as np

from TPTBox import NII, BIDS_Global_info
from TPTBox.core.poi import POI
from torch.utils.data import Dataset

import eval as ev
from prepare_data import get_bounding_box
from utils.dataloading_utils import compute_surface, pad_array_to_shape
from utils.misc import surface_project_coords
from torch.utils.data.dataloader import default_collate


def get_subreg(container):
    subreg_query = container.new_query(flatten=True)
    subreg_query.filter_format("msk")
    subreg_query.filter_filetype("nii.gz")  # only nifti files
    subreg_query.filter("seg", "subreg")
    if not subreg_query.candidates:
        print("ERROR: No subreg candidates found!")
        return None
    subreg_candidate = subreg_query.candidates[0]

    try:
        subreg = subreg_candidate.open_nii()
        return subreg
    except Exception as e:
        print(f"Error opening subreg: {str(e)}")
        return None

def get_vertseg(container):
    vertseg_query = container.new_query(flatten=True)
    vertseg_query.filter_format("msk")
    vertseg_query.filter_filetype("nii.gz")  # only nifti files
    vertseg_query.filter("seg", "vert")
    if not vertseg_query.candidates:
        print("ERROR: No vertseg candidate found!")
        return None
    vertseg_candidate = vertseg_query.candidates[0]

    try:
        vertseg = vertseg_candidate.open_nii()
        return vertseg
    except Exception as e:
        print(f"Error opening vertseg: {str(e)}")
        return None

def get_poi(container):
    poi_query = container.new_query(flatten=True)
    poi_query.filter_format("poi")    
    if not poi_query.candidates:
        return None
    poi_candidate = poi_query.candidates[0]
    return str(poi_candidate.file["json"])


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
        assert (
            entry["original_shape"] == expected_shape
        ), "Original shapes do not match."
        assert entry["original_zoom"] == expected_zoom, "Original zooms do not match."
        assert (
            entry["original_orientation"] == expected_orientation
        ), "Original orientations do not match."

        assert np.allclose(
            entry["original_rotation"], expected_rotation, rtol=1e-10
        ), "Original rotations do not match."

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

class GruberInferenceDataset(Dataset):
    def __init__(
        self,
        master_df,
        input_shape,
        input_data_type,
        include_vert_list,
        zoom=(1, 1, 1),
        poi_indices=[
            81,
            82, 
            83, 
            84,
            85,
            86,
            87,
            88,
            89, 
            101,
            102,
            103,
            104,
            105, 
            106, 
            107, 
            108, 
            109,
            110,
            111,
            112,
            113, 
            114, 
            115, 
            116, 
            117,
            118,
            119,
            120,
            121, 
            122, 
            123, 
            124, 
            125,
            127
        ],
    ):
        self.master_df = master_df
        self.input_shape = input_shape
        self.input_data_type = input_data_type
        self.zoom = zoom
        self.poi_indices = torch.tensor(poi_indices)
        self.poi_idx_to_list_idx = {poi: idx for idx, poi in enumerate(poi_indices)}
        self.vert_idx_to_list_idx = {
            vert: idx for idx, vert in enumerate(include_vert_list)
        }

    def __len__(self):
        return len(self.master_df)

    def __getitem__(self, index):
        data_dict = {}

        # Read the row from the master dataframe
        row = self.master_df.iloc[index]
        vertebra = row["vert"]
        vert_path = row["vert_path"]
        subreg_path = row["subreg_path"]
        x_min = row["x_min"]
        y_min = row["y_min"]
        z_min = row["z_min"]

        original_orientation = row["original_orientation"]
        original_zoom = row["original_zoom"]
        original_shape = row["original_shape"]
        original_rotation = row["original_rotation"]
        original_origin = row["original_origin"] 

        preprocessed_orientation = row["preprocessed_orientation"]
        preprocessed_zoom = row["preprocessed_zoom"]
        preprocessed_rotation = row["preprocessed_rotation"]
        preprocessed_origin = row["preprocessed_origin"]
        preprocessed_shape = row["preprocessed_shape"]

        subject = row["subject"]

        subreg = NII.load(subreg_path, seg=True)
        vertseg = NII.load(vert_path, seg=True)

        assert subreg.shape == vertseg.shape
        assert subreg.orientation == vertseg.orientation
        assert subreg.orientation == ("L", "A", "S")
        assert subreg.zoom == vertseg.zoom
        assert subreg.zoom == (1, 1, 1)

        subreg = subreg.get_array()
        vertseg = vertseg.get_array()
        mask = vertseg == vertebra

        # ct = ct * mask
        subreg = subreg * mask


        ###        
        if any(s > t for s, t in zip(subreg.shape, self.input_shape)):
            print(f"Skipping subject {subject} vertebra {vertebra} (shape {subreg.shape} > {self.input_shape})")
            return None        
        elif any(s > t for s, t in zip(vertseg.shape, self.input_shape)):
            print(f"Skipping subject {subject} vertebra {vertebra} (shape {vertseg.shape} > {self.input_shape})")
            return None
        ###

        subreg, offset = pad_array_to_shape(subreg, self.input_shape)
        vertseg, _ = pad_array_to_shape(vertseg, self.input_shape)

        # Convert subreg and vertseg to tensors
        subreg = torch.from_numpy(subreg.astype(float))
        vertseg = torch.from_numpy(vertseg.astype(float))

        # Add channel dimension
        subreg = subreg.unsqueeze(0)
        vertseg = vertseg.unsqueeze(0)

        # Uses default iterations of 1, must be changed if model was trained with more iterations ("thicker" surface)
        surface = compute_surface(subreg)

        if self.input_data_type == "vertseg":
            data_dict["input"] = vertseg
        elif self.input_data_type == "subreg":
            data_dict["input"] = subreg

        data_dict["surface"] = surface
        data_dict["vertebra"] = vertebra
        data_dict["padding_offset"] = torch.tensor(offset).float()
        data_dict["poi_indices"] = self.poi_indices
        data_dict["poi_list_idx"] = torch.tensor(
            [self.poi_idx_to_list_idx[poi.item()] for poi in self.poi_indices]
        )
        data_dict["vert_list_idx"] = torch.tensor([self.vert_idx_to_list_idx[vertebra]])
        data_dict["cutout_offset"] = torch.tensor([x_min, y_min, z_min])

        data_dict["original_orientation"] = str(original_orientation)
        data_dict["original_zoom"] = original_zoom
        data_dict["original_shape"] = original_shape
        data_dict["original_rotation"] = original_rotation #ALISSA
        data_dict["original_origin"] = original_origin

        data_dict["preprocessed_orientation"] = str(preprocessed_orientation)
        data_dict["preprocessed_zoom"] = preprocessed_zoom
        data_dict["preprocessed_rotation"] = preprocessed_rotation
        data_dict["preprocessed_origin"] = preprocessed_origin
        data_dict["preprocessed_shape"] = preprocessed_shape

        data_dict["subject"] = subject

        data_dict["vert_path"] = vert_path
        data_dict["subreg_path"] = subreg_path

        return data_dict

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # All items skipped
    return default_collate(batch)

def preprocess_segmentation_masks(
    subject,
    vert_msk,
    subreg_msk,
    vert_list,
    zoom=(1, 1, 1)
):
    """
    Preprocess segmentation masks and create a master dataframe.
    """
    print(f"preprocessing subject: {subject}")  

    # Save original parameters to restore them later
    original_orientation = vert_msk.orientation
    original_zoom = vert_msk.zoom
    original_shape = vert_msk.shape
    original_rotation = vert_msk.rotation
    original_origin = vert_msk.origin 

    # Create temp directory
    temp_dir = "tmp/"
    os.makedirs(os.path.join(temp_dir, subject), exist_ok=True)

    # Get vertebrae that are both in the vert_list and in the vert mask
    msk_vert_list = vert_msk.unique()
    vertebrae = [v for v in vert_list if v in msk_vert_list]

    # Bring the masks to standard orientation. Zoom is applied AFTER cutting out the vertebrae
    vert_msk.reorient_(("L", "A", "S"))
    subreg_msk.reorient_(("L", "A", "S"))

    # Load the data array
    vertseg_arr = vert_msk.get_array()

    # Create vertebra-wise cutouts and a master_df in a temporary directory
    cutout_info = []
    for vert in vertebrae:
        # This uses the standard margin of 5 voxels around the vertebra in each direction. When the model is trained with a different margin, this should be adjusted!
        x_min, x_max, y_min, y_max, z_min, z_max = get_bounding_box(vertseg_arr, vert)

        subreg_path = os.path.join(temp_dir, subject, f"vert_{vert}-subreg.nii.gz")
        vert_path = os.path.join(temp_dir, subject, f"vert_{vert}-vertseg.nii.gz")

        subreg_cropped = subreg_msk.apply_crop(
            ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
        )

        vert_cropped = vert_msk.apply_crop(
            ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
        )

        # rescale the cutouts to zoom mm resolution
        vert_cropped.rescale_(zoom)
        subreg_cropped.rescale_(zoom)

        vert_cropped.save(vert_path, verbose=False)
        subreg_cropped.save(subreg_path, verbose=False)

        # Get preprocessed parameters
        preprocessed_origin = vert_cropped.origin
        preprocessed_rotation = vert_cropped.rotation
        preprocessed_orientation = vert_cropped.orientation
        preprocessed_zoom = vert_cropped.zoom
        preprocessed_shape = vert_cropped.shape

        # Save the slice indices as json to reconstruct the original POI file (there probably is a more BIDS-like approach to this)
        cutout_info.append(
            {
                "subject": subject,
                "vert": vert,
                "x_min": int(x_min),
                "x_max": int(x_max),
                "y_min": int(y_min),
                "y_max": int(y_max),
                "z_min": int(z_min),
                "z_max": int(z_max),
                "vert_path": vert_path,
                "subreg_path": subreg_path,

                "preprocessed_orientation": preprocessed_orientation,
                "preprocessed_zoom": preprocessed_zoom,
                "preprocessed_rotation": preprocessed_rotation,
                "preprocessed_origin": preprocessed_origin,
                "preprocessed_shape": preprocessed_shape,

                "original_orientation": original_orientation,
                "original_zoom": original_zoom,
                "original_shape": original_shape,
                "original_rotation": original_rotation,  
                "original_origin": original_origin,  
            }
        )

    # Read the cutout info into a DataFrame
    master_df = pd.DataFrame(cutout_info)

    # Save the master_df to a csv file
    master_df_path = os.path.join(temp_dir, subject, "cutout_df.csv")
    master_df.to_csv(master_df_path, index=False)

    return master_df, temp_dir

def create_prediction_poi_files(
    subject,
    vert_msk,
    subreg_msk,
    dm_path,
    model_path,
    save_dir,
    project_to_surface=False,
):
    # Load data module parameters
    dm_params = json.load(open(dm_path, "r"))
    input_shape = dm_params["input_shape"]
    input_data_type = dm_params["input_data_type"]
    vert_list = dm_params["include_vert_list"]
    poi_indices = dm_params["include_poi_list"]
    zoom = dm_params.get("zoom", (1, 1, 1))

    # preprocess segmentation masks and then save the info in a master_df ( create a /tmp)
    master_df, temp_dir = preprocess_segmentation_masks(subject, vert_msk, subreg_msk, vert_list, zoom)
    
    print(f"inferencing subject: {subject}")
    # get data_module and create dataset
    ds = GruberInferenceDataset(
        master_df, input_shape=input_shape, input_data_type=input_data_type, include_vert_list=vert_list, zoom=zoom
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,  collate_fn=safe_collate)

    # load checkpoint
    model = ev.load_model_from_checkpoint(model_path)

    partial_centroids = []
    # predict POIs
    for batch in dl:

        if batch is None:
            continue

        # Bring all tensors to device
        batch = {
            k: v.to(model.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        batch = model(batch)

        refined_preds_batch = batch["refined_preds"]

        if project_to_surface:
            refined_preds_projected_batch, _ = surface_project_coords(
                refined_preds_batch, batch["surface"]
            )
            pred_coords = refined_preds_projected_batch.squeeze().detach().cpu().numpy()
        else:
            pred_coords = refined_preds_batch.squeeze().detach().cpu().numpy()

        # Extract batch information
        padding_offset = batch["padding_offset"].squeeze().detach().cpu().numpy()
        vertebra = batch["vertebra"].squeeze().detach().cpu().numpy()
        poi_indices = batch["poi_indices"].squeeze().detach().cpu().numpy()
        cutout_offset = batch["cutout_offset"].squeeze().detach().cpu().numpy()
        subject = batch["subject"][0]
        
        # Get the preprocessed parameters
        preprocessed_rotation = batch["preprocessed_rotation"][0].detach().cpu().numpy() #ALISSA
        preprocessed_orientation = ast.literal_eval(batch["preprocessed_orientation"][0])
        preprocessed_zoom = (
            batch["preprocessed_zoom"][0][0].item(),
            batch["preprocessed_zoom"][1][0].item(),
            batch["preprocessed_zoom"][2][0].item(),
        )
        preprocessed_origin = (
            batch["preprocessed_origin"][0][0].item(),
            batch["preprocessed_origin"][1][0].item(),
            batch["preprocessed_origin"][2][0].item(),
        )
        preprocessed_shape = (
            batch["preprocessed_shape"][0][0].item(),
            batch["preprocessed_shape"][1][0].item(),
            batch["preprocessed_shape"][2][0].item(),
        )

        # Get the original parameters
        original_rotation = batch["original_rotation"][0].detach().cpu().numpy() 
        original_orientation = ast.literal_eval(batch["original_orientation"][0])
        original_zoom = (
            batch["original_zoom"][0][0].item(),
            batch["original_zoom"][1][0].item(),
            batch["original_zoom"][2][0].item(),
        )
        original_origin = (
            batch["original_origin"][0][0].item(),
            batch["original_origin"][1][0].item(),
            batch["original_origin"][2][0].item(),
        )
        original_shape = (
            batch["original_shape"][0][0].item(),
            batch["original_shape"][1][0].item(),
            batch["original_shape"][2][0].item(),
        )
 
        # get segmentation mask path
        vert_path = batch["vert_path"][0]      
        subreg_path = batch["subreg_path"][0]

        # Create the new POI file
        unpadded_refined_preds_ctd = ev.np_to_ctd(
            pred_coords,
            vertebra=vertebra.item(),
            origin=preprocessed_origin,
            rotation= preprocessed_rotation, 
            idx_list=poi_indices,
            shape=preprocessed_shape,
            zoom=preprocessed_zoom,
            offset=padding_offset,
            orientation=preprocessed_orientation
        )

        subject_dir = os.path.join(save_dir, str(subject), "cutouts-preproccessed")
        os.makedirs(subject_dir, exist_ok=True)

        # save POI and Segmentation masks (cutouts)
        ctd_save_path = os.path.join(
            subject_dir, str(subject) + "_" + str(vertebra) + "_pred.json"
        )
        ctd_global_save_path = ctd_save_path.replace("_pred.json", "_pred_global.json")

        unpadded_refined_preds_ctd.save(ctd_save_path, verbose=False)
        unpadded_refined_preds_ctd_poi = POI.load(ctd_save_path)
        unpadded_refined_preds_ctd_poi.to_global().save_mrk(ctd_global_save_path)
        
        # copy segmentation masks
        vertseg_save_path = ctd_save_path.replace("_pred.json", "_vertseg.nii.gz")
        subreg_save_path = ctd_save_path.replace("_pred.json", "_subreg.nii.gz")

        if os.path.exists(vert_path):
            shutil.copy(vert_path, vertseg_save_path)
        else:
            print(f"⚠️ Segmentation file not found: {vert_path}")

        if os.path.exists(subreg_path):
            shutil.copy(subreg_path, subreg_save_path)
        else:
            print(f"⚠️ Segmentation file not found: {subreg_path}")

        
        # TODO: combine centroids (rescale, add cutoutoffset and reorient to original space)
        unpadded_refined_preds_ctd.rescale_(original_zoom)

        new_centroids = {}
        for v, p_idx, c in unpadded_refined_preds_ctd.centroids.items():
            new_coords = c + cutout_offset
            new_centroids[(v, p_idx)] = (new_coords[0], new_coords[1], new_coords[2])

        unpadded_refined_preds_ctd.centroids = new_centroids

        unpadded_refined_preds_ctd.reorient_(original_orientation) 

        partial_centroids.append(
            {
                "subject": subject,
                "original_shape": original_shape,
                "original_zoom": original_zoom,
                "original_orientation": original_orientation,
                "original_rotation": original_rotation, #ALISSA 
                "original_origin": original_origin, #ALISSA
                "centroids": unpadded_refined_preds_ctd.centroids,
            }
        )
    
    sub, pois = combine_centroids(partial_centroids)

    
    pois.save(os.path.join(save_dir, sub, "poi_predicted.json"))
    pois.to_global().save_mrk(os.path.join(save_dir, sub, "poi_predicted_global.json"))

    #vert_msk_path
    vert_msk.save(os.path.join(save_dir, sub, "vertseg.nii.gz"))



if __name__ == "__main__":
    
    #bgi = BIDS_Global_info(
    #    datasets=["/home/student/alissa/3dVertPois/src/predictions/dataset-myelom"],
    #    parents=["derivatives"],
    #)

    #bgi = BIDS_Global_info(
    #    datasets=["/home/student/alissa/3dVertPois/src/dataset/data_preprocessing/dataset-folder-test"],
    #    parents=["derivatives"],
    #)

    bgi = BIDS_Global_info(
        datasets=["/home/student/alissa/3dVertPois/src/dataset/data_preprocessing/dataset-verse19"],
        parents=["derivatives"],
    )

    save_dir = "/home/student/alissa/3dVertPois/src/predictions/verse19-inferenced-LAS/subreg-project_gt-no_freeze-SADenseNet-standard_architecture-excel_outliers_exclude"
    #dm_path = "ablation_study/dataloader/training/include_pois/subreg-project_gt-no_freeze-standard_architecture-excel_outliers_exclude/version_0/data_module_params.json"
    #model_path = "ablation_study/dataloader/training/include_pois/subreg-project_gt-no_freeze-standard_architecture-excel_outliers_exclude/version_0/checkpoints/sad-pt-epoch=74-fine_mean_distance_val=1.77.ckpt"

    dm_path = "ablation_study/architecture/training/subreg-no_project_gt-no_freeze-standard_architecture-excel_outliers_exclude/version_0/data_module_params.json"
    model_path = "ablation_study/architecture/training/subreg-no_project_gt-no_freeze-standard_architecture-excel_outliers_exclude/version_0/checkpoints/sad-pt-epoch=60-fine_mean_distance_val=1.76.ckpt"
    
    #subjects_inferenced = 0

    for sub, container in bgi.enumerate_subjects():
        print(f"Subject: {sub}")
        #if subjects_inferenced >= 10:
        #    print(f"10 Subjects have been inferenced. Break.")
        #    break

        vert_msk = get_vertseg(container)
        subreg_msk = get_subreg(container)

        #gt_poi_path = get_poi(container)

        if vert_msk is None or subreg_msk is None:
            print(f"Skip Subject: {sub} - not all data available")
            continue

        if vert_msk.shape != subreg_msk.shape:
            print(f"Skip Subject: {sub} - vertseg {vert_msk.shape} and subreg {subreg_msk.shape} shapes don't match")
            continue

        if vert_msk.orientation != subreg_msk.orientation:
            print(f"Skip Subject: {sub} - vertseg {vert_msk.orientation} and subreg {subreg_msk.orientation} orientations don't match")
            continue

        if vert_msk.orientation != ("L", "A", "S"):
            print(f"Skip Subject: {sub} - vertseg orientation {vert_msk.orientation} is not LAS")
            continue

        create_prediction_poi_files(
            sub,
            vert_msk,
            subreg_msk,
            dm_path,
            model_path,
            save_dir,
            project_to_surface=False,
        ) 

        #subjects_inferenced += 1
    

