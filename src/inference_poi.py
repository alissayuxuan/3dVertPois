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
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd
import torch
import numpy as np

from TPTBox import NII, BIDS_Global_info, BIDS_FILE, np_utils, v_idx_order
from TPTBox.core.poi import POI
from TypeSaveArgParse import Class_to_ArgParse
from torch.utils.data import Dataset

import eval as ev
from prepare_data import get_bounding_box
from utils.dataloading_utils import compute_surface, pad_array_to_shape, get_ct, get_subreg, get_vertseg, get_vertseg_bfile
from utils.misc import surface_project_coords
from torch.utils.data.dataloader import default_collate
from eval import combine_centroids
from model_sel import TrainedModelInfo

poi_flip_pairs = {
    # These are the middle points, i.e. the ones that are not flipped
    81: 81,
    101: 101,
    103: 103,
    102: 102,
    104: 104,
    105: 105,  #
    106: 106,  #
    107: 107,  #
    108: 108,  #
    125: 125,
    127: 127,
    # Flipped left to right
    83: 82,
    84: 85,
    86: 87,
    88: 89,
    109: 117,
    111: 119,
    110: 118,
    112: 120,
    113: 121,  #
    114: 122,  #
    115: 123,  #
    116: 124,  #
    # Flipped right to left
    82: 83,
    85: 84,
    87: 86,
    89: 88,
    117: 109,
    118: 110,
    119: 111,
    120: 112,
    121: 113,
    122: 114,
    123: 115,
    124: 116,
}


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
            127,
        ],
    ):
        self.master_df = master_df
        self.input_shape = input_shape
        self.input_data_type = input_data_type
        self.zoom = zoom
        self.poi_indices = torch.tensor(poi_indices)
        self.poi_idx_to_list_idx = {poi: idx for idx, poi in enumerate(poi_indices)}
        self.vert_idx_to_list_idx = {vert: idx for idx, vert in enumerate(include_vert_list)}

    def __len__(self):
        return len(self.master_df)

    def __getitem__(self, index):
        data_dict = {}

        # Read the row from the master dataframe
        row = self.master_df.iloc[index]
        vertebra = row["vert"]
        vert_path = row["vert_path"]
        input_data_path = row["input_data_path"]
        surface_path = row["surface_path"]
        # subreg_path = row["subreg_path"]
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

        if self.input_data_type == "ct":
            input_data = NII.load(input_data_path, seg=False)
            input_data.normalize_ct(min_out=0, max_out=1, inplace=True)  # Here correct??
        else:
            input_data = NII.load(input_data_path, seg=True)
        vertseg = NII.load(vert_path, seg=True)
        surface = NII.load(surface_path, seg=True)
        surface.extract_label_(surface.unique())

        assert input_data.shape == vertseg.shape
        assert input_data.orientation == vertseg.orientation
        assert input_data.orientation == ("L", "A", "S")
        assert input_data.zoom == vertseg.zoom
        # assert subreg.zoom == (1, 1, 1)

        print("zoom in __getitem__: ", input_data.zoom)

        input_data = input_data.get_array()
        vertseg = vertseg.get_array()
        surface = surface.get_array()
        mask = vertseg == vertebra

        # ct = ct * mask
        input_data = input_data * mask
        surface = surface * mask

        ###
        if any(s > t for s, t in zip(input_data.shape, self.input_shape)):
            print(f"Skipping subject {subject} vertebra {vertebra} (shape {input_data.shape} > {self.input_shape})")
            return None
        elif any(s > t for s, t in zip(vertseg.shape, self.input_shape)):
            print(f"Skipping subject {subject} vertebra {vertebra} (shape {vertseg.shape} > {self.input_shape})")
            return None
        ###

        input_data, offset = pad_array_to_shape(input_data, self.input_shape)
        vertseg, _ = pad_array_to_shape(vertseg, self.input_shape)
        surface, _ = pad_array_to_shape(surface, self.input_shape)

        # Convert subreg and vertseg to tensors
        input_data = torch.from_numpy(input_data.astype(float))
        vertseg = torch.from_numpy(vertseg.astype(float))
        surface = torch.from_numpy(surface.astype(float))

        # Add channel dimension
        input_data = input_data.unsqueeze(0)
        vertseg = vertseg.unsqueeze(0)
        surface = surface.unsqueeze(0)

        # Uses default iterations of 1, must be changed if model was trained with more iterations ("thicker" surface)
        # surface = compute_surface(input_data)

        # if self.input_data_type == "vertseg":
        #    data_dict["input"] = vertseg
        # elif self.input_data_type == "subreg":
        #    data_dict["input"] = input_data
        ## elif self.input_data_type == "ct":
        ##    data_dict["input"] = ct
        # elif self.input_data_type == "surface_msk":
        #    data_dict["input"] = surface

        data_dict["input"] = input_data
        data_dict["surface"] = surface
        data_dict["vertebra"] = vertebra
        data_dict["padding_offset"] = torch.tensor(offset).float()
        data_dict["poi_indices"] = self.poi_indices
        data_dict["poi_list_idx"] = torch.tensor([self.poi_idx_to_list_idx[poi.item()] for poi in self.poi_indices])
        data_dict["vert_list_idx"] = torch.tensor([self.vert_idx_to_list_idx[vertebra]])
        data_dict["cutout_offset"] = torch.tensor([x_min, y_min, z_min])

        data_dict["original_orientation"] = str(original_orientation)
        data_dict["original_zoom"] = original_zoom
        data_dict["original_shape"] = original_shape
        data_dict["original_rotation"] = original_rotation  # ALISSA
        data_dict["original_origin"] = original_origin

        data_dict["preprocessed_orientation"] = str(preprocessed_orientation)
        data_dict["preprocessed_zoom"] = preprocessed_zoom
        data_dict["preprocessed_rotation"] = preprocessed_rotation
        data_dict["preprocessed_origin"] = preprocessed_origin
        data_dict["preprocessed_shape"] = preprocessed_shape

        data_dict["subject"] = subject

        data_dict["vert_path"] = vert_path
        data_dict["input_data_path"] = input_data_path
        # data_dict["subreg_path"] = subreg_path

        return data_dict


class GruberInferenceNeighborDataset(Dataset):
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
            127,
        ],
    ):
        self.master_df = master_df
        self.input_shape = input_shape
        self.input_data_type = input_data_type
        self.zoom = zoom
        self.poi_indices = torch.tensor(poi_indices)
        self.poi_idx_to_list_idx = {poi: idx for idx, poi in enumerate(poi_indices)}
        self.include_vert_list = include_vert_list
        self.vert_idx_to_list_idx = {vert: idx for idx, vert in enumerate(include_vert_list)}

    def __len__(self):
        return len(self.master_df)

    def __getitem__(self, index):
        data_dict = {}

        # Read the row from the master dataframe
        row = self.master_df.iloc[index]
        vertebra = row["vert"]
        vert_path = row["vert_path"]
        input_data_path = row["input_data_path"]
        surface_path = row["surface_path"]
        # subreg_path = row["subreg_path"]
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

        if self.input_data_type == "ct":
            input_data = NII.load(input_data_path, seg=False)
            input_data.normalize_ct(min_out=0, max_out=1, inplace=True)  # Here correct??
        else:
            input_data = NII.load(input_data_path, seg=True)
        vertseg = NII.load(vert_path, seg=True)
        surface = NII.load(surface_path, seg=True)
        surface.extract_label_(surface.unique())

        assert input_data.shape == vertseg.shape
        assert input_data.orientation == vertseg.orientation
        assert input_data.orientation == ("L", "A", "S")
        assert input_data.zoom == vertseg.zoom
        # assert subreg.zoom == (1, 1, 1)

        print("zoom in __getitem__: ", input_data.zoom)

        input_data = input_data.get_array()
        vertseg = vertseg.get_array()
        surface = surface.get_array()

        # extract neighbors
        v_idx = self.include_vert_list.index(vertebra)
        neighbor_top = self.include_vert_list[v_idx - 1] if v_idx > 0 else 0  # 0 = dummy (no top/bottom neighbor)
        neighbor_bottom = self.include_vert_list[v_idx + 1] if v_idx < len(self.include_vert_list) - 1 else 0
        # neighbor_top = neighbor_top if neighbor_top in vertseg else 0
        # neighbor_bottom = neighbor_bottom if neighbor_bottom in vertseg else 0

        all_vert = [
            ("current", vertebra),
            ("top", neighbor_top),
            ("bottom", neighbor_bottom),
        ]
        actual_vert = [vert for _, vert in all_vert if vert != 0]
        # TODO: t13 (28) and t11 (18-20)

        mask = np.isin(vertseg, actual_vert)

        # ct = ct * mask
        input_data = input_data * mask
        surface = surface * mask

        ###
        if any(s > t for s, t in zip(input_data.shape, self.input_shape)):
            print(f"Skipping subject {subject} vertebra {vertebra} (shape {input_data.shape} > {self.input_shape})")
            return None
        elif any(s > t for s, t in zip(vertseg.shape, self.input_shape)):
            print(f"Skipping subject {subject} vertebra {vertebra} (shape {vertseg.shape} > {self.input_shape})")
            return None
        ###

        input_data, offset = pad_array_to_shape(input_data, self.input_shape)
        vertseg, _ = pad_array_to_shape(vertseg, self.input_shape)
        surface, _ = pad_array_to_shape(surface, self.input_shape)

        # Convert subreg and vertseg to tensors
        input_data = torch.from_numpy(input_data.astype(float))
        vertseg = torch.from_numpy(vertseg.astype(float))
        surface = torch.from_numpy(surface.astype(float))

        # Add channel dimension
        input_data = input_data.unsqueeze(0)
        vertseg = vertseg.unsqueeze(0)
        surface = surface.unsqueeze(0)

        # Uses default iterations of 1, must be changed if model was trained with more iterations ("thicker" surface)
        # surface = compute_surface(input_data)

        # if self.input_data_type == "vertseg":
        #    data_dict["input"] = vertseg
        # elif self.input_data_type == "subreg":
        #    data_dict["input"] = input_data
        ## elif self.input_data_type == "ct":
        ##    data_dict["input"] = ct
        # elif self.input_data_type == "surface_msk":
        #    data_dict["input"] = surface

        data_dict["input"] = input_data
        data_dict["surface"] = surface
        data_dict["vertebra"] = vertebra
        data_dict["padding_offset"] = torch.tensor(offset).float()
        #
        data_dict["poi_indices"] = self.poi_indices
        data_dict["poi_indices"] = torch.cat([data_dict["poi_indices"], data_dict["poi_indices"], data_dict["poi_indices"]])
        data_dict["poi_list_idx"] = torch.tensor([self.poi_idx_to_list_idx[poi.item()] for poi in self.poi_indices])
        data_dict["poi_list_idx"] = torch.cat([data_dict["poi_list_idx"], data_dict["poi_list_idx"], data_dict["poi_list_idx"]])
        #
        data_dict["vert_list_idx"] = torch.tensor([self.vert_idx_to_list_idx[vertebra]])
        data_dict["cutout_offset"] = torch.tensor([x_min, y_min, z_min])

        data_dict["original_orientation"] = str(original_orientation)
        data_dict["original_zoom"] = original_zoom
        data_dict["original_shape"] = original_shape
        data_dict["original_rotation"] = original_rotation  # ALISSA
        data_dict["original_origin"] = original_origin

        data_dict["preprocessed_orientation"] = str(preprocessed_orientation)
        data_dict["preprocessed_zoom"] = preprocessed_zoom
        data_dict["preprocessed_rotation"] = preprocessed_rotation
        data_dict["preprocessed_origin"] = preprocessed_origin
        data_dict["preprocessed_shape"] = preprocessed_shape

        data_dict["subject"] = subject

        data_dict["vert_path"] = vert_path
        data_dict["input_data_path"] = input_data_path
        # data_dict["subreg_path"] = subreg_path

        # print the shape of all tensors in the data_dict
        # for i, k in data_dict.items():
        #    if isinstance(k, torch.Tensor):
        #        print(i, k.shape)

        return data_dict


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # All items skipped
    return default_collate(batch)


def preprocess_segmentation_masks(
    subject,
    vert_msk: NII,
    input_data: NII | None,
    input_data_type: str,
    vert_list,
    zoom=(1, 1, 1),
    tmp_name: str = "tmp",
):
    """
    Preprocess segmentation masks and create a master dataframe.
    """
    print(f"preprocessing subject: {subject}")
    print("preprocess_seg - zoom: ", zoom)

    # Save original parameters to restore them later
    original_orientation = vert_msk.orientation
    original_zoom = vert_msk.zoom
    original_shape = vert_msk.shape
    original_rotation = vert_msk.rotation
    original_origin = vert_msk.origin

    print(
        "Original msk meta",
        original_orientation,
        original_zoom,
        original_shape,
        original_rotation,
        original_origin,
    )

    # Create temp directory
    temp_dir = f"/DATA/NAS/ongoing_projects/hendrik/poi_prediction/tmp/{tmp_name}/"
    # os.path.join("tmp", tmp_name)
    os.makedirs(os.path.join(temp_dir, subject), exist_ok=True)

    # Get vertebrae that are both in the vert_list and in the vert mask
    msk_vert_list = vert_msk.unique()
    vertebrae = sorted([v for v in vert_list if v in msk_vert_list], key=lambda x: v_idx_order.index(x))

    # Bring the masks to standard orientation. Zoom is applied AFTER cutting out the vertebrae
    vert_msk = vert_msk.reorient(("L", "A", "S"), verbose=True).map_labels_({50: 49}, verbose=False)
    original_zoom = vert_msk.zoom  # update original zoom after reorientation
    original_shape = vert_msk.shape  # update original shape after reorientation
    if input_data is not None:
        input_data.reorient_(("L", "A", "S"))

    # Load the data array
    vertseg_arr = vert_msk.get_array()

    # Create vertebra-wise cutouts and a master_df in a temporary directory
    cutout_info = []
    first = False  # Change to True for printing first info only
    for vert in vertebrae:

        # rescale the cutouts to zoom mm resolution
        vert_msk.rescale_(zoom, verbose=first)
        if input_data is not None:
            input_data.resample_from_to_(vert_msk, verbose=False)
        # This uses the standard margin of 5 voxels around the vertebra in each direction. When the model is trained with a different margin, this should be adjusted!
        com = np_utils.np_center_of_mass(vert_msk.extract_label(vert).get_seg_array())[1]
        # x_min, x_max, y_min, y_max, z_min, z_max = get_bounding_box(vertseg_arr, vert)

        input_data_path = os.path.join(temp_dir, subject, f"vert_{vert}-input.nii.gz")
        vert_path = os.path.join(temp_dir, subject, f"vert_{vert}-vertseg.nii.gz")
        surface_path = os.path.join(temp_dir, subject, f"vert_{vert}-surface.nii.gz")

        _, crop, padding = np_utils.np_calc_crop_around_centerpoint(
            (round(com[0]), round(com[1]), round(com[2])),
            vert_msk.get_seg_array(),
            cutout_size=(128, 128, 144),
        )
        # PAD HERE
        # REMOVE FROM OFFSET
        if input_data is not None:
            # input_data_cropped = input_data.apply_crop(ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max)))
            input_data_cropped = input_data.apply_crop(crop).apply_pad(padding, verbose=False)
            # print("Cropped input data from shape", input_data.shape, "to", input_data_cropped.shape) if first else None

        # vert_cropped = vert_msk.apply_crop(ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max)))
        vert_cropped = vert_msk.apply_crop(crop).apply_pad(padding, verbose=False)
        # urface_cropped = surface.apply_crop(ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max)))
        print("Cropped from shape", vert_msk.shape, "to", vert_cropped.shape) if first else None
        x_min, x_max, y_min, y_max, z_min, z_max = crop[0].start, crop[0].stop, crop[1].start, crop[1].stop, crop[2].start, crop[2].stop
        x_min -= padding[0][0]
        y_min -= padding[1][0]
        z_min -= padding[2][0]
        # input_data_cropped.rescale_(zoom)

        # if inference_flipped:
        #    vert_cropped.flip(vert_cropped.get_axis("R"), inplace=True)

        vert_cropped.save(vert_path, verbose=False)
        # input_data_cropped.save(input_data_path, verbose=False)
        surface = vert_cropped.compute_surface_mask(connectivity=3)
        surface.save(surface_path, verbose=False)
        if input_data is not None:
            input_data_cropped.save(input_data_path, verbose=False)

        if input_data_type == "vertseg":
            input_data_path = vert_path
        elif input_data_type == "surface_msk":
            input_data_path = surface_path

        # Get preprocessed parameters
        preprocessed_origin = vert_cropped.origin
        preprocessed_rotation = vert_cropped.rotation
        preprocessed_orientation = vert_cropped.orientation
        preprocessed_zoom = vert_cropped.zoom
        preprocessed_shape = vert_cropped.shape

        (
            print(
                "Preprocessed msk meta",
                preprocessed_orientation,
                preprocessed_zoom,
                preprocessed_shape,
                preprocessed_rotation,
                preprocessed_origin,
            )
            if first
            else None
        )
        first = False

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
                "input_data_path": input_data_path,
                "surface_path": surface_path,
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
    vert_msk_ref: BIDS_FILE,
    # vert_msk,
    # subreg_msk,
    dm_path,
    model_path,
    vert_out,
    poi_out,
    poi_global_out,
    project_to_surface=False,
    inference_flipped=False,
):
    # Load data module parameters
    dm_params = json.load(open(dm_path, "r"))
    print(dm_params)
    input_shape = dm_params["input_shape"]
    input_data_type = dm_params["input_data_type"]
    vert_list = sorted(dm_params["include_vert_list"], key=lambda x: v_idx_order.index(x))
    poi_indices = dm_params["include_poi_list"]
    zoom = dm_params.get("zoom", (1, 1, 1))
    print("zoom: ", zoom)

    vert_msk = vert_msk_ref.open_nii()
    vert_list = [v for v in vert_list if v in vert_msk.unique()]
    split_info = vert_msk_ref.info["split"] if "split" in vert_msk_ref.info else None
    if input_data_type == "vertseg":
        input_data = vert_msk
    elif input_data_type == "subreg":
        input_data = get_subreg(container, split=split_info)
    elif input_data_type == "ct":
        input_data = get_ct(container, split=split_info)
    elif input_data_type == "surface_msk":
        # input_data = None  # vert_msk.compute_surface_mask(connectivity=3, dilated_surface=False)
        input_data = vert_msk.compute_surface_mask(connectivity=3, dilated_surface=False)
    else:
        raise ValueError(f"Unknown input data type: {input_data_type}")

    if vert_msk is None or input_data is None:
        print(f"Skip Subject: {subject} - not all data available")
        return

    if vert_msk.shape != input_data.shape:
        print(f"Skip Subject: {subject} - vertseg {vert_msk.shape} and input_data {input_data.shape} shapes don't match")
        return

    # preprocess segmentation masks and then save the info in a master_df ( create a /tmp)
    master_df, temp_dir = preprocess_segmentation_masks(
        subject,
        vert_msk,
        input_data,
        input_data_type,
        vert_list,
        zoom,
        tmp_name="tmp_" + vert_out.name,
    )

    print(f"inferencing subject: {subject}")
    use_neighbor = dm_params["dataset"] == "GruberNeighbor"
    # get data_module and create dataset
    if use_neighbor:
        ds = GruberInferenceNeighborDataset(
            master_df, input_shape=input_shape, input_data_type=input_data_type, include_vert_list=vert_list, zoom=zoom
        )
    else:
        ds = GruberInferenceDataset(
            master_df, input_shape=input_shape, input_data_type=input_data_type, include_vert_list=vert_list, zoom=zoom
        )
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=safe_collate)

    # load checkpoint
    model = ev.load_model_from_checkpoint(model_path)

    partial_centroids = []
    # predict POIs
    first = False  # Change for printing first info only
    for batch in dl:

        if batch is None:
            continue

        # Bring all tensors to device
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if inference_flipped:
            flip_axis = input_data.get_axis("L") + 2  # batch, channel
            batch["input"] = torch.flip(batch["input"], dims=[flip_axis])  # flip along the sagittal axis

        batch = model(batch)

        refined_preds_batch = batch["refined_preds"]
        if inference_flipped:
            # idx = [slice(None)] * refined_preds_batch.ndim
            # idx[flip_axis] = 0
            backflip_axis = flip_axis - 2
            # refined_preds_batch = torch.flip(refined_preds_batch, dims=[flip_axis])  # flip back
            refined_preds_batch[:, :, backflip_axis] = (
                batch["input"].shape[flip_axis] - refined_preds_batch[:, :, backflip_axis]
            )  # flip x-coordinates back
            # TODO: need to flip LABELS as well

        # if use_neighbor:
        #    n_poi = refined_preds_batch.shape[1] // 3
        # refined_preds_batch = refined_preds_batch[:, n_poi : n_poi * 2, :]
        #    refined_preds_batch = refined_preds_batch[:, :n_poi, :]

        # assert refined_preds_batch.shape[1] == 35, f"wrong refined_preds_batch shape {refined_preds_batch.shape}"

        if project_to_surface:
            refined_preds_projected_batch, _ = surface_project_coords(refined_preds_batch, batch["surface"], requires_filling=True)
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
        preprocessed_rotation = batch["preprocessed_rotation"][0].detach().cpu().numpy()  # ALISSA
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
        # subreg_path = batch["subreg_path"][0]
        input_data_path = batch["input_data_path"][0]

        # Create the new POI file
        print("np_to_ctd input-zoom: ", preprocessed_zoom)
        unpadded_refined_preds_ctd: POI = ev.np_to_ctd(
            pred_coords[:35, :],
            vertebra=vertebra.item(),
            origin=preprocessed_origin,
            rotation=preprocessed_rotation,
            idx_list=poi_indices[:35],
            shape=preprocessed_shape,
            zoom=preprocessed_zoom,
            offset=padding_offset,
            orientation=preprocessed_orientation,
        )
        if use_neighbor:
            unpadded_refined_preds_ctd_top = ev.np_to_ctd(
                pred_coords[35:70, :],
                vertebra=vertebra.item() - 1,
                origin=preprocessed_origin,
                rotation=preprocessed_rotation,
                idx_list=poi_indices[35:70],
                shape=preprocessed_shape,
                zoom=preprocessed_zoom,
                offset=padding_offset,
                orientation=preprocessed_orientation,
            )
            unpadded_refined_preds_ctd_bot = ev.np_to_ctd(
                pred_coords[70:, :],
                vertebra=vertebra.item() + 1,
                origin=preprocessed_origin,
                rotation=preprocessed_rotation,
                idx_list=poi_indices[70:],
                shape=preprocessed_shape,
                zoom=preprocessed_zoom,
                offset=padding_offset,
                orientation=preprocessed_orientation,
            )
            unpadded_cutout_poi = unpadded_refined_preds_ctd.join_left(unpadded_refined_preds_ctd_top)
            unpadded_cutout_poi = unpadded_cutout_poi.join_left(unpadded_refined_preds_ctd_bot)
        else:
            unpadded_cutout_poi = unpadded_refined_preds_ctd
        print("unpadded_refined_preds_ctd: ", unpadded_refined_preds_ctd) if first else None

        if inference_flipped:
            # swap ids in unpadded_cutout_poi
            unpadded_cutout_poi.map_labels_(label_map_subregion=poi_flip_pairs)
            if use_neighbor:  # <- I have no idea why
                unpadded_refined_preds_ctd.map_labels_(label_map_subregion=poi_flip_pairs)

        # subject_dir = os.path.join(save_dir, str(subject), "cutouts-preproccessed")
        # os.makedirs(subject_dir, exist_ok=True)
        subject_dir: Path = vert_out.parent.joinpath("cutouts-preproccessed")
        subject_dir.mkdir(parents=True, exist_ok=True)

        # save POI and Segmentation masks (cutouts)
        ctd_save_path = os.path.join(subject_dir, str(subject) + "_" + str(vertebra) + "_pred.json")
        ctd_global_save_path = ctd_save_path.replace("_pred.json", "_pred_global.json")

        unpadded_cutout_poi.save(ctd_save_path, verbose=True)
        unpadded_refined_preds_ctd_poi = POI.load(ctd_save_path)
        unpadded_refined_preds_ctd_poi.to_global().save_mrk(ctd_global_save_path)

        # copy segmentation masks
        vertseg_save_path = ctd_save_path.replace("_pred.json", "_vertseg.nii.gz")
        input_data_save_path = ctd_save_path.replace("_pred.json", f"_{input_data_type}.nii.gz")
        # subreg_save_path = ctd_save_path.replace("_pred.json", "_subreg.nii.gz")

        if os.path.exists(vert_path):
            shutil.copy(vert_path, vertseg_save_path)
        else:
            print(f"⚠️ Segmentation file not found: {vert_path}")

        if os.path.exists(input_data_path):
            shutil.copy(input_data_path, input_data_save_path)
        else:
            print(f"⚠️ Segmentation file not found: {input_data_path}")

        unpadded_refined_preds_ctd.rescale_(original_zoom, verbose=first)

        # TODO: combine centroids (rescale, add cutoutoffset and reorient to original space)
        new_centroids = {}
        for v, p_idx, c in unpadded_refined_preds_ctd.centroids.items():
            new_coords = c + cutout_offset
            new_centroids[(v, p_idx)] = (new_coords[0], new_coords[1], new_coords[2])
        unpadded_refined_preds_ctd.centroids = new_centroids
        unpadded_refined_preds_ctd.shape = original_shape
        #

        unpadded_refined_preds_ctd.reorient_(original_orientation, verbose=first)
        first = False

        partial_centroids.append(
            {
                "subject": subject,
                "original_shape": unpadded_refined_preds_ctd.shape,
                "original_zoom": unpadded_refined_preds_ctd.zoom,
                "original_orientation": unpadded_refined_preds_ctd.orientation,
                "original_rotation": original_rotation,  # ALISSA
                "original_origin": original_origin,  # ALISSA
                "centroids": unpadded_refined_preds_ctd.extract_region(vertebra).centroids,
            }
        )

    sub, pois = combine_centroids(partial_centroids)

    # os.makedirs(os.path.join(save_dir, subject), exist_ok=True)
    # pois.save(os.path.join(save_dir, sub, "poi_predicted.json"))
    # pois.to_global().save_mrk(os.path.join(save_dir, sub, "poi_predicted_global.json"))

    # vert_msk_path
    # vert_msk.save(os.path.join(save_dir, sub, "vertseg.nii.gz"))

    pois.save(poi_out)  # os.path.join(save_dir, sub, "poi_predicted.json"))
    pois.to_global().save_mrk(
        poi_global_out,
        split_by_region=True,
    )  # os.path.join(save_dir, sub, "poi_predicted_global.json"))

    # vert_msk_path
    vert_msk.save(vert_out)  # os.path.join(save_dir, sub, "vertseg.nii.gz"))

    # Clear the temporary directory
    os.system(f"rm -r {temp_dir}")


@dataclass
class InferenceConfig(Class_to_ArgParse):
    datasets: list[str] = field(
        default_factory=lambda: [
            "dataset-verse19training_1mmiso",
            # "dataset-verse20training_1mmiso",
            # "dataset-verse19validation_1mmiso",
            # "dataset-verse20validation_1mmiso",
            # "dataset-verse19test_1mmiso",
            # "dataset-verse20test_1mmiso",
        ]
    )
    der_msk: str = "derivatives_combined"
    der_out_base: str = "derivatives_poi_"
    model_info: TrainedModelInfo = TrainedModelInfo.T_N_SURFACE
    inference_flipped: bool = True
    project_to_surface: bool = False


if __name__ == "__main__":

    # bgi = BIDS_Global_info(
    #    datasets=["/home/student/alissa/3dVertPois/src/dataset/data_preprocessing/dataset-folder-test"],
    #    parents=["derivatives"],
    # )
    opt = InferenceConfig().get_opt()
    ds_names = opt.datasets
    DER_MSK = opt.der_msk

    model_info = opt.model_info  # TrainedModelInfo.GRUBER_S_SURFACE
    model_dir = model_info.model_dir

    #
    project_to_surface = opt.project_to_surface

    # LOOP
    for ds_name in ds_names:
        ds_path = f"/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/{ds_name}/"
        bgi = BIDS_Global_info(
            datasets=[ds_path],
            parents=[DER_MSK],
        )

        der_out = f"{opt.der_out_base}{model_dir}-v{model_info.version}"
        if opt.inference_flipped:
            der_out += "_flipped"
        if project_to_surface:
            der_out += "_sproj"
        # dm_path = "ablation_study/dataloader/training/include_pois/subreg-project_gt-no_freeze-standard_architecture-excel_outliers_exclude/version_0/data_module_params.json"
        # model_path = "ablation_study/dataloader/training/include_pois/subreg-project_gt-no_freeze-standard_architecture-excel_outliers_exclude/version_0/checkpoints/sad-pt-epoch=74-fine_mean_distance_val=1.77.ckpt"

        dm_path = model_info.data_module_params_path
        model_path = model_info.model_path

        if not model_path.endswith(".ckpt"):
            model_path += ".ckpt"

        inference_subjects = 0

        for sub, container in bgi.enumerate_subjects():
            # if "004" not in sub:
            #    continue
            print(f"Subject: {sub}")

            vert_msk_refs: BIDS_FILE = get_vertseg_bfile(container)
            for vert_msk_ref in vert_msk_refs:
                if not vert_msk_ref.exists():
                    continue

                # if "split" not in vert_msk_ref.info:
                #    continue
                # subreg_msk = get_subreg(container)
                subreg_msk_p = vert_msk_ref.get_changed_path(info={"seg": "subreg"}, parent=vert_msk_ref.parent)
                if not subreg_msk_p.exists():
                    print(f"Skip Subject: {sub} - subreg file not found for vertseg {vert_msk_ref.path}")
                    continue
                # gt_poi_path = get_poi(container)

                if vert_msk_ref is None:  # or subreg_msk is None:
                    print(f"Skip Subject: {sub} - not all data available")
                    continue

                vert_out = vert_msk_ref.get_changed_path(info={"seg": "vert"}, parent=der_out)
                poi_out = vert_msk_ref.get_changed_path(
                    bids_format="poi", info={"mod": "ct", "seg": "vert"}, file_type="json", parent=der_out
                )
                poi_global_out = vert_msk_ref.get_changed_path(
                    bids_format="poi", info={"mod": "ct", "seg": "vert", "space": "global"}, file_type="mrk.json", parent=der_out
                )

                if poi_global_out.exists() and poi_out.exists():
                    print(f"Skip Subject: {sub} - output POI files already exist")
                    continue

                try:
                    vert_msk = vert_msk_ref.open_nii()
                    subreg_msk = NII.load(subreg_msk_p, seg=True)
                except Exception as e:
                    print(f"Error opening vertseg: {str(e)}")
                    vert_msk = None

                if vert_msk is None:  # or subreg_msk is None:
                    print(f"Skip Subject: {sub} - not all data available")
                    continue
                # if vert_msk.shape != subreg_msk.shape:
                #    print(f"Skip Subject: {sub} - vertseg {vert_msk.shape} and subreg {subreg_msk.shape} shapes don't match")
                #    continue

                vert_msk.assert_affine(other=subreg_msk)
                # print("Original msk meta", vert_msk)

                create_prediction_poi_files(
                    sub,
                    vert_msk_ref,
                    # vert_msk,
                    # subreg_msk,
                    dm_path,
                    model_path,
                    vert_out,
                    poi_out,
                    poi_global_out,
                    project_to_surface=project_to_surface,
                    inference_flipped=opt.inference_flipped,
                )
