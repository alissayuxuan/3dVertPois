import ast
import os

import torch
import numpy as np
#from BIDS import NII, POI
from TPTBox import NII
from TPTBox.core.poi import POI
from torch.utils.data import Dataset

from transforms.transforms import Compose, LandMarksRandHorizontalFlip # was src.transforms.transforms
from utils.dataloading_utils import compute_surface, get_gt_pois, pad_array_to_shape


class PoiDataset(Dataset):
    def __init__(
        self,
        master_df,
        poi_indices,
        include_vert_list,
        poi_flip_pairs=None,
        input_data_type="subreg",
        input_shape=(128, 128, 96),
        transforms=None,
        flip_prob=0.5,
        include_com=False,
        poi_file_ending="poi.json",
        iterations=1,
    ):

        # If master_df has a column use_sample, filter on it
        if "use_sample" in master_df.columns:
            master_df = master_df[master_df["use_sample"]]
        self.master_df = master_df
        self.input_data_type = input_data_type
        self.input_shape = input_shape
        self.poi_indices = poi_indices
        self.transform = Compose(transforms) if transforms else None
        if flip_prob > 0:
            self.transform = Compose(
                [self.transform, LandMarksRandHorizontalFlip(flip_prob, poi_flip_pairs)]
            )
        self.include_com = include_com
        self.poi_flip_pairs = poi_flip_pairs
        self.flip_prob = flip_prob
        self.poi_file_ending = poi_file_ending
        self.poi_idx_to_list_idx = {poi: idx for idx, poi in enumerate(poi_indices)}
        self.vert_idx_to_list_idx = {
            vert: idx for idx, vert in enumerate(include_vert_list)
        }
        self.iterations = iterations

    def __len__(self):
        return len(self.master_df)

    def __getitem__(self, index):
        data_dict = {}

        # Read the row from the master dataframe
        row = self.master_df.iloc[index]
        subject = row["subject"]
        vertebra = row["vertebra"]
        file_dir = row["file_dir"]

        # If the master_dir has a column bad_poi_list, use this to create a loss mask
        if "bad_poi_list" in self.master_df.columns:
            bad_poi_list = ast.literal_eval(row["bad_poi_list"])
            bad_poi_list = [int(poi) for poi in bad_poi_list]
            bad_poi_list = torch.tensor(bad_poi_list)
        else:
            bad_poi_list = torch.tensor([], dtype=torch.int)

        # Get the paths
        ct_path = os.path.join(file_dir, "ct.nii.gz")
        msk_path = os.path.join(file_dir, "vertseg.nii.gz")
        subreg_path = os.path.join(file_dir, "subreg.nii.gz")
        surface_msk_path = os.path.join(file_dir, "surface_msk.nii.gz")
        poi_path = os.path.join(file_dir, self.poi_file_ending)

        # Load the BIDS objects
        ct = NII.load(ct_path, seg = False)
        subreg = NII.load(subreg_path, seg=True)
        vertseg = NII.load(msk_path, seg=True)
        surface_msk = NII.load(surface_msk_path, seg=True)
        poi = POI.load(poi_path)

        zoom = (1, 1, 1)

        ct.rescale_and_reorient_(
            axcodes_to=('L', 'A', 'S'), voxel_spacing = zoom, verbose = False
        )
        subreg.rescale_and_reorient_(
            axcodes_to=("L", "A", "S"), voxel_spacing=zoom, verbose=False
        )
        vertseg.rescale_and_reorient_(
            axcodes_to=("L", "A", "S"), voxel_spacing=zoom, verbose=False
        )
        surface_msk.rescale_and_reorient_(
            axcodes_to=("L", "A", "S"), voxel_spacing=zoom, verbose=False
        )
        poi.reorient_(axcodes_to=("L", "A", "S"), verbose=False).rescale_(
            zoom, verbose=False
        )

        #TODO: muss ich CT scans normalisieren?
        ct.normalize_ct(min_out=0, max_out=1, inplace=True)


        # Get the ground truth POIs
        poi, missing_pois = get_gt_pois(poi, vertebra, self.poi_indices)

        poi_indices = torch.tensor(self.poi_indices)

        # Get arrays
        ct = ct.get_array()
        subreg = subreg.get_array()
        vertseg = vertseg.get_array()
        surface_msk = surface_msk.get_array()

        mask = vertseg == vertebra

        ct = ct * mask
        subreg = subreg * mask
        vertseg = vertseg * mask
        surface_msk = surface_msk * mask
        

        subreg, offset = pad_array_to_shape(subreg, self.input_shape)
        vertseg, _ = pad_array_to_shape(vertseg, self.input_shape)
        surface_msk, _ = pad_array_to_shape(surface_msk, self.input_shape)
        ct, _ = pad_array_to_shape(ct, self.input_shape)

        poi = poi + torch.tensor(offset)

        # Convert subreg, vertseg and surface_msk to tensors
        ct = torch.from_numpy(ct.astype(float))
        subreg = torch.from_numpy(subreg.astype(float))
        vertseg = torch.from_numpy(vertseg.astype(float))
        surface_msk = torch.from_numpy(surface_msk.astype(float))

        # Add channel dimension
        ct = ct.unsqueeze(0)
        subreg = subreg.unsqueeze(0)
        vertseg = vertseg.unsqueeze(0)
        surface_msk = surface_msk.unsqueeze(0)

        if self.input_data_type == "vertseg":  
            data_dict["input"] = vertseg  
        elif self.input_data_type == "subreg":   
            data_dict["input"] = subreg  
        elif self.input_data_type == "ct":  
            data_dict["input"] = ct
        elif self.input_data_type == "surface_msk":
            data_dict["input"] = surface_msk

        #data_dict["input"] = vertseg#subreg
        data_dict["target"] = poi
        data_dict["target_indices"] = poi_indices

        data_dict = self.transform(data_dict) if self.transform else data_dict

        # Identify pois outside of the input shape
        max_x = self.input_shape[0] - 1
        max_y = self.input_shape[1] - 1
        max_z = self.input_shape[2] - 1

        outside_poi_indices = (
            (data_dict["target"][:, 0] < 0)
            | (data_dict["target"][:, 0] > max_x)
            | (data_dict["target"][:, 1] < 0)
            | (data_dict["target"][:, 1] > max_y)
            | (data_dict["target"][:, 2] < 0)
            | (data_dict["target"][:, 2] > max_z)
        )

        # Create a loss mask for pois shifted oustide of the image due to augmentation,
        # missing pois from the ground truth and bad pois
        loss_mask = torch.ones_like(data_dict["target"][:, 0])
        loss_mask[outside_poi_indices] = 0
        bad_poi_list_idx = [
            self.poi_idx_to_list_idx[bad_poi.item()]
            for bad_poi in bad_poi_list
            if bad_poi.item() in self.poi_indices
        ]
        loss_mask[bad_poi_list_idx] = 0
        missing_poi_list_idx = [
            self.poi_idx_to_list_idx[missing_poi.item()] for missing_poi in missing_pois
        ]
        loss_mask[missing_poi_list_idx] = 0

        data_dict["loss_mask"] = loss_mask.bool()

        transformed_mask = data_dict["input"] > 0
        surface = compute_surface(transformed_mask, iterations=self.iterations)

        data_dict["surface"] = surface
        data_dict["subject"] = str(subject)
        data_dict["vertebra"] = vertebra
        data_dict["zoom"] = torch.tensor(zoom).float()
        data_dict["offset"] = torch.tensor(offset).float()
        data_dict["ct_path"] = ct_path
        data_dict["msk_path"] = msk_path
        data_dict["subreg_path"] = subreg_path
        data_dict["poi_path"] = poi_path
        data_dict["poi_list_idx"] = torch.tensor(
            [self.poi_idx_to_list_idx[poi.item()] for poi in poi_indices]
        )
        data_dict["vert_list_idx"] = torch.tensor([self.vert_idx_to_list_idx[vertebra]])

        return data_dict


    
    def ex_getitem(self, index):
        data_dict = {}

        # Read the row from the master dataframe
        row = self.master_df.iloc[index]
        subject = row["subject"]
        vertebra = row["vertebra"]
        file_dir = row["file_dir"]

        # Get the paths
        ct_path = os.path.join(file_dir, "ct.nii.gz")
        msk_path = os.path.join(file_dir, "vertseg.nii.gz")
        subreg_path = os.path.join(file_dir, "subreg.nii.gz")
        surface_msk_path = os.path.join(file_dir, "surface_msk.nii.gz")
        poi_path = os.path.join(file_dir, self.poi_file_ending)

        # Load the BIDS objects
        ct = NII.load(ct_path, seg = False)
        subreg = NII.load(subreg_path, seg=True)
        vertseg = NII.load(msk_path, seg=True)
        surface_msk = NII.load(surface_msk_path, seg=True)
        poi = POI.load(poi_path)

        zoom = (1, 1, 1)

        ct.rescale_and_reorient_(
            axcodes_to=('L', 'A', 'S'), voxel_spacing = zoom, verbose = False
        )
        subreg.rescale_and_reorient_(
            axcodes_to=("L", "A", "S"), voxel_spacing=zoom, verbose=False
        )
        vertseg.rescale_and_reorient_(
            axcodes_to=("L", "A", "S"), voxel_spacing=zoom, verbose=False
        )
        surface_msk.rescale_and_reorient_(
            axcodes_to=("L", "A", "S"), voxel_spacing=zoom, verbose=False
        )
        poi.reorient_(axcodes_to=("L", "A", "S"), verbose=False).rescale_(
            zoom, verbose=False
        )

        ct.normalize_ct(min_out=0, max_out=1, inplace=True)

        # 0 as dummy vertebra
        neighbor_top = vertebra - 1 if vertebra > 1 else 0
        neighbor_bottom = vertebra + 1 if vertebra < 24 else 0 

        all_vertebrae = [vertebra]
        if neighbor_top != 0:
            all_vertebrae.append(neighbor_top)
        if neighbor_bottom != 0:
            all_vertebrae.append(neighbor_bottom)


        current_poi, current_missing_pois = get_gt_pois(poi, vertebra, self.poi_indices)
        top_poi, top_missing_pois = get_gt_pois(poi, neighbor_top, self.poi_indices)
        bottom_poi, bottom_missing_pois = get_gt_pois(poi, neighbor_bottom, self.poi_indices)

        poi_indices = torch.tensor(self.poi_indices) #???

        # Get arrays
        ct = ct.get_array()
        subreg = subreg.get_array()
        vertseg = vertseg.get_array()
        surface_msk = surface_msk.get_array()

        mask = np.isin(vertseg, all_vertebrae)

        ct = ct * mask
        subreg = subreg * mask
        vertseg = vertseg * mask
        surface_msk = surface_msk * mask
        

        subreg, offset = pad_array_to_shape(subreg, self.input_shape)
        vertseg, _ = pad_array_to_shape(vertseg, self.input_shape)
        surface_msk, _ = pad_array_to_shape(surface_msk, self.input_shape)
        ct, _ = pad_array_to_shape(ct, self.input_shape)

        #poi = poi + torch.tensor(offset) TODO: how to handle the offset?

        # Convert subreg, vertseg and surface_msk to tensors
        ct = torch.from_numpy(ct.astype(float))
        subreg = torch.from_numpy(subreg.astype(float))
        vertseg = torch.from_numpy(vertseg.astype(float))
        surface_msk = torch.from_numpy(surface_msk.astype(float))

        # Add channel dimension
        ct = ct.unsqueeze(0)
        subreg = subreg.unsqueeze(0)
        vertseg = vertseg.unsqueeze(0)
        surface_msk = surface_msk.unsqueeze(0)

        if self.input_data_type == "vertseg":  
            data_dict["input"] = vertseg  
        elif self.input_data_type == "subreg":   
            data_dict["input"] = subreg  
        elif self.input_data_type == "ct":  
            data_dict["input"] = ct
        elif self.input_data_type == "surface_msk":
            data_dict["input"] = surface_msk

        #data_dict["input"] = vertseg#subreg
        #data_dict["target"] = poi
        #data_dict["target_indices"] = poi_indices

        data_dict = self.transform(data_dict) if self.transform else data_dict

        

        # get bad_poi_list
        if "bad_poi_list" in self.master_df.columns:
            current_bad_poi_list = ast.literal_eval(row["bad_poi_list"])
            current_bad_poi_list = [int(poi) for poi in current_bad_poi_list]
            current_bad_poi_list = torch.tensor(current_bad_poi_list)

            if neighbor_top != 0:
                top_row = self.master_df[
                    (self.master_df["subject"] == subject) & 
                    (self.master_df["vertebra"] == neighbor_top)
                ]
                
                top_bad_poi_list = ast.literal_eval(top_row.iloc[0]["bad_poi_list"])
                top_bad_poi_list = [int(poi) for poi in top_bad_poi_list]
                top_bad_poi_list = torch.tensor(top_bad_poi_list)
            else:
                top_bad_poi_list = torch.tensor([], dtype=torch.int)
            
            if neighbor_bottom != 0:
                bottom_row = self.master_df[
                    (self.master_df["subject"] == subject) & 
                    (self.master_df["vertebra"] == neighbor_bottom)
                ]

                bottom_bad_poi_list = ast.literal_eval(bottom_row.iloc[0]["bad_poi_list"])
                bottom_bad_poi_list = [int(poi) for poi in bottom_bad_poi_list]
                bottom_bad_poi_list = torch.tensor(bottom_bad_poi_list)

            else: 
                bottom_bad_poi_list = torch.tensor([], dtype=torch.int)
                    
        else:
            current_bad_poi_list = torch.tensor([], dtype=torch.int)
            top_bad_poi_list = torch.tensor([], dtype=torch.int)
            bottom_bad_poi_list = torch.tensor([], dtype=torch.int)

        current_loss_mask = torch.ones_like(current_poi) # is it the same as: loss_mask = torch.ones_like(data_dict["target"][:, 0]) ?
        curent_bad_poi_list_idx = [
            self.poi_idx_to_list_idx[bad_poi.item()]
            for bad_poi in current_bad_poi_list
            if bad_poi.item() in self.poi_indices
        ]
        current_loss_mask[curent_bad_poi_list_idx] = 0
        current_missing_poi_list_idx = [
            self.poi_idx_to_list_idx[missing_poi.item()] for missing_poi in current_missing_pois
        ]
        current_loss_mask[current_missing_poi_list_idx] = 0

        # TODO: repeat for top and bottom neighbor

        # TODO: combine results: combine_pois (if neighbor == 0, add dummy neighbor) and combine_loss_mask

        # TODO: add offset

        # TODO: set data_dict

        # TODO: identify any outside of input shape pois

        # Identify pois outside of the input shape
        max_x = self.input_shape[0] - 1
        max_y = self.input_shape[1] - 1
        max_z = self.input_shape[2] - 1

        outside_poi_indices = (
            (data_dict["target"][:, 0] < 0)
            | (data_dict["target"][:, 0] > max_x)
            | (data_dict["target"][:, 1] < 0)
            | (data_dict["target"][:, 1] > max_y)
            | (data_dict["target"][:, 2] < 0)
            | (data_dict["target"][:, 2] > max_z)
        )

        data_dict["loss_mask"] = loss_mask.bool()

        transformed_mask = data_dict["input"] > 0
        surface = compute_surface(transformed_mask, iterations=self.iterations)

        data_dict["surface"] = surface
        data_dict["subject"] = str(subject)
        data_dict["vertebra"] = vertebra
        data_dict["zoom"] = torch.tensor(zoom).float()
        data_dict["offset"] = torch.tensor(offset).float()
        data_dict["ct_path"] = ct_path
        data_dict["msk_path"] = msk_path
        data_dict["subreg_path"] = subreg_path
        data_dict["poi_path"] = poi_path
        data_dict["poi_list_idx"] = torch.tensor(
            [self.poi_idx_to_list_idx[poi.item()] for poi in poi_indices]
        )
        data_dict["vert_list_idx"] = torch.tensor([self.vert_idx_to_list_idx[vertebra]])

        return data_dict


class ImplantsDataset(PoiDataset):
    def __init__(
        self,
        master_df,
        input_shape=(128, 128, 96),
        transforms=None,
        flip_prob=0.5,
        include_com=False,
        include_poi_list=None,
        include_vert_list=None,
        poi_file_ending="poi.json",
        iterations=1,
    ):
        super().__init__(
            master_df,
            poi_indices=(
                include_poi_list
                if include_poi_list
                else (
                    [90, 91, 92, 93, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
                    if include_com
                    else [90, 91, 92, 93]
                )
            ),
            include_vert_list=(
                include_vert_list
                if include_vert_list
                else [
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                ]
            ),
            poi_flip_pairs={
                90: 91,
                91: 90,
                92: 93,
                93: 92,
                94: 95,
                95: 94,
                # Center of mass is not flipped
                41: 41,
                42: 42,
                43: 43,
                44: 44,
                45: 45,
                46: 46,
                47: 47,
                48: 48,
                49: 49,
                50: 50,
                0: 0,
            },
            input_shape=input_shape,
            transforms=transforms,
            flip_prob=flip_prob,
            include_com=include_com,
            poi_file_ending=poi_file_ending,
            iterations=iterations,
        )


class GruberDataset(PoiDataset):
    def __init__(
        self,
        master_df,
        input_data_type="subreg",
        input_shape=(128, 128, 96),
        transforms=None,
        flip_prob=0.5,
        include_com=False,
        include_poi_list=None,
        include_vert_list=None,
        poi_file_ending="poi.json",
        iterations=1,
    ):
        super().__init__(
            master_df,
            poi_indices=(
                include_poi_list
                if include_poi_list
                else (
                    [
                        81,
                        82, #
                        83, 
                        84,
                        85,
                        86,
                        87,
                        88,
                        89, #
                        101,
                        102,
                        103,
                        104,
                        105, #
                        106, #
                        107, #
                        108, #
                        109,
                        110,
                        111,
                        112,
                        113, #
                        114, #
                        115, #
                        116, #
                        117,
                        118,
                        119,
                        120,
                        121, #
                        122, #
                        123, #
                        124, #
                        125,
                        127,

                        41,
                        42,
                        43,
                        44,
                        45,
                        46,
                        47,
                        48,
                        49,
                        50,
                    ]
                    if include_com
                    else [
                        81,
                        82, #
                        83, 
                        84,
                        85,
                        86,
                        87,
                        88,
                        89, #
                        101,
                        102,
                        103,
                        104,
                        105, #
                        106, #
                        107, #
                        108, #
                        109,
                        110,
                        111,
                        112,
                        113, #
                        114, #
                        115, #
                        116, #
                        117,
                        118,
                        119,
                        120,
                        121, #
                        122, #
                        123, #
                        124, #
                        125,
                        127,
                    ]
                )
            ),
            include_vert_list=(
                include_vert_list
                if include_vert_list
                else [
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                ]
            ),
            poi_flip_pairs={
                # These are the middle points, i.e. the ones that are not flipped
                81: 81,
                101: 101,
                103: 103,
                102: 102,
                104: 104,
                105: 105, #
                106: 106, #
                107: 107, #
                108: 108, #
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
                113: 121, #
                114: 122, #
                115: 123, #
                116: 124, #
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
                # Center of mass, does not need to be flipped
                # TODO: Passt das so??? geht das auch wenn include_com=false ist und die POIs gar nicht definiert sind?
                41: 41,
                42: 42,
                43: 43,
                44: 44,
                45: 45,
                46: 46,
                47: 47,
                48: 48,
                49: 49,
                50: 50,
                0: 0,
            },
            input_data_type=input_data_type,
            input_shape=input_shape,
            transforms=transforms,
            flip_prob=flip_prob,
            include_com=include_com,
            poi_file_ending=poi_file_ending,
            iterations=iterations,
        )


class JointDataset(PoiDataset):
    def __init__(
        self,
        master_df,
        input_shape=(128, 128, 96),
        transforms=None,
        flip_prob=0.5,
        include_poi_list=None,
        include_vert_list=None,
        poi_file_ending="poi.json",
    ):
        super().__init__(
            master_df,
            poi_indices=(
                include_poi_list
                if include_poi_list
                else [
                    81,
                    101,
                    102,
                    103,
                    104,
                    109,
                    110,
                    111,
                    112,
                    117,
                    118,
                    119,
                    120,
                    125,
                    127,
                    134,
                    136,
                    141,
                    142,
                    143,
                    144,
                    149,
                    151,
                    90,
                    91,
                    92,
                    93,
                ]
            ),
            include_vert_list=(
                include_vert_list
                if include_vert_list
                else [
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                ]
            ),
            poi_flip_pairs={
                # These are the middle points, i.e. the ones that are not flipped
                81: 81,
                101: 101,
                103: 103,
                102: 102,
                104: 104,
                125: 125,
                127: 127,
                134: 134,
                136: 136,
                # Flipped left to right
                109: 117,
                111: 119,
                110: 118,
                112: 120,
                149: 141,
                151: 143,
                142: 144,
                # Flipped right to left
                117: 109,
                119: 111,
                118: 110,
                120: 112,
                141: 149,
                143: 151,
                144: 142,
                # Center of mass, does not need to be flipped
                41: 41,
                42: 42,
                43: 43,
                44: 44,
                45: 45,
                46: 46,
                47: 47,
                48: 48,
                49: 49,
                50: 50,
                0: 0,
                # Implants
                90: 91,
                91: 90,
                92: 93,
                93: 92,
                94: 95,
                95: 94,
            },
            input_shape=input_shape,
            transforms=transforms,
            flip_prob=flip_prob,
            poi_file_ending=poi_file_ending,
        )


class PoiNeighborDataset(Dataset):
    def __init__(
        self,
        master_df,
        poi_indices,
        include_vert_list,
        poi_flip_pairs=None,
        input_data_type="subreg",
        input_shape=(120, 121, 149),
        transforms=None,
        flip_prob=0.0,
        include_com=False,
        poi_file_ending="poi.json",
        iterations=1,
    ):

        # If master_df has a column use_sample, filter on it
        if "use_sample" in master_df.columns:
            master_df = master_df[master_df["use_sample"]]
        self.master_df = master_df
        self.input_data_type = input_data_type
        self.input_shape = input_shape
        self.poi_indices = poi_indices
        self.transform = Compose(transforms) if transforms else None
        if flip_prob > 0:
            self.transform = Compose(
                [self.transform, LandMarksRandHorizontalFlip(flip_prob, poi_flip_pairs)]
            )
        self.include_com = include_com
        self.poi_flip_pairs = poi_flip_pairs
        self.flip_prob = flip_prob
        self.poi_file_ending = poi_file_ending
        self.poi_idx_to_list_idx = {poi: idx for idx, poi in enumerate(poi_indices)}
        self.vert_idx_to_list_idx = {
            vert: idx for idx, vert in enumerate(include_vert_list)
        }
        self.iterations = iterations

    def __len__(self):
        return len(self.master_df)

    def _process_single_vert(self, poi, subject, vertebra):
        """
        Processes a single vertebra and returns its POIs and loss mask.
        """
        pois, missing_pois = get_gt_pois(poi, vertebra, self.poi_indices)

        loss_mask = torch.ones(len(self.poi_indices), dtype=torch.float)

        missing_poi_list_idx = [
            self.poi_idx_to_list_idx[missing_poi.item()] for missing_poi in missing_pois
        ]
        loss_mask[missing_poi_list_idx] = 0 

        if "bad_poi_list" not in self.master_df.columns or vertebra == 0:
            bad_poi_list = torch.tensor([], dtype=torch.int)

        else:
            vert_row = self.master_df[
                (self.master_df["subject"] == subject) &
                (self.master_df["vertebra"] == vertebra)
            ]   

            if len(vert_row) == 0:
                bad_poi_list = torch.tensor([], dtype=torch.int)
                #print(f"\n\n\nEMPTY VERT_ROW\nvertebra: {vertebra}\nsubject: {subject}\n\n\n")

            else:
                bad_poi_list = ast.literal_eval(vert_row.iloc[0]["bad_poi_list"])
                bad_poi_list = [int(poi) for poi in bad_poi_list]
                bad_poi_list = torch.tensor(bad_poi_list)

        bad_poi_list_idx = [
            self.poi_idx_to_list_idx[bad_poi.item()]
            for bad_poi in bad_poi_list
            if bad_poi.item() in self.poi_indices
        ]
        loss_mask[bad_poi_list_idx] = 0

        return pois, loss_mask
    
    def __getitem__(self, index):
        data_dict = {}

        # Read the row from the master dataframe
        row = self.master_df.iloc[index]
        subject = row["subject"]
        vertebra = row["vertebra"]
        file_dir = row["file_dir"]

        # Get the paths
        ct_path = os.path.join(file_dir, "ct.nii.gz")
        msk_path = os.path.join(file_dir, "vertseg.nii.gz")
        subreg_path = os.path.join(file_dir, "subreg.nii.gz")
        surface_msk_path = os.path.join(file_dir, "surface_msk.nii.gz")
        poi_path = os.path.join(file_dir, self.poi_file_ending)

        # Load the BIDS objects
        ct = NII.load(ct_path, seg = False)
        subreg = NII.load(subreg_path, seg=True)
        vertseg = NII.load(msk_path, seg=True)
        surface_msk = NII.load(surface_msk_path, seg=True)
        poi = POI.load(poi_path)

        zoom = (1, 1, 1)

        ct.rescale_and_reorient_(
            axcodes_to=('L', 'A', 'S'), voxel_spacing = zoom, verbose = False
        )
        subreg.rescale_and_reorient_(
            axcodes_to=("L", "A", "S"), voxel_spacing=zoom, verbose=False
        )
        vertseg.rescale_and_reorient_(
            axcodes_to=("L", "A", "S"), voxel_spacing=zoom, verbose=False
        )
        surface_msk.rescale_and_reorient_(
            axcodes_to=("L", "A", "S"), voxel_spacing=zoom, verbose=False
        )
        poi.reorient_(axcodes_to=("L", "A", "S"), verbose=False).rescale_(
            zoom, verbose=False
        )

        ct.normalize_ct(min_out=0, max_out=1, inplace=True)

        # Define neighbor vertebrae
        neighbor_top = vertebra - 1 if vertebra > 1 else 0 # 0 = dummy (no top/bottom neighbor)
        neighbor_bottom = vertebra + 1 if vertebra < 24 else 0 

        all_vert = [
            ("current", vertebra),
            ("top", neighbor_top),
            ("bottom", neighbor_bottom)
        ]

        # Filter out dummy for seg mask
        actual_vert = [vert for _, vert in all_vert if vert != 0]

        # Get arrays
        ct = ct.get_array()
        subreg = subreg.get_array()
        vertseg = vertseg.get_array()
        surface_msk = surface_msk.get_array()

        mask = np.isin(vertseg, actual_vert)

        # mask vert with neighbors
        ct = ct * mask
        subreg = subreg * mask
        vertseg = vertseg * mask
        surface_msk = surface_msk * mask
        
        # Padding and offset
        subreg, offset = pad_array_to_shape(subreg, self.input_shape)
        vertseg, _ = pad_array_to_shape(vertseg, self.input_shape)
        surface_msk, _ = pad_array_to_shape(surface_msk, self.input_shape)
        ct, _ = pad_array_to_shape(ct, self.input_shape)


        # process each vertebra separately
        all_pois = []
        all_loss_masks = []

        for label, vert in all_vert:
            #print(f"vertebra: {vert}")
            if vert == 0: # dummy
                vert_pois = torch.full((len(self.poi_indices), 3), -1)
                vert_loss_mask = torch.zeros(len(self.poi_indices), dtype=torch.float)
                #print(f"  Created DUMMY: pois shape {vert_pois.shape}, mask shape {vert_loss_mask.shape}")

            else: # real
                vert_pois, vert_loss_mask = self._process_single_vert(poi, subject, vert)
                #print(f"  Created REAL: pois shape {vert_pois.shape}, mask shape {vert_loss_mask.shape}")

            all_pois.append(vert_pois)
            all_loss_masks.append(vert_loss_mask)

        #print(f"\n=== SAMPLE {index}: VERTEBRA {vertebra} ===")
        #print(f"all_vert: {all_vert}")

        # combine pois and loss masks
        combined_pois = torch.cat(all_pois, dim=0)
        combined_loss_mask = torch.cat(all_loss_masks, dim=0)

        # add global offset
        combined_pois = combined_pois + torch.tensor(offset)    

        # Convert subreg, vertseg and surface_msk to tensors
        ct = torch.from_numpy(ct.astype(float))
        subreg = torch.from_numpy(subreg.astype(float))
        vertseg = torch.from_numpy(vertseg.astype(float))
        surface_msk = torch.from_numpy(surface_msk.astype(float))

        # Add channel dimension
        ct = ct.unsqueeze(0)
        subreg = subreg.unsqueeze(0)
        vertseg = vertseg.unsqueeze(0)
        surface_msk = surface_msk.unsqueeze(0)


        if self.input_data_type == "vertseg":  
            data_dict["input"] = vertseg  
        elif self.input_data_type == "subreg":   
            data_dict["input"] = subreg  
        elif self.input_data_type == "ct":  
            data_dict["input"] = ct
        elif self.input_data_type == "surface_msk":
            data_dict["input"] = surface_msk

        data_dict["target"] = combined_pois

        poi_indices = torch.tensor(self.poi_indices)
        repeat_poi_indices = torch.cat([poi_indices for _ in range(3)])
        data_dict["target_indices"] = repeat_poi_indices

        # apply transform
        data_dict = self.transform(data_dict) if self.transform else data_dict

        # Identify pois outside of the input shape after transform
        max_x = self.input_shape[0] - 1
        max_y = self.input_shape[1] - 1
        max_z = self.input_shape[2] - 1

        outside_poi_indices = (
            (data_dict["target"][:, 0] < 0)
            | (data_dict["target"][:, 0] > max_x)
            | (data_dict["target"][:, 1] < 0)
            | (data_dict["target"][:, 1] > max_y)
            | (data_dict["target"][:, 2] < 0)
            | (data_dict["target"][:, 2] > max_z)
        )
        
        combined_loss_mask[outside_poi_indices] = 0
        
        data_dict["loss_mask"] = combined_loss_mask.bool()



        #print(f"FINAL combined_pois shape: {combined_pois.shape}")
        #print(f"FINAL combined_loss_mask shape: {combined_loss_mask.shape}")
        #print(f"Expected shape: ({3 * len(self.poi_indices)}, 3) and ({3 * len(self.poi_indices)},)")



        transformed_mask = data_dict["input"] > 0
        surface = compute_surface(transformed_mask, iterations=self.iterations)

        data_dict["surface"] = surface
        data_dict["subject"] = str(subject)
        data_dict["vertebra"] = vertebra
        data_dict["zoom"] = torch.tensor(zoom).float()
        data_dict["offset"] = torch.tensor(offset).float()
        data_dict["ct_path"] = ct_path
        data_dict["msk_path"] = msk_path
        data_dict["subreg_path"] = subreg_path
        data_dict["poi_path"] = poi_path
        data_dict["poi_list_idx"] = torch.tensor(
            [self.poi_idx_to_list_idx[poi.item()] for poi in repeat_poi_indices]
        )
        data_dict["vert_list_idx"] = torch.tensor([self.vert_idx_to_list_idx[vertebra]])

        #data_dict["n_vertebrae"] = len([vert for _, vert in all_vert if vert != 0])
        #data_dict["vertebrae_list"] = torch.tensor(all_vert)  
        #data_dict["current_vertebra_idx"] = 0
        data_dict["current_vertebra"] = vertebra
        data_dict["n_pois_per_vertebra"] = len(self.poi_indices)

        """
        print("=== BASIC SHAPES ===")
        print(f"Input shape: {data_dict['input'].shape}")
        print(f"Target shape: {data_dict['target'].shape}")  
        print(f"Loss mask shape: {data_dict['loss_mask'].shape}")
        print(f"Expected: {data_dict['n_vertebrae']} vertebrae × {self.poi_indices} POIs = {data_dict['n_vertebrae'] * len(self.poi_indices)} total POIs")


        print("\n=== LOSS MASK VALIDATION ===")
        loss_mask = data_dict['loss_mask']
        target = data_dict['target']
        
        # Check außerhalb der Grenzen
        input_shape = self.input_shape
        outside_bounds = (
            (target[:, 0] < 0) | (target[:, 0] >= input_shape[0]) |
            (target[:, 1] < 0) | (target[:, 1] >= input_shape[1]) |
            (target[:, 2] < 0) | (target[:, 2] >= input_shape[2])
        )
        
        print(f"POIs outside bounds: {outside_bounds.sum()}")
        print(f"Outside POIs masked: {(outside_bounds & ~loss_mask).sum() == outside_bounds.sum()}")
        
        # Check dummy POIs
        dummy_pois = (target == -1).all(dim=1)
        print(f"Dummy POIs: {dummy_pois.sum()}")
        print(f"Dummy POIs masked: {(dummy_pois & loss_mask).sum() == 0}")
        """
        return data_dict

class GruberNeighborDataset(PoiNeighborDataset):
    def __init__(
        self,
        master_df,
        input_data_type="subreg",
        input_shape=(120, 121, 149),
        transforms=None,
        flip_prob=0.0,
        include_com=False,
        include_poi_list=None,
        include_vert_list=None,
        poi_file_ending="poi.json",
        iterations=1,
    ):
        super().__init__(
            master_df,
            poi_indices=(
                include_poi_list
                if include_poi_list
                else (
                    [
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

                        41,
                        42,
                        43,
                        44,
                        45,
                        46,
                        47,
                        48,
                        49,
                        50,
                    ]
                    if include_com
                    else [
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
                    ]
                )
            ),
            include_vert_list=(
                include_vert_list
                if include_vert_list
                else [
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                ]
            ),
            poi_flip_pairs={
                # no flips
                0:0,
                41: 41,
                42: 42,
                43: 43,
                44: 44,
                45: 45,
                46: 46,
                47: 47,
                48: 48,
                49: 49,
                50: 50,
                81: 81,
                82: 82, 
                83: 83, 
                84: 84,
                85: 85,
                86: 86,
                87: 87,
                88: 88,
                89: 89, 
                101: 101,
                102: 102,
                103: 103,
                104: 104,
                105: 105, 
                106: 106, 
                107: 107, 
                108: 108, 
                109: 109,
                110: 110,
                111: 111,
                112: 112,
                113: 113, 
                114: 114, 
                115: 115, 
                116: 116, 
                117: 117,
                118: 118,
                119: 119,
                120: 120,
                121: 121, 
                122: 122, 
                123: 123, 
                124: 124, 
                125: 125,
                127: 127,                
            },
            input_data_type=input_data_type,
            input_shape=input_shape,
            transforms=transforms,
            flip_prob=flip_prob,
            include_com=include_com,
            poi_file_ending=poi_file_ending,
            iterations=iterations,
        )


def custom_collate_fn(batch):
    """Custom collate function die dir genau zeigt wo der Fehler ist"""
    print(f"\n=== COLLATING BATCH OF SIZE {len(batch)} ===")
    
    # Prüfe jeden Key einzeln
    for key in batch[0].keys():
        print(f"\nTrying to collate key: '{key}'")
        try:
            values = [item[key] for item in batch]
            
            # Zeige Shapes/Types für diesen Key
            for i, val in enumerate(values):
                if isinstance(val, torch.Tensor):
                    print(f"  Sample {i} - {key}: {val.shape} (dtype: {val.dtype})")
                else:
                    print(f"  Sample {i} - {key}: {type(val)} - {val}")
            
            # Versuche zu kollationieren
            if isinstance(values[0], torch.Tensor):
                collated = torch.stack(values)
                print(f"  -> Successfully collated {key}: {collated.shape}")
            else:
                print(f"  -> {key} is not a tensor, skipping...")
                
        except Exception as e:
            print(f"  -> ERROR collating {key}: {e}")
            print(f"  -> This is likely the problematic tensor!")
            raise e

