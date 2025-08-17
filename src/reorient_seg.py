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

def get_vertseg(container):
    vertseg_query = container.new_query(flatten=True)
    vertseg_query.filter_format("msk")
    vertseg_query.filter_filetype("nii.gz")  # only nifti files
    vertseg_query.filter("seg", "vert")
    vertseg_candidate = vertseg_query.candidates[0]
    return str(vertseg_candidate.file["nii.gz"])

def reorient_rescale_seg(vertseg_path, save_path):
    vert_msk = NII.load(vertseg_path, seg=True)


    print("orientation before:", vert_msk.orientation)
    print("origin before:", vert_msk.origin)
    print("zoom before:", vert_msk.zoom)

    vert_msk.reorient_(("L", "A", "S"))
    vert_msk.rescale_((1, 1, 1))

    print("orientation after:", vert_msk.orientation)
    print("origin after:", vert_msk.origin)
    print("zoom after:", vert_msk.zoom)
    


    os.makedirs(save_path, exist_ok=True)

    seg_path = os.path.join(save_path, str(sub) + "_preproccessed-seg.nii.gz")


    vert_msk.save(seg_path, verbose=False)

def analyze_segmentation(temp_dir):
    for filename in os.listdir(temp_dir):
        if "vertseg" in filename:
            full_path = os.path.join(temp_dir, filename)
            vertseg = NII.load(full_path, seg=True)
            print("filename:", filename)
            print("Orientation:", vertseg.orientation)
            print("Origin:", vertseg.origin)
            print("Zoom:", vertseg.zoom)  
            print("\n\n")  

def analyze_prediction():
    pass        



if __name__ == "__main__":                                                                                              

    bgi = BIDS_Global_info(
        datasets=["/home/student/alissa/3dVertPois/src/dataset/data_preprocessing/dataset-folder-test"],
        parents=["derivatives"],
    )

    save_path = "/home/student/alissa/3dVertPois/src/predictions/preproccessed-vertseg-combined"

    #for sub, container in bgi.enumerate_subjects():
    #    vert_path = get_vertseg(container)

    #    reorient_rescale_seg(vert_path, save_path)

    temp_dir = "/home/student/alissa/3dVertPois/src/tmp/WS-05"
    analyze_segmentation(temp_dir)

