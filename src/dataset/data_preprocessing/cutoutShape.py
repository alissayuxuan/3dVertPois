import os
import numpy as np
from TPTBox import NII
from TPTBox.core.poi import POI

import pandas as pd
import os

def copy_and_update_master_df(base_dir: str, save_dir: str, old_folder: str = "cutouts", new_folder: str = "cutouts-2.0_zoom"):
    """
    Copies master_df.csv from base_dir to save_dir and updates file_dir paths
    
    Args:
        base_dir: Source directory containing master_df.csv
        save_dir: Destination directory for the updated master_df.csv
        old_folder: Folder name to replace in file_dir (default: "cutouts")
        new_folder: New folder name (default: "cutouts-0.5_zoom")
    """
    # Load the master_df
    master_df_path = os.path.join(base_dir, 'master_df-excel_outliers_exclude.csv')
    df = pd.read_csv(master_df_path)
    
    # Update the file_dir column
    df['file_dir'] = df['file_dir'].str.replace(f'/{old_folder}/', f'/{new_folder}/', regex=False)
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the updated dataframe
    output_path = os.path.join(save_dir, 'master_df-excel_outliers_exclude.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Updated master_df.csv saved to: {output_path}")
    print(f"Updated {len(df)} rows, replacing '{old_folder}' with '{new_folder}' in file paths")




def find_max_shape():
    """
    Goes through all segmentation masks and returns the maximum shape found as a tuple.
    """
    base_dir = 'cutout-folder/cutouts-2.0_zoom'#'cutout-folder/cutouts-0.5_zoom'  # change!

    max_shape = None

    for ws_folder in os.listdir(base_dir):
        ws_path = os.path.join(base_dir, ws_folder)
        if not os.path.isdir(ws_path):
            continue
        for subfolder in os.listdir(ws_path):
            sub_path = os.path.join(ws_path, subfolder)
            if not os.path.isdir(sub_path):
                continue
            file_path = os.path.join(sub_path, 'subreg.nii.gz')
            if os.path.isfile(file_path):
                seg_mask = NII.load(file_path, seg=True) 
                shape = seg_mask.shape  
                print(shape)
                if max_shape is None:
                    max_shape = shape
                else:
                    max_shape = tuple(max(m, s) for m, s in zip(max_shape, shape))

    print("Max shape:", max_shape)


def rescale_cutouts(zoom:tuple):
    """
    rescales all files to specified zoom
    """
    base_dir = 'cutout-folder/cutouts' # change
    save_dir = 'cutout-folder/cutouts-2.0_zoom' #change


    for ws_folder in os.listdir(base_dir):
        ws_path = os.path.join(base_dir, ws_folder)
        if not os.path.isdir(ws_path):
            continue
        for subfolder in os.listdir(ws_path):
            sub_path = os.path.join(ws_path, subfolder)
            if not os.path.isdir(sub_path):
                continue

            output_dir = os.path.join(save_dir, ws_folder, subfolder)
            os.makedirs(output_dir, exist_ok=True)

            files_to_process = [
                ('subreg.nii.gz', True), #(filename, is_segmentation)
                ('vertseg.nii.gz', True),
                ('surface_msk.nii.gz', True),
                ('ct.nii.gz', False)
            ]
            # Segmentation masks and CT scans
            for filename, is_seg in files_to_process:
                input_path = os.path.join(sub_path, filename)
                output_path = os.path.join(output_dir, filename)

                if os.path.isfile(input_path):
                    nii_data = NII.load(input_path, seg=is_seg) 
                    nii_data.rescale_(zoom)
                    nii_data.save(output_path, verbose=False)

            # POI files
            poi_path = os.path.join(sub_path, 'poi.json')
            if os.path.isfile(poi_path):
                save_poi_path = os.path.join(output_dir, 'poi.json')
                save_global_poi_path = os.path.join(output_dir, 'poi_global.json')
                
                poi = POI.load(poi_path) 
                poi.rescale_(zoom)
                poi.save(save_poi_path, verbose=False)
                poi.to_global().save_mrk(save_global_poi_path)           


def mask_seg():
    subreg_16_12_path = "cutout-folder/cutouts/WS-16/12/subreg.nii.gz"
    vertseg_16_12_path = "cutout-folder/cutouts/WS-16/12/vertseg.nii.gz"
    vert = 12

    subreg = NII.load(subreg_16_12_path, seg=True)
    vertseg = NII.load(vertseg_16_12_path, seg=True)

    # Extract only vertebra 12 from the vertebra segmentation  
    vert_12_mask = vertseg.extract_label(12)  
      
    # Apply the mask to the subregion  
    masked_subreg = subreg.apply_mask(vert_12_mask)  
      
    masked_subreg.save("ppt_seg/subreg_16_12.nii.gz")
    """
    subreg_arr = subreg.get_array()
    vertseg_arr = vertseg.get_array()
    

    mask = (vertseg_arr == vert)
    new_subreg = subreg_arr * mask # correct??

    subreg.set_array(new_subreg)

    subreg.save("ppt_seg/subreg_16_12.nii.gz")
    """



if __name__ == "__main__":
    #find_max_shape()
    #rescale_cutouts((2, 2, 2))

    #copy_and_update_master_df(
    #    base_dir='cutout-folder/cutouts',
    #    save_dir='cutout-folder/cutouts-2.0_zoom'
    #)

    mask_seg()