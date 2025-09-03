import os
import numpy as np
from TPTBox import NII

def find_max_shape():
    """
    Findet die maximale Form (Shape) aller 'vertseg.nii.gz' Dateien in den Unterordnern von '/cutouts'.
    Gibt die maximale Form als Tuple zurück.
    """
    base_dir = 'cutout-folder/cutouts-neighbor'  # Basisordner anpassen, falls nötig

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
                shape = seg_mask.shape  # tuple mit Dimensionen
                print(shape)
                if max_shape is None:
                    max_shape = shape
                else:
                    # max_shape wird komponentenweise maximiert
                    max_shape = tuple(max(m, s) for m, s in zip(max_shape, shape))

    print("Maximale Shape aller vertseg.nii.gz Dateien:", max_shape)


if __name__ == "__main__":
    find_max_shape()