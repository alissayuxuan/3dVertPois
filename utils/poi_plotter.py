import json
import os
from pathlib import Path

import matplotlib.colors
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
import sys

sys.path.append(str(Path(__file__).parent.parent))
from TPTBox import (
    Centroids,
    NII,
    POI_Reference,
    Image_Reference,
    to_nii_seg,
)
import numpy as np
from TPTBox.core.poi import POI
from TPTBox.core.np_utils import np_bbox_binary
import pyvista as pv
from tqdm import tqdm


def visualize_pois(
    ctd_in: POI_Reference,
    seg_vert: Image_Reference,
    vert_idx_list: list[int],
    cmap: matplotlib.colors.ListedColormap = None,
    save_path: Path | str | None = None,
):
    """Visualizes a given POIs on top of a segmentation image

    Args:
        ctd: Centroid reference containing the POIs
        seg_vert: Segmentation Mask
        vert_idx_list: list of vertebra indices to plot
        cmap: ListedColormap vor the segmentation. If None, uses pyvistas default cmap

    Returns:
        None (shows the visualization)
    """
    ctd = Centroids.load(ctd_in)
    seg = to_nii_seg(seg_vert).reorient_(verbose=True)
    seg_labels = seg.unique()
    ctd.reorient_(verbose=True)

    poi_vert: dict[int, dict] = {l: {} for l in seg_labels}
    for p_id, v_id, poi in ctd.items():
        if v_id in poi_vert:
            poi_vert[v_id][p_id] = poi

    # visualize POIs in a plot
    poi_coords = []

    p = pv.Plotter()
    p.set_background("black", top=None)
    p.add_axes()
    for vert_id in tqdm(vert_idx_list):
        if vert_id in seg_labels:
            vert_mesh = make_vert_mesh(seg, vert_id)
            p.add_mesh(vert_mesh, opacity=0.95, cmap=cmap)
            for p_id, coord in poi_vert[vert_id].items():
                poi_coords.append(coord)
    n = pv.PolyData(poi_coords)
    n["radius"] = np.ones(shape=len(poi_coords)) * 5
    geom = pv.Sphere(theta_resolution=8, phi_resolution=8)
    glyphed = n.glyph(scale="radius", geom=geom, progress_bar=False, orient=False)
    p.add_mesh(glyphed, color="red")

    if save_path is not None:
        p.export_obj(save_path)

    p.show()


def make_vert_mesh(seg: NII, vert_id: int):
    seg_arr = seg.reorient_().get_seg_array()
    vert_arr = seg_arr.copy()
    vert_arr[seg_arr != vert_id] = 0

    bbox_crop = np_bbox_binary(vert_arr, px_dist=2)
    x1, y1, z1 = bbox_crop[0].start, bbox_crop[1].start, bbox_crop[2].start
    arr_cropped = vert_arr[bbox_crop]

    vert_verts, vert_faces, vert_normals, vert_values = marching_cubes(arr_cropped, gradient_direction="ascent", step_size=1)
    vert_verts += (x1, y1, z1)  # so it has correct global coordinates

    vfaces = np.column_stack(
        (
            np.ones(
                len(vert_faces),
            )
            * 3,
            vert_faces,
        )
    ).astype(int)

    mesh = pv.PolyData(vert_verts, vfaces)
    mesh["Normals"] = vert_normals
    mesh["values"] = vert_values
    return mesh
