import nibabel as nib
import numpy as np
import torch

# from BIDS import NII, POI
from TPTBox import NII
from TPTBox.core.poi import POI

# from scipy.ndimage import distance_transform_edt
import torch.nn.functional as F
from skimage import measure
from TPTBox.core.np_utils import np_fill_holes
from nibabel.nifti1 import Nifti1Image

# from utils.raycast_torch import max_distance_ray_cast_convex_torch


def np_to_bids_nii(array: np.ndarray) -> NII:
    """Converts a numpy array to a BIDS NII object."""
    # NiBabel expects the orientation to be RAS+ (right, anterior, superior, plus),
    # we have LAS+ (left, posterior, superior, plus) so we need to flip along the second axis
    affine = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    nifty1img = nib.Nifti1Image(array, affine)
    return NII(nifty1img)


def one_hot_encode_batch(batch_tensor: torch.Tensor) -> torch.Tensor:
    """One hot encodes a batch of labels."""
    batch_tensor = batch_tensor.squeeze(1).long()

    num_classes = 11
    batch_tensor = batch_tensor - 40
    batch_tensor[batch_tensor < 0] = 0

    batch_tensor = torch.nn.functional.one_hot(batch_tensor, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    return batch_tensor


def surface_project_poi_vert_wise(poi: POI, surface_nii: NII, requires_filling: bool = True):
    poi_new = poi.make_empty_POI()
    for v in poi.keys_region():
        assert v in surface_nii.unique(), f"Surface NII does not contain vertebra {v}"

        # extract surface for this vertebra
        surf_v_nii = surface_nii.extract_label(v)
        poi_v = poi.extract_region(v)
        poi_v_proj = surface_project_poi(poi_v, surf_v_nii, requires_filling=requires_filling)
        for r, s, c in poi_v_proj.items():
            poi_new[r, s] = c
    return poi_new


def surface_project_poi(poi: POI, surface_nii: NII, requires_filling: bool = True):
    crop = surface_nii.compute_crop(dist=4)
    poi_crop = poi.apply_crop(crop)
    # convert poi to coordinates tensor
    coord_keys = []
    coords_list = []
    for r, s, c in poi_crop.items():
        coord_keys.append((r, s))
        coords_list.append(c)
    # surface_nii to torch tensor
    surface_tensor = torch.from_numpy(surface_nii.apply_crop(crop).get_array()).unsqueeze(0).to(torch.float32)  # (1,D,H,W)
    # run surface_project_coords
    coords_tensor = torch.from_numpy(np.asarray(coords_list)).unsqueeze(0).to(torch.float32)  # (1,N,3)
    projected_coords, _ = surface_project_coords(coords_tensor, surface_tensor, requires_filling=requires_filling)  # (1,N,3)
    # transfer back
    projected_poi = poi_crop.make_empty_POI()
    for i, (r, s) in enumerate(coord_keys):
        projected_poi[r, s] = projected_coords[0, i].cpu().numpy()
    final_poi = projected_poi.apply_crop_reverse(crop, surface_nii.shape)
    # return
    return final_poi


def surface_project_coords(
    coordinates,
    surface,
    debug=False,
    requires_filling: bool = True,
):
    return surface_project_coords_marchingcubes(coordinates, surface, debug=debug, requires_filling=requires_filling)


def fill_holes_3d(surface_mask):
    """
    Proper hole filling: flood-fill outside -> invert -> keep only internal cavities.
    Does NOT dilate or overfill the object.

    surface_mask: (B,1,D,H,W) boolean tensor:
        True = object (surface or any filled region)

    Returns:
        filled: same shape, boolean:
            object + internal cavities filled
    """

    surf = surface_mask.bool()
    B, _, D, H, W = surf.shape
    device = surf.device

    # --- Step 1: treat object as solid
    solid = surf.clone()

    # --- Step 2: create "outside" mask (everything not solid)
    outside = ~solid

    # --- Step 3: seeds = all border voxels that are outside
    seeds = torch.zeros_like(outside)
    seeds[:, :, 0, :, :] = True
    seeds[:, :, -1, :, :] = True
    seeds[:, :, :, 0, :] = True
    seeds[:, :, :, -1, :] = True
    seeds[:, :, :, :, 0] = True
    seeds[:, :, :, :, -1] = True

    seeds = seeds & outside  # only start flood from true outside

    # --- Step 4: flood fill outside
    kernel = torch.ones((1, 1, 3, 3, 3), device=device)

    prev = torch.zeros_like(seeds)
    cur = seeds.clone()

    # Iterate until stable
    while True:
        # Dilate current region
        expanded = F.conv3d(cur.float(), kernel, padding=1) > 0
        # Only expand into true outside
        expanded = expanded & outside

        if torch.equal(expanded, cur):
            break

        cur = expanded

    outside_filled = cur
    inside = ~outside_filled  # inside object + holes

    # --- Step 5: holes = inside but not originally solid
    holes = inside & (~solid)

    # --- Step 6: final = original object + holes filled
    filled = solid | holes

    return filled


def fill_holes_3d_6conn(surface_mask):
    """
    Fill holes using STRICT 6-connectivity (no diagonal connectivity).
    Works on GPU. Does not overfill thin structures.

    surface_mask: (B,1,D,H,W) boolean or int tensor
        True = object/surface voxels

    Returns:
        filled: (B,1,D,H,W) boolean tensor
            original object + filled 6-connected cavities
    """

    surf = surface_mask.bool()
    B, _, D, H, W = surf.shape
    device = surf.device

    # Step 1 — solid voxels as-is
    solid = surf

    # Step 2 — outside = everything not solid
    outside = ~solid

    # Step 3 — Outside seeds = boundary outside voxels
    seeds = torch.zeros_like(outside)
    seeds[:, :, 0, :, :] = True
    seeds[:, :, -1, :, :] = True
    seeds[:, :, :, 0, :] = True
    seeds[:, :, :, -1, :] = True
    seeds[:, :, :, :, 0] = True
    seeds[:, :, :, :, -1] = True
    seeds = seeds & outside

    # --- 6-connected kernel (center + 6 face neighbors)
    kernel = torch.zeros((1, 1, 3, 3, 3), device=device)
    kernel[0, 0, 1, 1, 0] = 1  # -x
    kernel[0, 0, 1, 1, 2] = 1  # +x
    kernel[0, 0, 1, 0, 1] = 1  # -y
    kernel[0, 0, 1, 2, 1] = 1  # +y
    kernel[0, 0, 0, 1, 1] = 1  # -z
    kernel[0, 0, 2, 1, 1] = 1  # +z

    # Step 4 — flood-fill outside with 6-connectivity
    cur = seeds.clone()

    while True:
        # 6-connected dilation
        expanded = F.conv3d(cur.float(), kernel, padding=1) > 0

        # Only grow into true outside
        expanded = expanded & outside

        if torch.equal(expanded, cur):
            break

        cur = expanded

    outside_filled = cur
    inside = ~outside_filled  # object + closed cavities

    # Holes = inside but not originally solid
    holes = inside & (~solid)

    # Final = original + holes
    filled = solid | holes

    return filled


def surface_project_coords_sdf(coordinates, surface, chamfer_iters=4):
    """
    Project coordinates onto the nearest surface using SDF + normal projection.

    coordinates: (B, N, 3) or (N, 3)
    surface:     (B, D, H, W) or (B, 1, D, H, W) or (D, H, W)

    Returns:
        projected coordinates: same shape as input
        distance to surface:  (B,N) or (N,)
    """
    # --- 0. normalize shapes ---
    unbatched_coords = coordinates.ndim == 2
    unbatched_surface = surface.ndim == 3

    if unbatched_coords:
        coordinates = coordinates.unsqueeze(0)
    if unbatched_surface:
        surface = surface.unsqueeze(0)
    if surface.ndim == 5 and surface.shape[1] == 1:
        surface = surface[:, 0]

    # --- 1. Fill holes (6-connectivity) ---
    surface = fill_holes_3d_6conn(surface.unsqueeze(1)).squeeze(1)

    B, D, H, W = surface.shape
    device = coordinates.device
    dtype = torch.float32

    surface = surface.to(device, dtype=dtype)

    # --- 2. Compute Chamfer-based SDF ---
    max_dist = D + H + W
    dist_out = torch.where(surface == 0, torch.zeros_like(surface), torch.full_like(surface, max_dist))
    dist_in = torch.where(surface == 1, torch.zeros_like(surface), torch.full_like(surface, max_dist))

    kernel = torch.ones((1, 1, 3, 3, 3), device=device, dtype=dtype)
    kernel[0, 0, 1, 1, 1] = 0

    for _ in range(chamfer_iters):
        neigh_out = F.conv3d(dist_out.unsqueeze(1), kernel, padding=1).squeeze(1)
        dist_out = torch.min(dist_out, neigh_out + 1)
        neigh_in = F.conv3d(dist_in.unsqueeze(1), kernel, padding=1).squeeze(1)
        dist_in = torch.min(dist_in, neigh_in + 1)

    sdf = dist_out - dist_in
    sdf = sdf.unsqueeze(1)

    # --- 3. Compute normals (finite difference) ---
    dx = torch.tensor([[-1, 0, 1]], device=device, dtype=dtype).reshape(1, 1, 1, 1, 3)
    dy = torch.tensor([[-1], [0], [1]], device=device, dtype=dtype).reshape(1, 1, 1, 3, 1)
    dz = torch.tensor([[-1], [0], [1]], device=device, dtype=dtype).reshape(1, 1, 3, 1, 1)

    gx = F.conv3d(sdf, dx, padding=(0, 0, 1))
    gy = F.conv3d(sdf, dy, padding=(0, 1, 0))
    gz = F.conv3d(sdf, dz, padding=(1, 0, 0))

    normals = torch.cat([gz, gy, gx], dim=1)
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-12)

    # --- 4. Sample SDF and normals at coordinates ---
    coords = coordinates.float()
    norm = coords.clone()
    norm[..., 0] = (coords[..., 2] / (W - 1)) * 2 - 1
    norm[..., 1] = (coords[..., 1] / (H - 1)) * 2 - 1
    norm[..., 2] = (coords[..., 0] / (D - 1)) * 2 - 1
    grid = norm.view(B, -1, 1, 1, 3)

    sdf_vals = F.grid_sample(sdf, grid, align_corners=True).view(B, -1)
    normal_vals = F.grid_sample(normals, grid, align_corners=True).view(B, 3, -1).permute(0, 2, 1)

    # --- 5. Project coordinates ---
    projected = coords - sdf_vals.unsqueeze(-1) * normal_vals
    proj_dist = sdf_vals.abs()

    if torch.isnan(projected).any():
        print("NaNs detected in projected coordinates!")

    if unbatched_coords:
        return projected[0], proj_dist[0]
    return projected, proj_dist


def extract_surface_vertices(mask, level=0.5):
    # mask: (Z,Y,X) numpy array
    # mask = fill_holes_3d_6conn(mask)
    verts, faces, _, _ = measure.marching_cubes(
        mask.astype(float),
        level=level,
    )
    return torch.from_numpy(verts.copy()).float()  # (M, 3)


def surface_project_coords_marchingcubes(
    coordinates,
    surface_mask,
    level=0.5,
    debug=False,
    requires_filling: bool = False,
):
    """
    coordinates: (B, N, 3) or (N, 3)
    surface_mask: (B, Z, Y, X) or (Z, Y, X)

    returns:
        surface_projected_targets: (B, N, 3) int64
        surface_projection_dist:   (B, N)   float
    """
    # ---------- Handle batching ----------
    unbatched_coords = coordinates.ndim == 2
    unbatched_surface = surface_mask.ndim == 3

    if unbatched_coords:
        coordinates = coordinates.unsqueeze(0)
    if unbatched_surface:
        surface_mask = surface_mask.unsqueeze(0)

    B, N, _ = coordinates.shape
    device = coordinates.device

    # ---------- Extract marching cubes surfaces for each batch ----------
    surface_vertices = []
    for b in range(B):
        mask_np = surface_mask[b].detach().cpu().numpy().astype(float)
        if mask_np.ndim == 4:
            mask_np = mask_np[0]

        if debug:
            np_to_bids_nii(mask_np).save("/DATA/NAS/ongoing_projects/hendrik/poi_prediction/test_mask_np.nii.gz")
        if requires_filling:
            mask_np = np_fill_holes(mask_np)
        if debug:
            np_to_bids_nii(mask_np).save("/DATA/NAS/ongoing_projects/hendrik/poi_prediction/test_mask_np_filled.nii.gz")
        (verts,) = (extract_surface_vertices(mask_np, level=level),)
        surface_vertices.append(verts)

    # pad surfaces to same size so batching works
    max_M = max(v.shape[0] for v in surface_vertices)
    padded_surfaces = torch.zeros((B, max_M, 3), device=device, dtype=torch.float32)
    surface_valid = torch.zeros((B, max_M), device=device, dtype=torch.bool)

    for b, v in enumerate(surface_vertices):
        M = v.shape[0]
        padded_surfaces[b, :M] = v
        surface_valid[b, :M] = True

    # print shapes
    # print("padded surfaces shape:", padded_surfaces.shape)
    # print("coordinates shape:", coordinates.shape)
    # print("surface_valid shape:", surface_valid.shape)
    # print("surface_mask_shape:", surface_mask.shape)
    # print("coords example:", coordinates[0, :5])
    # print("surface example:", padded_surfaces[0, :5])

    # ---------- Compute distances ----------
    coords_exp = coordinates.unsqueeze(2)  # (B, N, 1, 3)
    surf_exp = padded_surfaces.unsqueeze(1)  # (B, 1, M, 3)
    diff = coords_exp - surf_exp  # (B, N, M, 3)
    dist_sq = (diff**2).sum(dim=-1)  # (B, N, M)

    # mask invalid padded values
    # dist_sq[~surface_valid.unsqueeze(1)] = float("inf")
    dist_sq[~surface_valid.unsqueeze(1).expand_as(dist_sq)] = float("inf")

    # ---------- Find nearest surface vertex ----------
    min_dist_sq, min_idx = torch.min(dist_sq, dim=-1)  # (B, N)

    # ---------- Gather vertices ----------
    batch_idx = torch.arange(B, device=device).unsqueeze(-1)
    projected = padded_surfaces[batch_idx, min_idx]  # (B, N, 3)

    # ---------- Convert to voxel indices ----------
    # surface_projected_targets = projected.round().long()
    surface_projection_dist = torch.sqrt(min_dist_sq)

    # ---------- Unbatch if needed ----------
    if unbatched_coords:
        projected = projected.squeeze(0)
        surface_projection_dist = surface_projection_dist.squeeze(0)

    return projected, surface_projection_dist


def surface_project_coords_old(coordinates, surface):
    unbatched = len(coordinates.shape) == 2
    if unbatched:
        coordinates = coordinates.detach().clone().unsqueeze(0)
        surface = surface.detach().clone().unsqueeze(0)
    B, N, _ = coordinates.shape
    device = coordinates.device
    surface_projected_targets = torch.zeros_like(coordinates, dtype=torch.int64)
    surface_projection_dist = torch.zeros(B, N, dtype=torch.float32)

    for b in range(B):
        for i in range(N):
            coord = coordinates[b, i, :]
            # Create a mask of all 'true' surface points for this batch
            surface_points_indices = surface[b].squeeze(0).nonzero(as_tuple=False)
            # Calculate Euclidean distances from the current point to all surface points
            distances = torch.sqrt(((surface_points_indices - coord) ** 2).sum(dim=1))
            # Find the index of the closest surface point
            min_dist_index = torch.argmin(distances)
            closest_point = surface_points_indices[min_dist_index]
            surface_projected_targets[b, i, :] = closest_point
            surface_projection_dist[b, i] = distances[min_dist_index]
    return surface_projected_targets.to(device), surface_projection_dist.to(device)


# POI Visualization
# Define some useful utility functions
def get_dd_ctd(dd, poi_list=None):
    ctd = {}
    vertebra = dd["vertebra"]

    for poi_coords, poi_idx in zip(dd["target"], dd["target_indices"]):
        coords = (poi_coords[0].item(), poi_coords[1].item(), poi_coords[2].item())
        if poi_list is None or poi_idx in poi_list:
            ctd[vertebra, poi_idx.item()] = coords

    ctd = POI(centroids=ctd, orientation=("L", "A", "S"), zoom=(1, 1, 1), shape=(128, 128, 96))
    return ctd


def get_ctd(target, target_indices, vertebra, poi_list):
    ctd = {}
    for poi_coords, poi_idx in zip(target, target_indices):
        coords = (poi_coords[0].item(), poi_coords[1].item(), poi_coords[2].item())
        if poi_list is None or poi_idx in poi_list:
            ctd[vertebra, poi_idx.item()] = coords

    ctd = POI(centroids=ctd, orientation=("L", "A", "S"), zoom=(1, 1, 1), shape=(128, 128, 96))
    return ctd


def get_vert_msk_nii(dd):
    vertebra = dd["vertebra"]
    msk = dd["input"].squeeze(0)
    return vertseg_to_vert_msk_nii(vertebra, msk)


def vertseg_to_vert_msk_nii(vertebra, msk):
    vert_msk = (msk != 0) * vertebra
    vert_msk_nii = np_to_bids_nii(vert_msk.numpy().astype(np.int32))
    vert_msk_nii.seg = True
    return vert_msk_nii


def get_vertseg_nii(dd):
    vertseg = dd["input"].squeeze(0)
    vertseg_nii = np_to_bids_nii(vertseg.numpy().astype(np.int32))
    vertseg_nii.seg = True
    return vertseg_nii


def get_vert_points(dd):
    msk = dd["input"].squeeze(0)
    vert_points = torch.where(msk)
    vert_points = torch.stack(vert_points, dim=1)
    return vert_points


def get_target_entry_points(dd):
    ctd = get_ctd(dd)
    vertebra = dd["vertebra"]
    p_90 = torch.tensor(ctd[vertebra, 90])
    p_92 = torch.tensor(ctd[vertebra, 92])

    p_91 = torch.tensor(ctd[vertebra, 91])
    p_93 = torch.tensor(ctd[vertebra, 93])

    return p_90, p_92, p_91, p_93


def tensor_to_ctd(
    t,
    vertebra,
    origin,
    rotation,
    idx_list=None,
    shape=(128, 128, 96),
    zoom=(1, 1, 1),
    offset=(0, 0, 0),
):
    ctd = {}
    for i, coords in enumerate(t):
        coords = coords.float() - torch.tensor(offset)
        coords = (coords[0].item(), coords[1].item(), coords[2].item())
        if idx_list is None:
            ctd[vertebra, i] = coords
        elif i < len(idx_list):
            ctd[vertebra, idx_list[i]] = coords

    ctd = POI(
        centroids=ctd,
        orientation=("L", "A", "S"),
        zoom=zoom,
        shape=shape,
        origin=origin,
        rotation=rotation,
    )
    return ctd


if __name__ == "__main__":
    # simple test
    volume = torch.tensor(
        [
            [
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 1, 1, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 1, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 1, 1, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ]
    )
    print(volume.shape)
    print(volume)

    volume = fill_holes_3d_6conn(volume.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

    print(volume)

    coord = torch.Tensor([2, 2, 2])
    projected, proj_dist = surface_project_coords_sdf(coord, volume)
    print(projected, proj_dist)
