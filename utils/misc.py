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
from scipy.ndimage import distance_transform_edt

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
    return surface_project_coords_marchingcubes_continuous(coordinates, surface, debug=debug, requires_filling=requires_filling)


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


def surface_project_coords_sdf_continuous(
    coordinates,
    surface_mask,
    debug=False,
    step_size=0.1,
    max_steps=60,
    tol=1e-4,
    requires_filling: bool = False,
):
    # ---------- batching ----------
    unbatched_coords = coordinates.ndim == 2
    unbatched_surface = surface_mask.ndim == 3

    if unbatched_coords:
        coordinates = coordinates.unsqueeze(0)
    if unbatched_surface:
        surface_mask = surface_mask.unsqueeze(0)

    B, N, _ = coordinates.shape
    device = coordinates.device

    projected = torch.zeros_like(coordinates)
    surface_projection_dist = torch.zeros((B, N), device=device)

    for b in range(B):

        # ====== 1) BUILD SDF ON CPU (unchanged) ======
        mask_np = surface_mask[b].detach().cpu().numpy().astype(np.uint8)
        if mask_np.ndim == 4:
            mask_np = mask_np[0]

        if requires_filling:
            mask_np = np_fill_holes(mask_np)

        dist_in = distance_transform_edt(mask_np)
        dist_out = distance_transform_edt(1 - mask_np)
        sdf = dist_in - dist_out  # positive inside, negative outside

        gz, gy, gx = np.gradient(sdf)

        # move to torch
        sdf_t = torch.from_numpy(sdf).float().to(device)  # (Z,Y,X)
        grad_t = torch.stack(
            [
                torch.from_numpy(gx).float().to(device),
                torch.from_numpy(gy).float().to(device),
                torch.from_numpy(gz).float().to(device),
            ],
            dim=0,
        )  # (3, Z, Y, X)

        Z, Y, X = sdf_t.shape

        # reshape for grid_sample
        sdf_t = sdf_t.unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)
        grad_t = grad_t.unsqueeze(0)  # (1,3,Z,Y,X)

        # ====== 2) PREPARE POINTS FOR GRID SAMPLE ======
        p = coordinates[b].clone()  # (N,3)
        p0 = p.clone()

        # convert voxel coords -> normalized [-1,1] for grid_sample
        # grid_sample expects (x,y,z) order but your coords are (x,y,z)
        grid = torch.stack(
            [
                2 * p[:, 0] / (X - 1) - 1,
                2 * p[:, 1] / (Y - 1) - 1,
                2 * p[:, 2] / (Z - 1) - 1,
            ],
            dim=-1,
        )  # (N,3)

        grid = grid.view(1, N, 1, 1, 3)  # (1,N,1,1,3)

        # ====== 3) ITERATIVE SDF STEPPING (FULLY PARALLEL) ======
        for _ in range(max_steps):

            # sample sdf and gradient at ALL points at once
            sdf_val = F.grid_sample(sdf_t, grid, align_corners=True).view(N)  # (N,)

            grad_val = F.grid_sample(grad_t, grid, align_corners=True).view(3, N).permute(1, 0)  # (N,3)

            # normalize gradient
            grad_norm = torch.norm(grad_val, dim=-1, keepdim=True) + 1e-8
            grad_val = grad_val / grad_norm

            # step toward zero level-set
            step = step_size * torch.sign(sdf_val).unsqueeze(-1) * grad_val

            # update voxel coords
            p = p - step

            # update normalized grid
            grid = torch.stack(
                [
                    2 * p[:, 0] / (X - 1) - 1,
                    2 * p[:, 1] / (Y - 1) - 1,
                    2 * p[:, 2] / (Z - 1) - 1,
                ],
                dim=-1,
            ).view(1, N, 1, 1, 3)

            # early stop if all close to surface
            if (sdf_val.abs() < tol).all():
                break

        # ====== 4) STORE RESULTS ======
        projected[b] = p
        surface_projection_dist[b] = torch.norm(p - p0, dim=-1)

    # unbatch if needed
    if unbatched_coords:
        projected = projected.squeeze(0)
        surface_projection_dist = surface_projection_dist.squeeze(0)

    return projected, surface_projection_dist


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


def closest_point_on_triangle(p, v0, v1, v2):
    # p, v0, v1, v2: (..., 3)

    e0 = v1 - v0
    e1 = v2 - v0
    v = p - v0

    d00 = (e0 * e0).sum(-1)
    d01 = (e0 * e1).sum(-1)
    d11 = (e1 * e1).sum(-1)
    d20 = (v * e0).sum(-1)
    d21 = (v * e1).sum(-1)

    denom = d00 * d11 - d01 * d01 + 1e-8

    b = (d11 * d20 - d01 * d21) / denom
    c = (d00 * d21 - d01 * d20) / denom
    a = 1.0 - b - c

    # ---- Correct clamping (vector-safe) ----
    b = torch.clamp(b, 0.0, 1.0)
    c = torch.minimum(torch.clamp(c, 0.0, 1.0), 1.0 - b)
    a = 1.0 - b - c
    # ----------------------------------------

    proj = a[..., None] * v0 + b[..., None] * v1 + c[..., None] * v2
    return proj


# -------------------------------------------------------------


def surface_project_coords_marchingcubes_continuous(
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
        projected: (B, N, 3) float  -- TRUE continuous surface points
        surface_projection_dist: (B, N) float
    """

    # ---------------- batching ----------------
    unbatched_coords = coordinates.ndim == 2
    unbatched_surface = surface_mask.ndim == 3

    if unbatched_coords:
        coordinates = coordinates.unsqueeze(0)
    if unbatched_surface:
        surface_mask = surface_mask.unsqueeze(0)

    B, N, _ = coordinates.shape
    device = coordinates.device

    # --------- extract surfaces per batch (verts + faces) ----------
    surfaces = []
    max_M = 0
    max_K = 0

    for b in range(B):
        mask_np = surface_mask[b].detach().cpu().numpy()

        if mask_np.ndim == 4:
            mask_np = mask_np[0]

        if requires_filling:
            mask_np = np_fill_holes(mask_np)

        verts, faces, _, _ = measure.marching_cubes(
            mask_np,
            level=level,
            allow_degenerate=False,
            step_size=1,
        )

        # verts = verts + 0.5  # shift from voxel corner to center

        verts = torch.from_numpy(verts.copy()).float().to(device)
        faces = torch.from_numpy(faces.copy()).long().to(device)

        surfaces.append((verts, faces))
        max_M = max(max_M, verts.shape[0])
        max_K = max(max_K, faces.shape[0])

    # -------- pad to batch tensors ----------
    verts_pad = torch.zeros((B, max_M, 3), device=device)
    faces_pad = torch.zeros((B, max_K, 3), device=device, dtype=torch.long)
    verts_valid = torch.zeros((B, max_M), device=device, dtype=torch.bool)
    faces_valid = torch.zeros((B, max_K), device=device, dtype=torch.bool)

    for b, (v, f) in enumerate(surfaces):
        M = v.shape[0]
        K = f.shape[0]

        verts_pad[b, :M] = v
        faces_pad[b, :K] = f
        verts_valid[b, :M] = True
        faces_valid[b, :K] = True

    # ------------- projection ----------------
    projected = torch.zeros_like(coordinates)
    surface_projection_dist = torch.zeros((B, N), device=device)

    for b in range(B):
        V = verts_pad[b]  # (M,3)
        F = faces_pad[b]  # (K,3)
        Fmask = faces_valid[b]  # (K,)

        for n in range(N):
            p = coordinates[b, n]  # (3,)

            # get valid triangles
            faces = F[Fmask]
            v0 = V[faces[:, 0]]
            v1 = V[faces[:, 1]]
            v2 = V[faces[:, 2]]

            # project to all triangles
            p_expand = p.unsqueeze(0).expand_as(v0)
            proj_all = closest_point_on_triangle(p_expand, v0, v1, v2)

            # distances
            dist = ((proj_all - p_expand) ** 2).sum(-1)

            # best triangle
            best_k = dist.argmin()
            best_proj = proj_all[best_k]

            projected[b, n] = best_proj
            surface_projection_dist[b, n] = torch.sqrt(dist[best_k])

    # -------- unbatch if needed ----------
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
