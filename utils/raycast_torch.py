from __future__ import annotations

from typing import Literal

import torch


def trilinear_interpolate(volume, x, y, z):
    xi, yi, zi = int(x), int(y), int(z)
    if xi < 0 or yi < 0 or zi < 0 or xi >= volume.shape[0] - 1 or yi >= volume.shape[1] - 1 or zi >= volume.shape[2] - 1:
        return 0.0

    xd, yd, zd = x - xi, y - yi, z - zi
    c000 = volume[xi, yi, zi]
    c100 = volume[xi + 1, yi, zi]
    c010 = volume[xi, yi + 1, zi]
    c110 = volume[xi + 1, yi + 1, zi]
    c001 = volume[xi, yi, zi + 1]
    c101 = volume[xi + 1, yi, zi + 1]
    c011 = volume[xi, yi + 1, zi + 1]
    c111 = volume[xi + 1, yi + 1, zi + 1]

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd
    return c0 * (1 - zd) + c1 * zd


# @njit(fastmath=True)
def max_distance_ray_cast_convex_torch(
    region_array: torch.Tensor,
    start_coord: torch.Tensor,
    direction_vector: torch.Tensor,
    acc_delta=0.05,
    max_iter=100,
):
    region_array.detach()
    start_coord.detach()
    direction_vector.detach()
    # Normalize direction
    norm_vec = direction_vector / torch.sqrt((direction_vector**2).sum())
    norm_vec.detach()

    # Quick exit if start point is outside
    if trilinear_interpolate(region_array, *start_coord) <= 0.5:
        return start_coord

    min_v = 0.0
    max_v = torch.sum(torch.tensor(region_array.shape))
    delta = max_v - min_v

    while delta > acc_delta and max_iter > 0:
        mid = 0.5 * (max_v + min_v)
        x = start_coord[0] + norm_vec[0] * mid
        y = start_coord[1] + norm_vec[1] * mid
        z = start_coord[2] + norm_vec[2] * mid
        val = trilinear_interpolate(region_array, x, y, z)
        if val > 0.5:
            min_v = mid
        else:
            max_v = mid
        delta = max_v - min_v
        max_iter -= 1

    dist = 0.5 * (min_v + max_v)
    new_coord = torch.tensor(
        [
            start_coord[0] + norm_vec[0] * dist,
            start_coord[1] + norm_vec[1] * dist,
            start_coord[2] + norm_vec[2] * dist,
        ]
    )
    # print(start_coord, direction_vector, new_coord, "\n")
    return new_coord
