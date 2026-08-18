import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))
from TPTBox import BIDS_FILE, BIDS_Global_info, No_Logger, NII, np_utils, Log_Type, Vertebra_Instance, Location, POI
import numpy as np
import math
import torch
from TPTBox import calc_poi_labeled_buffered
from TPTBox.core.vert_constants import DIRECTIONS, COORDINATE


# function that yields matrix to reorient to PIR coordinate orientation
def radian_to_degrees(radian: float) -> float:
    return math.degrees(radian)


def calc_orientation_poi(vert_nii: NII, sem_nii: NII, poi_out: str | Path | None = None):
    return calc_poi_labeled_buffered(
        vert_nii,
        sem_nii,
        subreg_id=[Location.Vertebra_Corpus, Location.Vertebra_Direction_Posterior],
        out_path=poi_out,
    )


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def unit_vector(v):
    return v / np.linalg.norm(v)


PIR_GLOBAL_VECTORS = {
    Location.Vertebra_Direction_Posterior.value: np.array([1, 0, 0]),
    Location.Vertebra_Direction_Inferior.value: np.array([0, 1, 0]),
    Location.Vertebra_Direction_Right.value: np.array([0, 0, 1]),
}


def repair_rel_to_corpus(
    rel_to_corpus: dict,
    tol_deg: float = 10.0,
    implausible_deg: float = 60.0,
    min_margin_deg: float = 20.0,
) -> tuple[dict | None, str]:
    """Detect and repair a broken vertebra direction vector.

    The three `Vertebra_Direction_*` POIs are supposed to span an orthonormal frame, but
    nothing upstream enforces that. When one of them is wrong the frame becomes skewed and
    near-singular, and `coord @ R` stops meaning "components along posterior/inferior/right"
    - which silently flips every spatial-logic check for that vertebra.

    Measured over 4145 vertebrae of the verse datasets: healthy frames are orthogonal to
    within 2 degrees, 83 are off by more than 10, and in *every* one of those the offending
    pair is (posterior, inferior) - the right vector is always clean. So orthogonality alone
    can never single out the culprit; it always leaves two candidates. The tie-break is the
    angle to the corresponding global axis: a "posterior" direction 90-112 degrees away from
    global posterior is anatomically impossible, whereas 17-41 degrees for inferior is a
    normal vertebral tilt.

    Once the culprit is known the other two vectors *determine* it, so it is rebuilt as their
    cross product (sign matched to the global axis) rather than replaced by the global axis -
    that keeps the vertebra's real tilt. Checked against the same vector in the nearest
    healthy neighbouring vertebra, the reconstruction sits a median 10 degrees away where the
    broken original sat 61 degrees away.

    Returns `(rel_to_corpus, status)`:
      * `(unchanged, "ok")`                  - frame is orthogonal within `tol_deg`
      * `(repaired, "repaired:<axis>")`      - one vector rebuilt from the other two
      * `(None, "degenerate")`               - culprit not identifiable; the caller should
                                               fall back to the global image axes entirely
    """
    keys = sorted(rel_to_corpus)
    if len(keys) != 3:
        return None, "degenerate"

    def deviation_from_orthogonal(a, b) -> float:
        return abs(90.0 - radian_to_degrees(angle_between(rel_to_corpus[a], rel_to_corpus[b])))

    bad_pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1 :] if deviation_from_orthogonal(a, b) > tol_deg]
    if not bad_pairs:
        return rel_to_corpus, "ok"

    # candidates = vectors implicated in every bad pair
    candidates = [k for k in keys if all(k in pair for pair in bad_pairs)]
    if len(candidates) != 2:
        return None, "degenerate"

    to_global = {k: radian_to_degrees(angle_between(rel_to_corpus[k], PIR_GLOBAL_VECTORS[k])) for k in candidates}
    culprit = max(to_global, key=to_global.get)
    other = min(to_global, key=to_global.get)
    # Only act when one candidate is both anatomically implausible on its own and clearly
    # worse than the other; otherwise the two are merely inconsistent and guessing which one
    # to overwrite would be a coin flip.
    if to_global[culprit] < implausible_deg or (to_global[culprit] - to_global[other]) < min_margin_deg:
        return None, "degenerate"

    good = [k for k in keys if k != culprit]
    rebuilt = np.cross(rel_to_corpus[good[0]], rel_to_corpus[good[1]])
    norm = np.linalg.norm(rebuilt)
    if norm < 1e-6:
        return None, "degenerate"
    rebuilt = rebuilt / norm
    alignment = float(np.dot(rebuilt, PIR_GLOBAL_VECTORS[culprit]))
    if abs(alignment) < 0.2:  # cannot orient the reconstruction reliably
        return None, "degenerate"
    if alignment < 0:
        rebuilt = -rebuilt

    repaired = dict(rel_to_corpus)
    repaired[culprit] = rebuilt
    return repaired, f"repaired:{Location(culprit).name.replace('Vertebra_Direction_', '')}"


def calc_orientation_from_poi_checked(poi: POI, region: int, repair_frame: bool = True, **repair_kwargs):
    """`calc_orientation_from_poi` plus the frame-validity status.

    On `"degenerate"` the rotation is the identity and `rel_to_corpus` is None, which makes
    every downstream helper (`get_axis_direction_vector`, the rotation in report_poi.py) fall
    back to the global image axes instead of a frame that cannot be trusted.
    """
    poi_v: POI = poi.extract_region(region)

    point_keys = [
        Location.Vertebra_Corpus,
        Location.Vertebra_Direction_Posterior,
        Location.Vertebra_Direction_Inferior,
        Location.Vertebra_Direction_Right,
    ]
    point_keys = [i.value for i in point_keys]
    points = {s: np.asarray(v) for r, s, v in poi_v.items() if s in point_keys}

    # A vertebra without all four reference points has no frame at all. This happens at the
    # outermost vertebrae of a scan, which is exactly where callers used to avoid asking.
    if any(k not in points for k in point_keys):
        missing = [Location(k).name for k in point_keys if k not in points]
        corpus_com = points.get(Location.Vertebra_Corpus.value, np.zeros(3))
        return np.eye(3), corpus_com, None, [], f"missing:{','.join(missing)}"

    # print("points", points)
    # calc corpus - three other to get directional vectors (and normalize)
    rel_to_corpus = {
        s: unit_vector(v - points[Location.Vertebra_Corpus.value]) for s, v in points.items() if s != Location.Vertebra_Corpus.value
    }

    status = "ok"
    if repair_frame:
        repaired, status = repair_rel_to_corpus(rel_to_corpus, **repair_kwargs)
        if repaired is None:
            corpus_com = points[Location.Vertebra_Corpus.value]
            return np.eye(3), corpus_com, None, [], status
        rel_to_corpus = repaired

    PIR_angles = [angle_between(v, PIR_GLOBAL_VECTORS[s]) for s, v in rel_to_corpus.items()]
    PIR_angle_degrees = [radian_to_degrees(i) for i in PIR_angles]
    # print("PIR_angle_degrees", PIR_angle_degrees)
    # print()

    # R's columns are the direction vectors in ascending Location order, i.e. posterior,
    # inferior, right - matching the PIR image orientation everything is reoriented to, so
    # `coord @ R` lands in the same axis order the callers' `get_axis` assumes.
    R = np.asarray([[rel_to_corpus[s][idx] for s in sorted(rel_to_corpus)] for idx in range(3)])
    corpus_com = points[Location.Vertebra_Corpus.value]
    # print("R", R)
    # then orientation from that?
    # from that orientation calc matrix to inverse-orient basically?
    return R, corpus_com, rel_to_corpus, PIR_angle_degrees, status


def calc_orientation_from_poi(poi: POI, region: int, repair_frame: bool = True, **repair_kwargs):
    return calc_orientation_from_poi_checked(poi, region, repair_frame=repair_frame, **repair_kwargs)[:4]


def get_axis_direction_from_rel_to_corpus(rel_to_corpus, dir: DIRECTIONS):
    if dir == "A":
        return rel_to_corpus[Location.Vertebra_Direction_Posterior.value] * -1
    elif dir == "P":
        return rel_to_corpus[Location.Vertebra_Direction_Posterior.value]
    elif dir == "S":
        return rel_to_corpus[Location.Vertebra_Direction_Inferior.value] * -1
    elif dir == "I":
        return rel_to_corpus[Location.Vertebra_Direction_Inferior.value]
    elif dir == "L":
        return rel_to_corpus[Location.Vertebra_Direction_Right.value] * -1
    elif dir == "R":
        return rel_to_corpus[Location.Vertebra_Direction_Right.value]
    else:
        raise ValueError(f"Unknown direction {dir}")


def get_axis_direction_vector(rel_to_corpus, poi_ref, axis: DIRECTIONS):
    if rel_to_corpus is None:
        axis_idx = poi_ref.get_axis(axis)
        inversed = poi_ref.orientation[axis_idx] != axis
        vector = np.zeros(3)
        vector[axis_idx] = -1 if inversed else 1
    else:
        vector = get_axis_direction_from_rel_to_corpus(rel_to_corpus, axis)
    return vector


def move_poi_along_axis(c: COORDINATE | np.ndarray, poi_ref, axis: DIRECTIONS, rel_to_corpus: dict, distance_voxel: int):
    vector = get_axis_direction_vector(rel_to_corpus, poi_ref, axis)
    return np.asarray(c) + vector * distance_voxel


def find_extreme_point_along_axis(
    poi_coords: list[COORDINATE | np.ndarray],
    poi_ref,
    axis: DIRECTIONS,
    rel_to_corpus: dict,
    trim: float | None = None,
):
    """The candidate that lies furthest along `axis`.

    With `trim` set (in voxels) and at least three candidates, the plain `argmax` is
    replaced by a trimmed one: only candidates within `trim` of the median projection are
    eligible. A plain maximum over N noisy predictions is driven by whichever prediction is
    most wrong, which is what puts a single point off the corpus edge and produces the
    sharp-bend reports; the trimmed version takes the most extreme *plausible* candidate.
    """
    vector = get_axis_direction_vector(rel_to_corpus, poi_ref, axis)
    projections = np.asarray([np.dot(np.asarray(c), vector) for c in poi_coords])
    if trim is not None and len(projections) >= 3:
        median = float(np.median(projections))
        eligible = np.flatnonzero(projections <= median + trim)
        if len(eligible) > 0:
            return poi_coords[int(eligible[np.argmax(projections[eligible])])]
    extreme_idx = int(np.argmax(projections))
    return poi_coords[extreme_idx]


def rotate_3darray(array, rotation, center: tuple | None = None):
    # rotate the 3D numpy array using given parameters around a defined center
    # create meshgrid
    from scipy.ndimage import map_coordinates

    dim = array.shape
    ax = np.arange(dim[0])
    ay = np.arange(dim[1])
    az = np.arange(dim[2])
    coords = np.meshgrid(ax, ay, az)

    if center is None:
        # adjust coords to center
        center = (float(dim[0]) / 2, float(dim[1]) / 2, float(dim[2]) / 2)

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz = np.vstack(
        [
            coords[0].reshape(-1) - center[0],  # x coordinate, centered
            coords[1].reshape(-1) - center[1],  # y coordinate, centered
            coords[2].reshape(-1) - center[2],  # z coordinate, centered
        ]
    )  # z coordinate, centered

    r = rotation
    mat = r.copy()  # r.as_matrix()

    # apply transformation
    transformed_xyz = np.dot(mat, xyz)

    # extract coordinates
    x = transformed_xyz[0, :] + center[0]
    y = transformed_xyz[1, :] + center[1]
    z = transformed_xyz[2, :] + center[2]

    x = x.reshape((dim[1], dim[0], dim[2]))
    y = y.reshape((dim[1], dim[0], dim[2]))
    z = z.reshape((dim[1], dim[0], dim[2]))
    new_xyz = [x, y, z]

    # sample
    # arrayR = map_coordinates(array, new_xyz, order=0, mode='nearest')
    arrayR = map_coordinates(array, new_xyz, order=0, mode="constant")
    arrayR = np.swapaxes(arrayR, 0, 1)
    return arrayR
