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


def calc_orientation_from_poi(poi: POI, region: int):
    poi_v: POI = poi.extract_region(region)

    point_keys = [
        Location.Vertebra_Corpus,
        Location.Vertebra_Direction_Posterior,
        Location.Vertebra_Direction_Inferior,
        Location.Vertebra_Direction_Right,
    ]
    point_keys = [i.value for i in point_keys]
    points = {s: np.asarray(v) for r, s, v in poi_v.items() if s in point_keys}
    # print("points", points)
    # calc corpus - three other to get directional vectors (and normalize)
    rel_to_corpus = {
        s: unit_vector(v - points[Location.Vertebra_Corpus.value]) for s, v in points.items() if s != Location.Vertebra_Corpus.value
    }
    pir_global_vectors = {
        Location.Vertebra_Direction_Posterior.value: np.array([1, 0, 0]),
        Location.Vertebra_Direction_Inferior.value: np.array([0, 1, 0]),
        Location.Vertebra_Direction_Right.value: np.array([0, 0, 1]),
    }
    PIR_angles = [angle_between(v, pir_global_vectors[s]) for s, v in rel_to_corpus.items()]
    PIR_angle_degrees = [radian_to_degrees(i) for i in PIR_angles]
    # print("PIR_angle_degrees", PIR_angle_degrees)
    # print()

    # R = [x_x, y_x, z_x; x_y, y_y, y_z; z_x, z_y, z_z]
    # print("rel_to_corpus", rel_to_corpus)
    R = np.asarray([[v[idx] for v in rel_to_corpus.values()] for idx in range(3)])
    corpus_com = points[Location.Vertebra_Corpus.value]
    # print("R", R)
    # then orientation from that?
    # from that orientation calc matrix to inverse-orient basically?
    return R, corpus_com, rel_to_corpus, PIR_angle_degrees


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
