import pandas as pd
from dataclasses import dataclass
from TPTBox import Location, NII, POI, Vertebra_Instance, No_Logger, Log_Type
from TPTBox.core.vert_constants import DIRECTIONS, COORDINATE
from pathlib import Path

logger = No_Logger(prefix="logic_report")


@dataclass
class LogicReport:
    subject_name: str
    vertebra: int | Vertebra_Instance
    location: Location
    relevant_data: dict
    description: str

    def __str__(self):
        return f"{self.subject_name}-v={self.vertebra}-{self.location, self.location.value}: {self.description}\n ({self.relevant_data})"

    def get_dict(self):
        return {
            "subject_name": self.subject_name,
            "vertebra": self.vertebra,
            "location": self.location.value,
            "relevant_data": self.relevant_data,
            "description": self.description,
        }


def save_logic_report(report_list: list[LogicReport], save_path: str | Path) -> None:
    """Save logic report to CSV file."""
    df = pd.DataFrame([r.get_dict() for r in report_list])
    df.to_excel(save_path, index=False)
    logger.print(f"Logic report saved to {save_path}", Log_Type.SAVE)


def poi_touches_subreg(coord: COORDINATE, subregion_nii: NII):
    rounded_coord = tuple(int(round(c)) for c in coord)
    counts = {}
    dw = [-2, -1, 0, 1, 2]
    # check 26-neighborhood
    for dx in dw:
        for dy in dw:
            for dz in dw:
                neighbor_coord = (rounded_coord[0] + dx, rounded_coord[1] + dy, rounded_coord[2] + dz)
                voxel_value = subregion_nii[neighbor_coord]
                if voxel_value != 0:
                    if voxel_value not in counts:
                        counts[voxel_value] = 0
                    counts[voxel_value] += 1
    if counts:
        # return the subregion with the most touching voxels
        return max(counts, key=counts.get)
    return None


@dataclass
class SpatialLC:
    location1: Location | int | list[Location | int]
    axis: DIRECTIONS | list[DIRECTIONS]
    location2: Location | int | list[Location | int]

    """Means location1 is in the corresponding axis higher than location2
    """

    def __str__(self):
        return f"Spatial LC({self.location1} in {self.axis} > {self.location2})"


SPATIAL_LOGIC_CONSTRAINTS = [
    # Anterior Corpus
    SpatialLC(
        [117, 101, 109, 124, 108, 116, 103, 111, 119],
        "A",
        [121, 105, 113, 85, 84, 123, 107, 115],
    ),
    SpatialLC(
        [121, 105, 113, 85, 84, 123, 107, 115],
        "A",
        [112, 104, 120, 118, 102, 110, 106, 122, 125],
    ),
    # Right Corpus
    SpatialLC(
        [85, 118, 121, 117, 124, 119],
        "R",
        [102, 105, 101, 108, 103, 104, 106, 107],
    ),
    SpatialLC(
        [102, 105, 101, 108, 103, 104, 106, 107],
        "R",
        [110, 113, 109, 116, 111, 84, 115, 112, 114],
    ),
    # Superior Corpus
    SpatialLC(
        [118, 102, 110, 121, 105, 113, 117, 101, 109],
        "S",
        [124, 108, 116, 84, 85, 114, 106, 122],
    ),
    SpatialLC(
        [124, 108, 116, 84, 85, 114, 106, 122],
        "S",
        [119, 103, 111, 123, 107, 115, 120, 104, 112],
    ),
    # Corpus to Posterior
    SpatialLC(
        [118, 102, 110, 114, 106, 122, 112, 104, 120],
        "A",
        [82, 83, 86, 81, 87, 127, 125],
    ),  # TODO 88,89 would require rotation
    SpatialLC([89, 88], "S", [118, 102, 110]),
    # Posterior elements to each other
    SpatialLC(82, "R", 89),
    SpatialLC(89, "R", 88),
    SpatialLC(88, "R", 83),
    SpatialLC(87, "R", 86),
    # Posterior Elements Superior
    SpatialLC([88, 89], "S", [83, 82]),
    SpatialLC([83, 82], "S", [86, 87]),
    # SpatialLC([86, 87], "S", [81]), #TODO <-- requires rotation
    # COM constraints
    SpatialLC(42, ["S", "A"], 81),
    SpatialLC(43, "R", 83),
    SpatialLC(44, "L", 82),
    SpatialLC([45, 46], "I", [88, 89]),
    SpatialLC([47, 48], "S", [86, 87]),
]

SPATIAL_LOGIC_CONSTRAINTS_DICT = {}
for lc in SPATIAL_LOGIC_CONSTRAINTS:
    for loc1 in lc.location1 if isinstance(lc.location1, list) else [lc.location1]:
        if loc1 not in SPATIAL_LOGIC_CONSTRAINTS_DICT:
            SPATIAL_LOGIC_CONSTRAINTS_DICT[loc1] = []
        SPATIAL_LOGIC_CONSTRAINTS_DICT[loc1].append(lc)


@dataclass
class SubregionLC:
    location1: Location | int | list[Location | int]
    subregion: int | list[int]

    """Means location1 must any of the given subregions
    """

    def __str__(self):
        return f"Subregion LC({self.location1} touches {self.subregion})"


# which subregion the POI may touch
SUBREGION_CONSTRAINT = [
    SubregionLC(
        [117, 101, 109, 124, 108, 116, 103, 111, 119, 84, 85, 121, 105, 113, 102],
        [49, 50],
    ),
    SubregionLC(
        [110, 118],
        [49, 50, 41],
    ),
    SubregionLC(81, 42),
    SubregionLC(83, 43),
    SubregionLC(82, 44),
    SubregionLC(88, 45),
    SubregionLC(89, 46),
    SubregionLC(86, 47),
    SubregionLC(87, 48),
]

SUBREGION_CONSTRAINT_DICT = {
    loc: lc.subregion if isinstance(lc.subregion, list) else [lc.subregion]
    for lc in SUBREGION_CONSTRAINT
    for loc in (lc.location1 if isinstance(lc.location1, list) else [lc.location1])
}
