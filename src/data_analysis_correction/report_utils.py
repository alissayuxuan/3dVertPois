import pandas as pd
from dataclasses import dataclass
from TPTBox import Location, NII, POI, Vertebra_Instance, No_Logger, Log_Type
from TPTBox.core.vert_constants import DIRECTIONS, COORDINATE
from pathlib import Path
import numpy as np

logger = No_Logger(prefix="")


@dataclass
class LogicReport:
    subject_name: str
    vertebra: int | Vertebra_Instance
    location: Location
    severity: float
    relevant_data: dict
    description: str

    def __str__(self):
        return f"{self.subject_name}-v={self.vertebra}-{self.location, self.location.value}: {self.description}\n ({self.severity}, {self.relevant_data})"

    def get_dict(self, round_d=3) -> dict:
        relevant_data_rounded = {
            k: (
                round(v, round_d)
                if isinstance(v, float)
                else (
                    [round(vv, round_d) if isinstance(vv, float) else vv for vv in v]
                    if (isinstance(v, list) or isinstance(v, tuple) or isinstance(v, np.ndarray))
                    else v
                )
            )
            for k, v in self.relevant_data.items()
        }
        return {
            "subject_name": self.subject_name,
            "vertebra": self.vertebra,
            "location": self.location.value,
            "severity": round(self.severity, round_d),
            "relevant_data": relevant_data_rounded,
            "description": self.description,
        }


def save_logic_report(report_list: list[LogicReport], save_path: str | Path, round_d=3) -> None:
    """Save logic report to CSV file."""
    df = pd.DataFrame([r.get_dict(round_d=round_d) for r in report_list])
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
                try:
                    voxel_value = subregion_nii[neighbor_coord]
                except IndexError:
                    continue
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

    def get_singular_str(self, loc1: Location | int, axis: DIRECTIONS, loc2: Location | int) -> str:
        return f"Spatial LC({loc1.value if isinstance(loc1, Location) else loc1} in {axis} > {loc2.value if isinstance(loc2, Location) else loc2})"


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
        [112, 104, 120, 118, 102, 110, 106, 122, 125, 82, 83],
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
        [86, 81, 87, 127, 125],
    ),
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
    SpatialLC(42, "S", 81),
    SpatialLC(42, "A", 81),
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
    soft_subregion: int | list[int] | None = None

    """Means location1 must any of the given subregions
    """

    def __str__(self):
        return f"Subregion LC({self.location1} touches {self.subregion})"


# which subregion the POI may touch
SUBREGION_CONSTRAINT = [
    SubregionLC(
        [117, 101, 109, 124, 108, 116, 103, 111, 119, 84, 85, 121, 105, 113, 102],
        [49, 50],
        [41],
    ),
    SubregionLC(
        [110, 118],
        [49, 50, 41],
    ),
    SubregionLC(81, 42, 41),
    SubregionLC(83, 43, 41),
    SubregionLC(82, 44, 41),
    SubregionLC(88, 45, 41),
    SubregionLC(89, 46, 41),
    SubregionLC(86, 47, 41),
    SubregionLC(87, 48, 41),
]

SUBREGION_CONSTRAINT_DICT = {
    loc: lc.subregion if isinstance(lc.subregion, list) else [lc.subregion]
    for lc in SUBREGION_CONSTRAINT
    for loc in (lc.location1 if isinstance(lc.location1, list) else [lc.location1])
}
SUBREGION_SOFTCONSTRAINT_DICT = {
    loc: lc.soft_subregion if isinstance(lc.soft_subregion, list) else [lc.soft_subregion]
    for lc in SUBREGION_CONSTRAINT
    for loc in (lc.location1 if isinstance(lc.location1, list) else [lc.location1])
}


def load_agg_report_df(root: str, prefix: str, der_name: str, file_name: str = "aggregated_poi_report.xlsx") -> pd.DataFrame | None:
    report_path = Path(root).joinpath(f"{prefix}{der_name}", file_name)
    if not report_path.exists():
        logger.print(f"Report {report_path} does not exist", Log_Type.FAIL)
        return None
    df = pd.read_excel(report_path)
    return df


def convert_agg_report_to_reported_bool_dict(
    df: pd.DataFrame,
    severity_threshold=0.0,
    vertebra_from: int = 8,
) -> dict[str, dict[int, dict[Location, bool]]]:
    report_dict = {}
    for _, row in df.iterrows():
        subject_name = row["subject_name"]
        vert = row["vertebra"]
        if vert < vertebra_from:
            continue
        location = Location(row["location"])
        severity = row["severity"]
        if subject_name not in report_dict:
            report_dict[subject_name] = {}
        if severity >= severity_threshold:
            report_dict[subject_name][(vert, location)] = True
    return report_dict


def is_poi_reported(report_dict, subject_ct_id, vert, location):
    if subject_ct_id in report_dict:
        if (vert, location) in report_dict[subject_ct_id]:
            return report_dict[subject_ct_id][(vert, location)]
    return False


def is_vert_reported(report_dict, subject_ct_id, vert):
    if subject_ct_id in report_dict:
        for v, location in report_dict[subject_ct_id]:
            if v == vert:
                return True
    return False


def _report_counts(df) -> tuple[int, int]:
    """Return (#reported POIs, #subjects with >=1 reported error) for a report df."""
    report_dict = convert_agg_report_to_reported_bool_dict(df)
    sum_per_subject = {s: len(v) for s, v in report_dict.items()}
    n_pois = sum(sum_per_subject.values())
    n_subjects_with_errors = sum(1 for v in sum_per_subject.values() if v > 0)
    return n_pois, n_subjects_with_errors


def print_one(df):
    # get unique ds names
    ds_names = df["ds"].unique()
    ds_names = [
        ds.split("dataset-")[-1]
        .split("_1mmiso")[0]
        .replace("verse", "v-")
        .replace("training", "-T")
        .replace("validation", "-V")
        .replace("test", "-TS")
        for ds in ds_names
    ]

    # Total across both report sources (deduplicated per subject by (vert, location))
    total_pois, total_subjects = _report_counts(df)
    lines = [f"Reported errors ({ds_names}):", f"#POIs {total_pois}\n#VERT {total_subjects}"]

    # Per-source breakdown (if the source was tagged before concatenation). Note: the
    # per-source counts can overlap, so their sum may exceed the deduplicated total.
    if "source" in df.columns:
        for source in df["source"].dropna().unique():
            src_pois, src_subjects = _report_counts(df[df["source"] == source])
            lines.append(f"  [{source}] #POIs {src_pois}  #VERT {src_subjects}")

    logger.print("\n".join(lines))
    print()


if __name__ == "__main__":
    ROOT = Path("/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/data_analysis/")
    PREFIX = "TEST_"

    der_names = ROOT.glob(f"{PREFIX}*")

    for der in der_names:
        if not der.is_dir():
            continue
        if "derivatives" not in der.name:
            continue
        if "poi" not in der.name:
            continue

        der = der.name.split(f"{PREFIX}")[1]
        logger.print(f"{der}", Log_Type.STAGE)

        df = load_agg_report_df(str(ROOT), PREFIX, der)
        if df is None:
            # subfolder logic
            logger.print(f"Loading report from subfolder: {der}", Log_Type.STAGE)
            for subfolder in (ROOT / f"{PREFIX+der}").iterdir():
                print(subfolder)
                df = load_agg_report_df(str(subfolder), "", "", "aggregated_poi_report.xlsx")
                df2 = load_agg_report_df(str(subfolder), "", "", "aggregated_poi_neighbor_angle_report.xlsx")
                if df is not None:
                    df["source"] = "normal"
                if df2 is not None:
                    df2["source"] = "angle"
                    # combine df and df2
                    df = pd.concat([df, df2], ignore_index=True)
                else:
                    continue
                print_one(df)
        else:
            df["source"] = "normal"
            df2 = load_agg_report_df(str(ROOT), PREFIX, der, "aggregated_poi_neighbor_angle_report.xlsx")
            if df2 is not None:
                df2["source"] = "angle"
                # combine df and df2
                df = pd.concat([df, df2], ignore_index=True)
            else:
                continue
            print_one(df)

        # break
