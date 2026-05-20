import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))
from pathlib import Path
from TPTBox import No_Logger, Log_Type, NII, BIDS_FILE, POI, Location, Vertebra_Instance, v_idx_order
from TPTBox.core.vert_constants import COORDINATE
from joblib import Parallel, delayed
from panoptica import Metric
from utils.filepaths import search_path_single, search_path
import numpy as np
from utils.misc import surface_project_poi
from utils.vertebra_rotation import rotate_3darray, calc_orientation_from_poi
from report_utils import (
    LogicReport,
    SPATIAL_LOGIC_CONSTRAINTS_DICT,
    SUBREGION_CONSTRAINT_DICT,
    poi_touches_subreg,
    save_logic_report,
    SUBREGION_SOFTCONSTRAINT_DICT,
)
from dataclasses import dataclass, field
from TypeSaveArgParse import Class_to_ArgParse
from tqdm import tqdm

logger = No_Logger(prefix="DStats")


@dataclass
class InferenceConfig(Class_to_ArgParse):
    root: Path = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/")
    derivatives: str = "derivatives_combined"
    datasets: list[str] = field(
        default_factory=lambda: [
            "dataset-verse19training_1mmiso",
            "dataset-verse20training_1mmiso",
            "dataset-verse19validation_1mmiso",
            "dataset-verse20validation_1mmiso",
            "dataset-verse19test_1mmiso",
            "dataset-verse20test_1mmiso",
        ]
    )

    vertebra_from: int = 6


if __name__ == "__main__":
    config = InferenceConfig.get_opt()
    logger.print(f"Using config: {config}")

    samples_data = []

    for ds_name in config.datasets:
        logger.print(f"Processing dataset: {ds_name}", Log_Type.STAGE)
        basepath = config.root.joinpath(ds_name)
        if not basepath.exists():
            logger.print(f"Dataset path {basepath} does not exist. Skipping.", Log_Type.WARNING)
            continue

        derivatives_dir = basepath.joinpath(config.derivatives)
        if not derivatives_dir.exists():
            logger.print(f"Derivatives path {derivatives_dir} does not exist. Skipping.", Log_Type.WARNING)
            continue

        nii_ps = search_path(derivatives_dir, query="**/*_seg-vert*_msk.nii.gz")
        logger.print(f"Found {len(nii_ps)} NII files in {derivatives_dir}")

        for nii_p in tqdm(nii_ps, desc=f"Processing {ds_name}"):
            nii = NII.load(nii_p, seg=True)
            vert_labels = [v for v in nii.unique() if v > 0 and (v < 26 or v == 28) and v >= config.vertebra_from]
            # logger.print(f"File: {nii_p}, Vertebrae found: {vert_labels}")

            samples_data.append((ds_name, nii_p, vert_labels))

    # Now samples_data contains tuples of (dataset_name, nii_path, list_of_vertebrae)
    logger.print()
    logger.print(f"Found a total of {len(samples_data)} samples across all datasets.")
    n_vertebra = sum(len(vert_labels) for _, _, vert_labels in samples_data)
    logger.print(f"Total number of vertebrae across all samples: {n_vertebra}")
