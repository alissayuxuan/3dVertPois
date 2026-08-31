"""Registry of trained model checkpoints, addressable by short name.

Each member records a training directory, a logger version and a checkpoint file
name, relative to the configured ``model_root`` (see :mod:`vertpois.paths`). The
checkpoints themselves are not distributed with this repository.
"""

from enum import Enum

from vertpois.paths import get_path


class TrainedModelInfo(Enum):
    @property
    def model_dir(self):
        return self.value[0]

    @property
    def version(self):
        return self.value[1]

    @property
    def checkpoint_name(self):
        return self.value[2]

    @property
    def data_module_params_path(self):
        return str(get_path("model_root") / self.model_dir / f"version_{self.version}" / "data_module_params.json")

    @property
    def model_path(self):
        return str(get_path("model_root") / self.model_dir / f"version_{self.version}" / "checkpoints" / f"{self.checkpoint_name}.ckpt")

    # POST MICCAI (with more sorted out for training)
    T_S_SURFACE = (
        "ForVerse/surface_cc3",
        0,
        "sad-pt-epoch=104-fine_mean_distance_val=1.85",
    )

    T_S_SURFACE_HIGHBS = (
        "ForVerse/surface_cc3-bs32",
        1,
        "sad-pt-epoch=133-fine_mean_distance_val=1.68",
    )

    T_S_SURFACE_HIGHBS2 = (
        "ForVerse/surface_cc3-bs32",
        3,
        "sad-pt-epoch=175-fine_mean_distance_val=1.45",
    )

    T_S_SURFACE_HIGHBS3 = (
        "ForVerse/surface_cc3-bs32",
        4,
        "sad-pt-epoch=108-fine_mean_distance_val=1.70",
    )

    T_N_SURFACE = (
        "ForVerse/surface_neighbor_cc3",
        1,
        "sad-pt-epoch=52-fine_mean_distance_val=4.22",
    )

    T_S_SURFACE_FIRST = (
        "FirstIter/surface_cc3-bs32",
        3,
        "sad-pt-epoch=134-fine_mean_distance_val=1.60",
    )

    T_S_SURFACE_FIRST2 = (
        "FirstIter/surface_cc3-bs32",
        5,
        "sad-pt-epoch=199-fine_mean_distance_val=1.11",
    )

    T_S_SURFACE_SEC = (
        "SecondIter/surface_cc3-bs32",
        0,
        "sad-pt-epoch=21-fine_mean_distance_val=1.51",
    )

    T_S_SURFACE_SEC2 = (
        "SecondIter/surface_cc3-bs32",
        1,
        "sad-pt-epoch=34-fine_mean_distance_val=1.39",
    )

    T_S_SURFACE_SEC3 = (
        "SecondIter/surface_cc3-bs32",
        2,
        "sad-pt-epoch=19-fine_mean_distance_val=1.38",
    )

    T_S_SURFACE_SWEEP2WINNER = (
        "SecondIterSweep2Winner/winner/surface_cc3",
        0,
        "sad-pt-epoch=194-fine_mean_distance_val=1.00",
    )

    ####################################################
    # Single best training MAE
    GRUBER_S_SURFACE = (
        "ForVerse/surface_project-gt_cc3-exclude6",
        0,
        "sad-pt-epoch=131-fine_mean_distance_val=1.50",
    )
    GRUBER_S_SURFACE_NEIGHSHOW = (
        "ForVerse/surface_project-gt_cc3-exclude6_neighborshow",
        0,
        "sad-pt-epoch=58-fine_mean_distance_val=1.60",
    )  # Needs inference code adaption
    GRUBER_N_SURFACE_NEIGHAUG = (
        "ForVerse/surface-neighbor-neighaug-project_gt_cc3-exclude6",
        0,
        "sad-pt-epoch=46-fine_mean_distance_val=3.70",
    )
    GRUBER_N_SURFACE_NEIGHAUG2 = (
        "ForVerse/surface-neighbor-neighaug-project_gt_cc3-exclude6",
        2,
        "sad-pt-epoch=70-fine_mean_distance_val=2.81",
    )
    GRUBER_N_SURFACE_NOVERT = (
        "ForVerse/surface-neighbor-no_vert-project_gt_cc3-exclude6",
        0,
        "sad-pt-epoch=109-fine_mean_distance_val=2.60",
    )
    GRUBER_N_SURFACE_NOVERT_NEIGHAUG = (
        "ForVerse/surface-neighbor-no_vert-project_gt_cc3-exclude6",
        1,
        "sad-pt-epoch=111-fine_mean_distance_val=3.52",
    )

    ######################
    # OLD
    ######################
    GRUBER_N_BOUNDARYBROKEN = (
        "ForVerse/surface-neighbor-project_gt_cc3-exclude6",
        1,
        "sad-pt-epoch=88-fine_mean_distance_val=2.34",
    )
