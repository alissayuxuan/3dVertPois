from enum import Enum

MODEL_ROOT = "/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/src/hendrik/trainings/largefov_pois-cc3-exclude6/"


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
        return f"{MODEL_ROOT}/{self.model_dir}/version_{self.version}/data_module_params.json"

    @property
    def model_path(self):
        return f"{MODEL_ROOT}/{self.model_dir}/version_{self.version}/checkpoints/{self.checkpoint_name}.ckpt"

    # Single best training MAE
    GRUBER_S_SURFACE = (
        "surface_project-gt_cc3-exclude6",
        0,
        "sad-pt-epoch=131-fine_mean_distance_val=1.50",
    )
    GRUBER_S_SURFACE_NEIGHSHOW = (
        "surface_project-gt_cc3-exclude6_neighborshow",
        0,
        "sad-pt-epoch=58-fine_mean_distance_val=1.60",
    )  # Needs inference code adaption
    #
    #
    GRUBER_N_SURFACE_NEIGHAUG = (
        "surface-neighbor-neighaug-project_gt_cc3-exclude6",
        0,
        "sad-pt-epoch=46-fine_mean_distance_val=3.70",
    )
    GRUBER_N_SURFACE_NEIGHAUG2 = (
        "surface-neighbor-neighaug-project_gt_cc3-exclude6",
        2,
        "sad-pt-epoch=70-fine_mean_distance_val=2.81",
    )
    GRUBER_N_SURFACE_NOVERT = (
        "surface-neighbor-no_vert-project_gt_cc3-exclude6",
        0,
        "sad-pt-epoch=109-fine_mean_distance_val=2.60",
    )
    GRUBER_N_SURFACE_NOVERT_NEIGHAUG = (
        "surface-neighbor-no_vert-project_gt_cc3-exclude6",
        1,
        "sad-pt-epoch=111-fine_mean_distance_val=3.52",
    )

    ######################
    # OLD
    ######################
    GRUBER_N_BOUNDARYBROKEN = (
        "surface-neighbor-project_gt_cc3-exclude6",
        1,
        "sad-pt-epoch=88-fine_mean_distance_val=2.34",
    )
