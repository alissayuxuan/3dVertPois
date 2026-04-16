from subprocess import run
from pathlib import Path
import os
from TypeSaveArgParse import Class_to_ArgParse
from dataclasses import dataclass, field

# "subreg_project-gt_cc3-exclude6/version_0/checkpoints/sad-pt-epoch=84-fine_mean_distance_val=1.79.ckpt",
# "vertseg_project-gt_cc3-exclude6/version_0/checkpoints/sad-pt-epoch=97-fine_mean_distance_val=2.24.ckpt",
# "ctraw_project-gt_cc3-exclude6/version_2/checkpoints/sad-pt-epoch=93-fine_mean_distance_val=3.02.ckpt",
###
###
# "ct_neighbor-project-gt_cc3-exclude6/version_0/checkpoints/sad-pt-epoch=65-fine_mean_distance_val=3.03.ckpt",
# "surface_neighbor-project-gt_cc3-exclude6/version_6/checkpoints/sad-pt-epoch=70-fine_mean_distance_val=2.97.ckpt",
# "subreg_neighbor-project-gt_cc3-exclude6/version_0/checkpoints/sad-pt-epoch=70-fine_mean_distance_val=2.98.ckpt",
# "ctraw_neighbor-project-gt_cc3-exclude6/version_0/checkpoints/sad-pt-epoch=41-fine_mean_distance_val=4.61.ckpt",
##
##
# "ct_project2-gt_cc3-exclude6/version_0/checkpoints/sad-pt-epoch=89-fine_mean_distance_val=2.31.ckpt",
# "ct_project2-gt_cc3-exclude6-L1/version_0/checkpoints/sad-pt-epoch=77-fine_mean_distance_val=2.39.ckpt",
# "ct_project2-gt_cc3-exclude6-L1-Wing/version_0/checkpoints/sad-pt-epoch=96-fine_mean_distance_val=2.44.ckpt",
# "surface_project2-gt_cc3-exclude6/version_0/checkpoints/sad-pt-epoch=102-fine_mean_distance_val=2.31.ckpt",
# "surface_project2-gt_cc3-exclude6-L1/version_0/checkpoints/sad-pt-epoch=47-fine_mean_distance_val=2.57.ckpt",
##
##
# "ct_project-gt_cc3-exclude6-L1/version_0/checkpoints/sad-pt-epoch=68-fine_mean_distance_val=2.45.ckpt",
# "ct_project-gt_cc3-exclude6-L1-L2/version_0/checkpoints/sad-pt-epoch=73-fine_mean_distance_val=3.27.ckpt",
# "ct_project-gt_cc3-exclude6-L2/version_0/checkpoints/sad-pt-epoch=76-fine_mean_distance_val=2.97.ckpt",
# "ct_project-gt_cc3-exclude6-Wing-Surface/version_1/checkpoints/sad-pt-epoch=87-fine_mean_distance_val=2.63.ckpt",
# "surface_project-gt_cc3-exclude6-L1/version_0/checkpoints/sad-pt-epoch=70-fine_mean_distance_val=2.48.ckpt",

#
# "surface_project-gt_cc3-exclude6/version_0/checkpoints/sad-pt-epoch=115-fine_mean_distance_val=1.69.ckpt",
#            "ct_project-gt_cc3-exclude6/version_0/checkpoints/sad-pt-epoch=118-fine_mean_distance_val=1.67.ckpt",
# "ct_project-gt_cc3-exclude6-cL2-rL1/version_0/checkpoints/sad-pt-epoch=28-fine_mean_distance_val=3.08.ckpt",
# "ct_project-gt_cc3-exclude6-L1-L2/version_0/checkpoints/sad-pt-epoch=52-fine_mean_distance_val=3.07.ckpt",
# "ct_project-gt_cc3-exclude6-L2/version_0/checkpoints/sad-pt-epoch=25-fine_mean_distance_val=3.79.ckpt",
# "ct_project-gt_noVert-cc3-exclude6/version_0/checkpoints/sad-pt-epoch=82-fine_mean_distance_val=2.23.ckpt",
#            "ct_projectcon-gt_cc3-exclude6/version_1/checkpoints/sad-pt-epoch=76-fine_mean_distance_val=2.76.ckpt",
# "ct_projectcon-gt-all_cc3-exclude6/version_0/checkpoints/sad-pt-epoch=107-fine_mean_distance_val=1.80.ckpt",
# "ct_projectcon-gt-coarse_cc3-exclude6/version_0/checkpoints/sad-pt-epoch=100-fine_mean_distance_val=2.26.ckpt",
# "ct_projectcon-gt-fine-coarse_cc3-exclude6/version_0/checkpoints/sad-pt-epoch=67-fine_mean_distance_val=2.13.ckpt",
# "subreg_project-gt_cc3-exclude6/version_0/checkpoints/sad-pt-epoch=69-fine_mean_distance_val=2.57.ckpt",
# "surface_project-gt_cc3-exclude6/version_0/checkpoints/sad-pt-epoch=102-fine_mean_distance_val=2.32.ckpt",
# "surface_project-gt_cc3-exclude6-L1/version_0/checkpoints/sad-pt-epoch=37-fine_mean_distance_val=2.72.ckpt",


@dataclass
class EvalConfig(Class_to_ArgParse):
    checkpoint_path: list[str] = field(
        default_factory=lambda: [
            "ct_cc3/version_0/checkpoints/sad-pt-epoch=75-fine_mean_distance_val=1.62.ckpt",
            # "/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/src/hendrik/trainings/miccai_trainings/ct_cc3/version_1/checkpoints/sad-pt-epoch=60-fine_mean_distance_val=1.78.ckpt",
            "subreg_cc3/version_0/checkpoints/sad-pt-epoch=103-fine_mean_distance_val=1.61.ckpt",
            "surface_cc3/version_0/checkpoints/sad-pt-epoch=163-fine_mean_distance_val=1.48.ckpt",
            "ctraw_cc3/version_1/checkpoints/sad-pt-epoch=149-fine_mean_distance_val=1.95.ckpt",
            "ct_cc3-traindropout/version_0/checkpoints/sad-pt-epoch=75-fine_mean_distance_val=1.94.ckpt",
        ]
    )
    split: str = "test"
    save_path: str = "/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/src/miccai_study/"
    gpu: int = 5


if __name__ == "__main__":

    opt = EvalConfig().get_opt()

    training_ROOT = "/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/src/hendrik/trainings/miccai_trainings/"

    if isinstance(opt.checkpoint_path, str):
        opt.checkpoint_path = (opt.checkpoint_path,)

    for c in opt.checkpoint_path:

        if training_ROOT not in c:
            model_dir = f"{training_ROOT}{c}"
        else:
            model_dir = c

        gpu = opt.gpu
        split = opt.split
        #
        save_path = Path(model_dir).parent.parent.parent.name
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        #
        cmd = [
            "python3",
            "/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/src/eval.py",
            "--checkpoint_path",
            model_dir,
            "--split",
            f"{split}",
            "--save_path",
            f"{opt.save_path}/{split}/gtnoproj_{save_path}/",
            # "--save_gt_proj",
            "--project",
            # "--save_files",
        ]
        print()
        print(cmd)
        run(cmd, shell=False)
