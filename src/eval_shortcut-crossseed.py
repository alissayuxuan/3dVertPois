import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Add project root to Python path

from subprocess import run
from pathlib import Path
import os
from TypeSaveArgParse import Class_to_ArgParse
from dataclasses import dataclass, field
from utils.filepaths import search_path


@dataclass
class EvalConfig(Class_to_ArgParse):
    model_dir_names: list[str] = field(
        default_factory=lambda: [
            "ct_cc3",
            # "/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/src/hendrik/trainings/miccai_trainings/ct_cc3/version_1/checkpoints/sad-pt-epoch=60-fine_mean_distance_val=1.78.ckpt",
            "subreg_cc3",
            "surface_cc3",
            "ctraw_cc3",
            #
            # "surface_cc3-bs32",
            "surface_cc3-L1",
            "surface_cc3-L2",
            "surface_cc3-noVert",
            "surface_cc3-noVertPoi",
            "surface_cc3-noPoi",
            "surface_cc3-noPoiFeature",
            # "surface_cc3-projectpred",
            #
            "surface_neighbor_cc3",
            "ct_neighbor_cc3",
            # "ct_cc3-traindropout",
        ]
    )
    split: str = "test"
    save_path: str = "/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/src/miccai_study/"
    gpu: int = 0


if __name__ == "__main__":

    opt = EvalConfig().get_opt()

    training_ROOT = "/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/src/hendrik/trainings/miccai_trainings/"

    if isinstance(opt.model_dir_names, str):
        opt.model_dir_names = (opt.model_dir_names,)

    for md in opt.model_dir_names:

        for v in range(5):
            model_v_dir = f"{training_ROOT}{md}/version_{v}/checkpoints/"
            if not os.path.exists(model_v_dir):
                continue

            # find best checkpoint in version
            ckpts = search_path(model_v_dir, "*epoch=*-fine_mean_distance_val=*.ckpt")
            min_value = 50
            best_ckpt = None
            for ckpt in ckpts:
                try:
                    value_str = str(ckpt).split("fine_mean_distance_val=")[-1].split(".ckpt")[0]
                    value = float(value_str)
                    if value < min_value:
                        min_value = value
                        best_ckpt = ckpt
                except Exception as e:
                    print(f"Error parsing checkpoint {ckpt}: {e}")

            if best_ckpt is None:
                print(f"No valid checkpoint found in {model_v_dir}")
                continue

            c = f"{model_v_dir}{best_ckpt.name}"

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
                f"{opt.save_path}/{split}/gtnoproj_{md}/{save_path}-{v}-{min_value:.2f}/",
                # "--save_gt_proj",
                "--project",
                # "--save_files",
            ]
            print()
            print(cmd)
            run(cmd, shell=False)
