from TPTBox.core.poi import POI
import os
import re

def create_global_pois(source_path, save_path):

    os.makedirs(save_path, exist_ok=True)

    for filename in os.listdir(source_path):
        if filename.endswith("_pred.json"):
            source_path_pred = os.path.join(source_path, filename)

            new_filename = filename.replace("_pred.json", "_pred_global.json")
            target_path = os.path.join(save_path, new_filename)

            POI.load(source_path_pred).to_global().save_mrk(target_path)

        if filename.endswith("_gt.json"):
            source_path_pred = os.path.join(source_path, filename)

            new_filename = filename.replace("_gt.json", "_gt_global.json")
            target_path = os.path.join(save_path, new_filename)

            POI.load(source_path_pred).to_global().save_mrk(target_path)






if __name__ == "__main__":
    source_path = "experiments/experiment_evaluation/gruber/surface/excel_excluded_pois/no_freeze/test/version_2_epoch_55/prediction_files"
    save_path =   "experiments/experiment_evaluation/gruber/surface/excel_excluded_pois/no_freeze/test/version_2_epoch_55/global_prediction_files"

    create_global_pois(source_path, save_path)
