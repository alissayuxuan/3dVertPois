CUDA_VISIBLE_DEVICES=3 python train.py --config /DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/src/hendrik/configsurfacecc3_trainexclude6/subreg-project_gt-no_freeze-surface-excel_exclude6-cc3.json


python3 eval.py --data_module_save_path <> --checkpoint_path <> --split <val/test> --save_path <> (--neighbor)
bsp:
python3 eval.py 
--checkpoint_path 
/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/src/ablation_study/dataloader/training/include_pois/subreg-project_gt-no_freeze-surface-excel_outliers_exclude-L2/version_1/checkpoints/sad-pt-epoch=54-fine_mean_distance_val=2.30.ckpt
--split val 
--save_path /DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/src/ablation_study/hendrik/val/subreg-project_gt-no_freeze-surface-excel_outliers_exclude-L2/


######
# Prepared data

python3 prepare_data.py --data_path /DATA/NAS/datasets_processed/CT_spine/dataset-poi-gruber --derivatives_name derivatives_seg derivatives_poi_new2g --save_path dataset/data_preprocessing/cutout-folder/cutouts-cc3-new --compute_surface_mask

