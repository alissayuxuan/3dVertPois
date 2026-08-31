import argparse
import json
import os

import pytorch_lightning as pl
import torch
from sklearn.model_selection import KFold

from vertpois.modules.data_modules import create_data_module
from vertpois.modules.poi_module import PoiPredictionModule
from vertpois.training_utils import create_callbacks, save_data_module_config


def run_cv(n_folds, experiment_config, save_predictions=False, poi_file_ending=None):
    # If the predictions are saved the file ending must be set
    if save_predictions and not poi_file_ending:
        raise ValueError("If predictions are saved the poi file ending must be set")

    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_subjects = experiment_config["data_module_config"]["params"]["train_subjects"]
    val_subjects = experiment_config["data_module_config"]["params"]["val_subjects"]

    # Add val subjects to train subjects and create random folds
    train_subjects += val_subjects
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_subjects)):
        train_subjects_fold = [train_subjects[i] for i in train_idx]
        val_subjects_fold = [train_subjects[i] for i in val_idx]

        data_module_config = experiment_config["data_module_config"]
        data_module_config["params"]["train_subjects"] = train_subjects_fold
        data_module_config["params"]["val_subjects"] = val_subjects_fold

        data_module = create_data_module(data_module_config)
        data_module.setup()
        poi_module = PoiPredictionModule(**experiment_config["module_config"]["params"])

        # Create callbacks from configuration
        callbacks = create_callbacks(experiment_config.get("callbacks_config", []))

        # Trainer configuration
        trainer_config = experiment_config.get("trainer_config", {})
        trainer_config["callbacks"] = callbacks
        trainer_config.setdefault("deterministic", "warn")

        # Add fold to path
        path = experiment_config["path"] + f"/fold_{fold}"
        trainer_config["logger"] = pl.loggers.TensorBoardLogger(
            path, name=experiment_config["name"]
        )

        trainer = pl.Trainer(**trainer_config)

        # Save DataModule config
        data_module_config_path = trainer.logger.log_dir
        save_data_module_config(data_module, data_module_config_path)

        trainer.fit(
            poi_module,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader(),
        )

        if save_predictions:
            # Get path of best model
            best_model_path = trainer.checkpoint_callback.best_model_path

            # `create_self_training_pois` was removed from eval.py in commit 4632148
            # ("added zoom consideration to model and eval.py") without updating this
            # call site, which left train_cv.py raising ImportError on every run. The
            # function predated that zoom-correctness pass, so it is not restored here
            # rather than reintroduced unverified. See CHANGES.md.
            raise NotImplementedError(
                "--save-predictions (self-training pseudo-label export) is not available: "
                "its implementation was removed during the zoom-handling fix and has not "
                "been re-derived. Run the cross-validation without this flag, or see "
                "CHANGES.md for what restoring it requires."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_folds", type=int, help="Number of folds for cross-validation"
    )
    parser.add_argument(
        "--save-predictions", action="store_true", help="Save predictions for each fold"
    )
    parser.add_argument(
        "--poi-file-ending",
        type=str,
        help="Ending of the poi file to save predictions to",
    )
    parser.add_argument("--config", type=str, help="Experiment config file")
    parser.add_argument(
        "--config-dir", type=str, help="Directory containing experiment config files"
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            experiment_config = json.load(f)
            run_cv(
                args.n_folds,
                experiment_config,
                save_predictions=args.save_predictions,
                poi_file_ending=args.poi_file_ending,
            )

    if args.config_dir:
        for config_file in os.listdir(args.config_dir):
            with open(os.path.join(args.config_dir, config_file), "r") as f:
                experiment_config = json.load(f)
                run_cv(
                    args.n_folds,
                    experiment_config,
                    save_predictions=args.save_predictions,
                    poi_file_ending=args.poi_file_ending,
                )
