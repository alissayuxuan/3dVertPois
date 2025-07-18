# added by me
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Points to /home/student/alissa/3dVertPois


import argparse
import json
import os

import pytorch_lightning as pl
import torch

from modules.PoiDataModules import create_data_module
from modules.PoiModule import PoiPredictionModule


def save_data_module_config(data_module, save_path):
    """Save the hyperparameters of the DataModule to a JSON file for documentation and
    easy reproducibility."""
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)
    # Save the hyperparameters to a JSON file
    with open(
        os.path.join(save_path, "data_module_params.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(data_module.hparams, file, indent=4)


def create_callbacks(callbacks_config):
    callbacks_list = []
    for callback_config in callbacks_config:
        callback_type = callback_config["type"]
        if callback_type == "ModelCheckpoint":
            callbacks_list.append(
                pl.callbacks.ModelCheckpoint(**callback_config["params"])
            )
        elif callback_type == "EarlyStopping":
            callbacks_list.append(
                pl.callbacks.EarlyStopping(**callback_config["params"])
            )
        # Add other callbacks as needed
    return callbacks_list


def run_experiment(experiment_config):

    # Set the matmul precision to 'medium' for better performance
    torch.set_float32_matmul_precision("medium")
    poi_module_config = experiment_config["module_config"]
    data_module_config = experiment_config["data_module_config"]

    data_module = create_data_module(data_module_config)
    data_module.setup()
    poi_module = PoiPredictionModule(**poi_module_config["params"])

    """
    # Create callbacks from configuration
    callbacks = create_callbacks(experiment_config.get("callbacks_config", []))

    # Trainer configuration
    trainer_config = experiment_config.get("trainer_config", {})
    trainer_config.setdefault("callbacks", callbacks)
    trainer_config.setdefault(
        "logger",
        pl.loggers.TensorBoardLogger(
            experiment_config["path"], name=experiment_config["name"]
        ),
    )

    trainer = pl.Trainer(
        **trainer_config, 
        #strategy="ddp_find_unused_parameters_true" 
    )

    # Save DataModule config
    data_module_config_path = trainer.logger.log_dir
    save_data_module_config(data_module, data_module_config_path)
    """

    # === 1. Create logger early and reuse ===
    logger = pl.loggers.TensorBoardLogger(
        save_dir=experiment_config["path"],
        name=experiment_config["name"]
    )

    # === 2. Save data_module config BEFORE trainer is created ===
    save_data_module_config(data_module, logger.log_dir)

    # === 3. Create callbacks and trainer ===
    callbacks = create_callbacks(experiment_config.get("callbacks_config", []))

    trainer_config = experiment_config.get("trainer_config", {})
    trainer_config.setdefault("callbacks", callbacks)
    trainer_config["logger"] = logger  # override any default

    trainer = pl.Trainer(**trainer_config)

    """
    trainer.fit(
        poi_module,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )
    """

    print("\n=== Starting Training ===")
    trainer.fit(
        poi_module,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )
    
    print("\n=== Final Debug Info ===")
    print(f"Epochs completed: {trainer.current_epoch}")
    print(f"Max epochs allowed: {trainer.max_epochs}")    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Experiment config file")
    parser.add_argument(
        "--config-dir", type=str, help="Directory containing experiment config files"
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            experiment_config = json.load(f)
            run_experiment(experiment_config)

    if args.config_dir:
        for config_file in os.listdir(args.config_dir):
            with open(os.path.join(args.config_dir, config_file), "r") as f:
                experiment_config = json.load(f)
                run_experiment(experiment_config)
