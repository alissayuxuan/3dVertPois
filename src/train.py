import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Points to project root

import argparse
import json
import os

import pytorch_lightning as pl
import torch

from modules.PoiDataModules import create_data_module
from modules.PoiModule import PoiPredictionModule
from utils.train_utils import create_callbacks, save_data_module_config


def run_experiment(experiment_config):
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")
    # warn_only=True would let 3D CUDA convs run non-deterministically.
    # Setting False raises on the first non-deterministic op — used to identify
    # the exact source of run-to-run variance.
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(mode=True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_module = create_data_module(experiment_config["data_module_config"])
    data_module.setup()

    poi_module = PoiPredictionModule(**experiment_config["module_config"]["params"])

    logger = pl.loggers.TensorBoardLogger(save_dir=experiment_config["path"], name=experiment_config["name"])
    save_data_module_config(data_module, logger.log_dir)

    callbacks = create_callbacks(experiment_config.get("callbacks_config", []))
    trainer_config = experiment_config.get("trainer_config", {})
    trainer_config.setdefault("callbacks", callbacks)
    trainer_config.setdefault("deterministic", "warn")
    trainer_config["logger"] = logger

    trainer = pl.Trainer(**trainer_config)

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
    parser.add_argument("--config-dir", type=str, help="Directory containing experiment config files")
    args = parser.parse_args()

    pl.seed_everything(42)

    if args.config:
        with open(args.config, "r") as f:
            run_experiment(json.load(f))

    if args.config_dir:
        for config_file in os.listdir(args.config_dir):
            with open(os.path.join(args.config_dir, config_file), "r") as f:
                run_experiment(json.load(f))
