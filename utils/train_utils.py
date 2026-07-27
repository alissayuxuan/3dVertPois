import json
import os

import pytorch_lightning as pl


def save_data_module_config(data_module, save_path):
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "data_module_params.json"), "w", encoding="utf-8") as f:
        json.dump(data_module.hparams, f, indent=4)


def create_callbacks(callbacks_config):
    callbacks_list = []
    for callback_config in callbacks_config:
        callback_type = callback_config["type"]
        if callback_type == "ModelCheckpoint":
            callbacks_list.append(pl.callbacks.ModelCheckpoint(**callback_config["params"]))
        elif callback_type == "EarlyStopping":
            callbacks_list.append(pl.callbacks.EarlyStopping(**callback_config["params"]))
    return callbacks_list
