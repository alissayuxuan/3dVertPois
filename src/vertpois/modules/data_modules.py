"""Lightning data modules: reading master_df.csv and serving vertebra cutouts."""

import json
import os
import random
from functools import partial
from os import PathLike
from typing import TypeVar

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from monai.data.utils import worker_init_fn as _monai_worker_init_fn
from pqdm.processes import pqdm
from torch.utils.data import DataLoader

# from BIDS import BIDS_Global_info
from TPTBox import BIDS_Global_info

from vertpois.data.dataloading import (
    get_ct,
    get_files,
    get_gruber_poi,
    get_subreg,
    get_vertseg,
    process_container,
)
from vertpois.data.dataset import GruberDataset, GruberNeighborDataset, PoiDataset
from vertpois.data.transforms import create_transform
from vertpois.registry import build


def _seed_worker(worker_id):
    """Seed one dataloader worker deterministically.

    Seeds the numpy and random globals from torch's per-worker seed, and every
    MONAI ``Randomizable`` attached to the worker's dataset. RandAffine holds its
    own PRNG state, so without this the augmentation stream diverges between runs
    even when ``pl.seed_everything`` was called.
    """
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    if torch.utils.data.get_worker_info() is not None:
        _monai_worker_init_fn(worker_id)


PoiType = TypeVar("PoiType", bound=PoiDataset)


class POIDataModule(pl.LightningDataModule):
    """Base data module: reads a master_df.csv and serves per-vertebra cutouts.

    Subclasses supply the dataset class and the BIDS file-lookup functions for a
    particular dataset layout.
    """

    def __init__(
        self,
        dataset: str,
        master_df: PathLike,
        train_subjects: list,
        val_subjects: list,
        test_subjects: list,
        input_data_type: str = "subreg",
        input_shape: tuple = (128, 128, 96),
        zoom: tuple = (1, 1, 1),
        flip_prob: float = 0.5,
        transform_config: dict | None = None,
        include_com: bool = False,
        include_poi_list=None,
        include_vert_list=None,
        batch_size: int = 1,
        num_workers: int = 0,
        poi_file_ending: str = "poi.json",
        surface_erosion_iterations: int = 1,
        show_neighbors: bool = False,
        neighbor_drop_prob: float = 0.05,
    ):
        super().__init__()
        self.dataset = dataset
        self.master_df_path = master_df
        self.train_subjects = train_subjects
        self.val_subjects = val_subjects
        self.test_subjects = test_subjects
        self.input_data_type = input_data_type
        self.input_shape = input_shape
        self.zoom = zoom
        self.flip_prob = flip_prob
        self.include_com = include_com
        self.include_poi_list = include_poi_list
        self.include_vert_list = include_vert_list
        self.poi_file_ending = poi_file_ending
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_config = transform_config
        self.surface_erosion_iterations = surface_erosion_iterations
        self.show_neighbors = show_neighbors
        self.neighbor_drop_prob = neighbor_drop_prob
        self.save_hyperparameters()

    def prepare_data(
        self,
        bids_surgery_info: BIDS_Global_info,
        save_path: str,
        get_files: callable,
        rescale_zoom: tuple | None = None,
    ) -> None:
        """Build the per-vertebra cutouts this data module reads."""
        master = []
        partial_process_container = partial(
            process_container,
            save_path=save_path,
            rescale_zoom=rescale_zoom,
            get_files=get_files,
        )
        master = pqdm(
            bids_surgery_info.enumerate_subjects(),
            partial_process_container,
            n_jobs=8,
            argument_type="args",
            exception_behaviour="immediate",
        )
        master = [item for sublist in master for item in sublist]
        master_df = pd.DataFrame(master)
        master_df.to_csv(os.path.join(save_path, "master_df.csv"), index=False)

    def setup(self, stage=None) -> None:  # noqa: ARG002 - Lightning passes it; unused here
        """Load master_df.csv and construct the train, val and test datasets."""
        self.master_df = pd.read_csv(self.master_df_path)
        # empty column bad_poi_list
        # n_rows = len(self.master_df)
        # self.master_df["bad_poi_list"] = None  # Initialize the 'bad_poi_list' column with None
        # self.master_df["bad_poi_list"] = ["[]" for _ in range(n_rows)]  # Initialize the 'bad_poi_list' column with empty lists

        # Only keep rows where vertebra is in "include_vert_list"
        if self.include_vert_list is not None:
            # convert vertebra column to int
            self.master_df["vertebra"] = self.master_df["vertebra"].astype(int)
            self.master_df = self.master_df[self.master_df["vertebra"].isin(self.include_vert_list)]

        all_assigned_subjects = set(self.train_subjects) | set(self.val_subjects) | set(self.test_subjects)
        train_mask = self.master_df["subject"].isin(self.train_subjects) | ~self.master_df["subject"].isin(all_assigned_subjects)
        self.train_df = self.master_df[train_mask]
        self.val_df = self.master_df[self.master_df["subject"].isin(self.val_subjects)]
        self.test_df = self.master_df[self.master_df["subject"].isin(self.test_subjects)]

        if self.transform_config is not None:
            transform = [create_transform(self.transform_config)]

        if self.dataset == "Gruber":
            self.train_dataset = GruberDataset(
                self.train_df,
                input_data_type=self.input_data_type,
                input_shape=self.input_shape,
                zoom=self.zoom,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_vert_list=self.include_vert_list,
                transforms=transform,
                flip_prob=self.flip_prob,
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
                show_neighbors=self.show_neighbors,
                neighbor_drop_prob=self.neighbor_drop_prob,
            )
            self.val_dataset = GruberDataset(
                self.val_df,
                input_data_type=self.input_data_type,
                input_shape=self.input_shape,
                zoom=self.zoom,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_vert_list=self.include_vert_list,
                transforms=None,
                flip_prob=0.0,
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
                show_neighbors=self.show_neighbors,
                neighbor_drop_prob=self.neighbor_drop_prob,
            )
            self.test_dataset = GruberDataset(
                self.test_df,
                input_data_type=self.input_data_type,
                input_shape=self.input_shape,
                zoom=self.zoom,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_vert_list=self.include_vert_list,
                transforms=None,
                flip_prob=0.0,
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
                show_neighbors=self.show_neighbors,
                neighbor_drop_prob=self.neighbor_drop_prob,
            )

        elif self.dataset == "GruberNeighbor":  # NEU
            self.train_dataset = GruberNeighborDataset(
                self.train_df,
                input_data_type=self.input_data_type,
                input_shape=self.input_shape,
                zoom=self.zoom,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_vert_list=self.include_vert_list,
                transforms=transform,
                flip_prob=self.flip_prob,  # Explizit deaktiviert
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
                neighbor_drop_prob=self.neighbor_drop_prob,
            )
            self.val_dataset = GruberNeighborDataset(
                self.val_df,
                input_data_type=self.input_data_type,
                input_shape=self.input_shape,
                zoom=self.zoom,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_vert_list=self.include_vert_list,
                transforms=None,
                flip_prob=0.0,
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
                neighbor_drop_prob=self.neighbor_drop_prob,
            )
            self.test_dataset = GruberNeighborDataset(
                self.test_df,
                input_data_type=self.input_data_type,
                input_shape=self.input_shape,
                zoom=self.zoom,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_vert_list=self.include_vert_list,
                transforms=None,
                flip_prob=0.0,
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
                neighbor_drop_prob=self.neighbor_drop_prob,
            )
        print("Setup complete. Dataset sizes:")
        print(f"  Train: {len(self.train_dataset)}")
        print(f"  Validation: {len(self.val_dataset)}")
        print(f"  Test: {len(self.test_dataset)}")

    def _dataloader_generator(self):
        return torch.Generator().manual_seed(42)

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader, with augmentation enabled."""
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            worker_init_fn=_seed_worker,
            generator=self._dataloader_generator(),
            # collate_fn=custom_collate_fn
        )

    def train_noaug_dataloader(self) -> DataLoader:
        """Return the training dataloader with augmentation disabled."""
        train_2_dataset = self.train_dataset
        train_2_dataset.flip_prob = 0.0
        train_2_dataset.transforms = None
        return DataLoader(
            dataset=train_2_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            worker_init_fn=_seed_worker,
            generator=self._dataloader_generator(),
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            worker_init_fn=_seed_worker,
            generator=self._dataloader_generator(),
            # collate_fn=custom_collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            worker_init_fn=_seed_worker,
            generator=self._dataloader_generator(),
        )


class GruberDataModule(POIDataModule):
    """Data module for the single-vertebra cutout dataset."""

    def __init__(
        self,
        master_df: PathLike,
        train_subjects: list,
        val_subjects: list,
        test_subjects: list,
        input_data_type: str = "subreg",
        input_shape: tuple = (128, 128, 96),
        zoom: tuple = (1, 1, 1),
        flip_prob: float = 0.5,
        transform_config: dict | None = None,
        include_com: bool = False,
        include_poi_list=None,
        include_vert_list=None,
        batch_size: int = 1,
        num_workers: int = 0,
        poi_file_ending: str = "poi.json",
        surface_erosion_iterations: int = 1,
        show_neighbors: bool = False,
        neighbor_drop_prob: float = 0.05,
    ):
        super().__init__(
            dataset="Gruber",
            master_df=master_df,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
            input_data_type=input_data_type,
            input_shape=input_shape,
            zoom=zoom,
            flip_prob=flip_prob,
            transform_config=transform_config,
            include_com=include_com,
            include_poi_list=include_poi_list,
            include_vert_list=include_vert_list,
            batch_size=batch_size,
            num_workers=num_workers,
            poi_file_ending=poi_file_ending,
            surface_erosion_iterations=surface_erosion_iterations,
            show_neighbors=show_neighbors,
            neighbor_drop_prob=neighbor_drop_prob,
        )

    def prepare_data(
        self,
        bids_surgery_info: BIDS_Global_info,
        save_path: str,
        rescale_zoom: tuple | None = None,
    ) -> None:
        """Build the per-vertebra cutouts this data module reads."""
        gruber_get_files = partial(
            get_files,
            get_poi=get_gruber_poi,
            get_ct=get_ct,
            get_subreg=get_subreg,
            get_vertseg=get_vertseg,
        )
        super().prepare_data(bids_surgery_info, save_path, gruber_get_files, rescale_zoom)


class GruberNeighborDataModule(POIDataModule):
    """Data module that also serves each vertebra's neighbours."""

    def __init__(self, **kwargs):
        # deactivate horizontal flip
        if kwargs.get("flip_prob", 0) > 0:
            print(f"WARNING: flip_prob set to {kwargs.get('flip_prob', 0)} for neighbor dataset")
            # kwargs["flip_prob"] = 0.0

        # kwargs['input_shape'] = (120, 121, 149)
        super().__init__(dataset="GruberNeighbor", **kwargs)

    def prepare_data(self, bids_surgery_info, save_path, rescale_zoom=None) -> None:
        """Build the cutouts this module needs, if they are not on disk yet."""
        gruber_get_files = partial(
            get_files,
            get_poi=get_gruber_poi,
            get_ct=get_ct,
            get_subreg=get_subreg,
            get_vertseg=get_vertseg,
        )
        super().prepare_data(bids_surgery_info, save_path, gruber_get_files, rescale_zoom)


#: Config ``"type"`` string -> data module.
DATA_MODULES = {
    "GruberDataModule": GruberDataModule,
    "GruberNeighborDataModule": GruberNeighborDataModule,
}


def create_data_module(config) -> POIDataModule:
    """Create the data module a config describes.

    Args:
        config: A ``{"type": ..., "params": {...}}`` mapping.

    Returns:
        The constructed data module.

    Raises:
        UnknownTypeError: If the config names a data module that is not registered.
    """
    return build(DATA_MODULES, "data module", config)


if __name__ == "__main__":
    config_path = "neighbors-test/datamodule_config.json"
    with open(config_path) as f:
        config = json.load(f)

    data_module = create_data_module(config)
    data_module.setup()
