"""Data module construction: defaults that differ between the two dataset layouts."""

from __future__ import annotations

import pytest

from verpex.modules.data_modules import (
    DATA_MODULES,
    NEIGHBOR_DATASETS,
    SINGLE_VERTEBRA_DATASETS,
    SPINE_DATASET,
    SPINE_NEIGHBOR_DATASET,
    SpineDataModule,
    SpineNeighborDataModule,
)

#: __init__ only stores configuration; nothing is read until setup(), so a
#: non-existent master_df is fine here.
BASE = {"master_df": "nonexistent.csv", "train_subjects": [], "val_subjects": [], "test_subjects": []}


def test_neighbour_module_defaults_to_no_flip():
    """It previously inherited 0.5 and flipped silently.

    The neighbour module printed a warning about flip_prob and then did nothing: the
    line that would have disabled flipping was commented out.
    """
    assert SpineNeighborDataModule(**BASE).flip_prob == 0.0


def test_single_vertebra_module_keeps_the_base_default():
    assert SpineDataModule(**BASE).flip_prob == 0.5


@pytest.mark.parametrize("flip_prob", [0.0, 0.25, 0.5])
def test_an_explicit_flip_prob_is_honoured(flip_prob):
    """A config's explicit setting must win over the module's default."""
    assert SpineNeighborDataModule(**BASE, flip_prob=flip_prob).flip_prob == flip_prob


@pytest.mark.parametrize("name", sorted(DATA_MODULES))
def test_every_registered_data_module_constructs(name):
    assert DATA_MODULES[name](**BASE) is not None


#: The cohort-named originals. Existing experiment configs use these, so they must
#: keep resolving to the renamed classes.
LEGACY_TYPE_ALIASES = {
    "GruberDataModule": SpineDataModule,
    "GruberNeighborDataModule": SpineNeighborDataModule,
}


@pytest.mark.parametrize(("legacy_name", "expected"), sorted(LEGACY_TYPE_ALIASES.items()))
def test_legacy_config_type_names_still_resolve(legacy_name, expected):
    assert DATA_MODULES[legacy_name] is expected


@pytest.mark.parametrize("legacy_value", ["Gruber", "GruberNeighbor"])
def test_legacy_dataset_values_are_recognised(legacy_value):
    """`data_module_params.json` from an earlier run stores the cohort-named value.

    Those files are read back at inference time, so the old value has to keep
    selecting the right dataset layout.
    """
    assert legacy_value in SINGLE_VERTEBRA_DATASETS + NEIGHBOR_DATASETS


def test_the_neighbour_flag_is_derived_from_either_spelling():
    """infer.py decides whether to use neighbours from the saved dataset value."""
    assert SPINE_NEIGHBOR_DATASET in NEIGHBOR_DATASETS
    assert "GruberNeighbor" in NEIGHBOR_DATASETS
    assert SPINE_DATASET not in NEIGHBOR_DATASETS
