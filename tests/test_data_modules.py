"""Data module construction: defaults that differ between the two dataset layouts."""

from __future__ import annotations

import pytest

from verpex.modules.data_modules import DATA_MODULES, GruberDataModule, GruberNeighborDataModule

#: __init__ only stores configuration; nothing is read until setup(), so a
#: non-existent master_df is fine here.
BASE = {"master_df": "nonexistent.csv", "train_subjects": [], "val_subjects": [], "test_subjects": []}


def test_neighbour_module_defaults_to_no_flip():
    """It previously inherited 0.5 and flipped silently.

    The neighbour module printed a warning about flip_prob and then did nothing: the
    line that would have disabled flipping was commented out.
    """
    assert GruberNeighborDataModule(**BASE).flip_prob == 0.0


def test_single_vertebra_module_keeps_the_base_default():
    assert GruberDataModule(**BASE).flip_prob == 0.5


@pytest.mark.parametrize("flip_prob", [0.0, 0.25, 0.5])
def test_an_explicit_flip_prob_is_honoured(flip_prob):
    """A config's explicit setting must win over the module's default."""
    assert GruberNeighborDataModule(**BASE, flip_prob=flip_prob).flip_prob == flip_prob


@pytest.mark.parametrize("name", sorted(DATA_MODULES))
def test_every_registered_data_module_constructs(name):
    assert DATA_MODULES[name](**BASE) is not None
