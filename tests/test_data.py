"""Pure-Python tests for atlas.data (no torch / no CUDA)."""
import os

import numpy as np
import pytest

from atlas import data

FIXTURE = os.path.join(os.path.dirname(__file__), os.pardir, "examples", "tiny_cube.npy")


def test_rank_for_defaults():
    assert data.rank_for("jasperridge") == 10
    assert data.rank_for("paviau") == 12
    assert data.rank_for("anything_else") == 12


def test_init_key_path_and_dataset():
    # Known dataset names are unchanged...
    assert data.init_key("paviau") == "paviau"
    assert data.init_key("PaviaU") == "paviau"
    # ...and paths collapse to the lowercased basename stem (the bug that was fixed:
    # `atlas init` and `atlas train` must derive the SAME key for a path input).
    assert data.init_key("examples/tiny_cube.npy") == "tiny_cube"
    assert data.init_key("/abs/Path/To/My_Cube.MAT") == "my_cube"


def test_load_cube_fixture_shape_and_range():
    cube = data.load_cube(FIXTURE)
    assert cube.shape == (48, 48, 20)
    assert cube.dtype == np.float32
    assert cube.min() >= 0.0 and cube.max() <= 1.0 + 1e-6


def test_load_cube_unknown_raises():
    with pytest.raises((FileNotFoundError, ValueError)):
        data.load_cube("definitely_not_a_dataset")
