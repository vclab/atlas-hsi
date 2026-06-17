"""Regression test: `atlas init` writes NMF files under the key `atlas train` looks up.

This guards the path-input bug where init keyed files by the basename stem
(`tiny_cube`) but train looked them up by the raw `--input` string. Uses sklearn's
NMF (no torch / no CUDA).
"""
import argparse
import os

from atlas import data
from atlas.cli import cmd_init

FIXTURE = os.path.join(os.path.dirname(__file__), os.pardir, "examples", "tiny_cube.npy")


def test_init_writes_key_train_resolves(tmp_path):
    init_dir = str(tmp_path / "init")
    cmd_init(argparse.Namespace(
        input=FIXTURE, rank=4, data_dir=data.DEFAULT_DATA_DIR, init_dir=init_dir,
    ))

    key = data.init_key(FIXTURE)  # what `atlas train` derives for the same input
    assert key == "tiny_cube"

    # The train side must resolve both files via this key.
    E = data.load_endmember(key, 4, init_dir=init_dir)
    A0 = data.load_abundance(key, 4, init_dir=init_dir)
    assert E.shape == (4, 20)
    assert A0 is not None and A0.shape == (48 * 48, 4)
