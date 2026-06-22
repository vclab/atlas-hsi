"""Round-trip tests for the ANS entropy codec + bitstream container (torch-free, CI-friendly)."""
import numpy as np
import pytest

from atlas.codec import _pack, _unpack
from atlas.quantize import (
    compress_matrix_flatten_categorical as enc,
)
from atlas.quantize import (
    decompress_matrix_flatten_categorical as dec,
)


def _roundtrip(mat):
    compressed, counts, unique = enc(mat.flatten().tolist())
    return dec(compressed, counts, unique, mat.size, mat.shape)


def test_roundtrip_multi_symbol():
    rng = np.random.default_rng(0)
    mat = rng.integers(0, 64, size=(200, 6)).astype(np.int32)   # 6-bit abundance-like
    assert np.array_equal(_roundtrip(mat), mat)


def test_roundtrip_signed():
    rng = np.random.default_rng(1)
    mat = rng.integers(-128, 128, size=(150, 3)).astype(np.int32)  # signed cholesky-like
    assert np.array_equal(_roundtrip(mat), mat)


def test_roundtrip_single_unique():
    mat = np.full((50, 3), 7, dtype=np.int32)   # encoder special-cases len(unique) == 1
    assert np.array_equal(_roundtrip(mat), mat)


def test_container_pack_unpack_lossless():
    meta = {"H": 4, "W": 5, "C": 3, "rank": 2, "coef": 1.5}
    arrays = {
        "xyz": np.arange(6, dtype=np.float16).reshape(3, 2),
        "uniq": np.array([-3, 0, 7], dtype=np.int16),     # signed symbol table
        "comp": np.array([1, 2, 3], dtype=np.uint32),
        "scale": np.array([0.5, 0.25], dtype=np.float32),
    }
    blob = _pack(meta, arrays)
    assert isinstance(blob, bytes) and blob[:4] == b"ATLS"  # real bytes, pickle-free
    m2, a2 = _unpack(blob)
    assert m2 == meta
    for k, v in arrays.items():
        assert a2[k].dtype == v.dtype and np.array_equal(a2[k], v)


def test_container_rejects_bad_magic():
    with pytest.raises(ValueError):
        _unpack(b"NOPE" + b"\x00" * 16)
