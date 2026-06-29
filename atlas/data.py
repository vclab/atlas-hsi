"""Dataset loading for ATLAS.

Provides a single ``load_cube`` entry point that accepts either a known benchmark
dataset name (with its paper-matching preprocessing) or a path to an arbitrary
``.mat`` / ``.npy`` / ENVI hyperspectral cube. All loaders return a ``(H, W, C)``
float array normalized per band into roughly ``[0, 1]``.
"""
import os
import re

import numpy as np

# CWD-relative defaults so the installed package writes/reads in the user's project,
# not inside site-packages.
DEFAULT_DATA_DIR = os.path.join("data", "raw")
DEFAULT_INIT_DIR = os.path.join("data", "init")

# Endmember rank per benchmark dataset (default 12).
_RANKS = {"jasperridge": 10}

_KNOWN = {"urban", "salinas", "jasperridge", "paviau", "paviac", "whu_honghu"}

# ENVI data-type code -> numpy base dtype (byte order applied separately).
_ENVI_DTYPE = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8",
               12: "u2", 13: "u4", 14: "i8", 15: "u8"}


def rank_for(name):
    """Default NMF rank for a known dataset name (12 unless overridden)."""
    return _RANKS.get(str(name).lower(), 12)


def init_key(dataset_or_path):
    """Canonical key for NMF init filenames, shared by `atlas init` and `atlas train`.

    For a known dataset name this is the name itself (e.g. ``paviau``); for a path it
    is the lowercased file stem (e.g. ``examples/tiny_cube.npy`` → ``tiny_cube``), so
    both commands agree on where the endmember/abundance ``.npy`` files live.
    """
    return os.path.splitext(os.path.basename(str(dataset_or_path)))[0].lower()


def _normalize_bands(cube):
    """Normalize each spectral band by its max (in place-safe copy)."""
    cube = cube.astype(np.float32, copy=True)
    for i in range(cube.shape[2]):
        m = np.max(cube[:, :, i])
        if m > 0:
            cube[:, :, i] /= m
    return cube


def _load_envi(data_path):
    """Read an ENVI cube (raw ``data_path`` + sidecar ``.hdr``) as ``(H, W, C)`` float.

    Supports bsq/bil/bip interleave and the common data types. In ENVI headers
    ``samples`` are columns (W), ``lines`` are rows (H), and ``bands`` are channels (C).
    """
    hdr_path = os.path.splitext(data_path)[0] + ".hdr"
    with open(hdr_path) as fh:
        hdr = fh.read()

    def _int(key, default=None):
        m = re.search(rf"^\s*{key}\s*=\s*(\d+)", hdr, re.IGNORECASE | re.MULTILINE)
        return int(m.group(1)) if m else default

    w, h, c, code = _int("samples"), _int("lines"), _int("bands"), _int("data type")
    if None in (w, h, c, code):
        raise ValueError(f"incomplete ENVI header: {hdr_path}")
    offset = _int("header offset", 0)
    endian = ">" if _int("byte order", 0) == 1 else "<"
    m = re.search(r"^\s*interleave\s*=\s*(\w+)", hdr, re.IGNORECASE | re.MULTILINE)
    interleave = (m.group(1).lower() if m else "bsq")

    raw = np.fromfile(data_path, dtype=np.dtype(endian + _ENVI_DTYPE[code]), offset=offset)
    if interleave == "bsq":      # stored (bands, lines, samples)
        return raw.reshape(c, h, w).transpose(1, 2, 0).astype(float)
    if interleave == "bil":      # stored (lines, bands, samples)
        return raw.reshape(h, c, w).transpose(0, 2, 1).astype(float)
    if interleave == "bip":      # stored (lines, samples, bands)
        return raw.reshape(h, w, c).astype(float)
    raise ValueError(f"unsupported ENVI interleave: {interleave}")


def _load_known(name, data_dir):
    """Load a benchmark dataset with its paper-matching preprocessing → (H, W, C)."""
    import scipy.io
    name = name.lower()
    if name == "urban":
        I = scipy.io.loadmat(os.path.join(data_dir, "Urban_R162.mat"))["Y"].astype(float)
        for i in range(162):
            I[i, :] /= np.max(I[i, :])
        return I.reshape(162, 307, 307).transpose(2, 1, 0)
    if name == "salinas":
        I = scipy.io.loadmat(os.path.join(data_dir, "Salinas.mat"))["salinas"].astype(float)
        I = np.clip(I, 0, None)
        # Crop to upper-right 180x180 (LoR-SGS paper evaluation region).
        I = I[0:180, 37:217, :]
        # Remove 20 water-absorption bands (1-indexed: 108-112, 154-167, 224).
        water = set(range(107, 112)) | set(range(153, 167)) | {223}
        keep = [i for i in range(224) if i not in water]
        return _normalize_bands(I[:, :, keep])  # (180, 180, 204)
    if name == "jasperridge":
        I = scipy.io.loadmat(os.path.join(data_dir, "jasperRidge2_R198.mat"))["Y"].astype(float)
        for i in range(198):
            I[i, :] /= np.max(I[i, :])
        return I.reshape(198, 100, 100).transpose(2, 1, 0)
    if name == "paviau":
        I = scipy.io.loadmat(os.path.join(data_dir, "PaviaU.mat"))["paviaU"].astype(float)
        I = _normalize_bands(I)
        return I[-340:, :, :]
    if name == "paviac":
        # Pavia Centre (GIC): the distributed Pavia.mat is the corrected 1096x715x102 region
        # (variable "pavia"), same ROSIS sensor and spectral family as PaviaU. Full scene,
        # no crop. ~6.8x the PaviaU 340x340 crop in spatial pixels.
        I = scipy.io.loadmat(os.path.join(data_dir, "Pavia.mat"))["pavia"].astype(float)
        return _normalize_bands(I)
    if name == "whu_honghu":
        # WHU-Hi-HongHu (RSIDEA, UAV Nano-Hyperspec): ENVI cube 940x475x270, float32,
        # 400-1000 nm. ~3.9x the PaviaU 340x340 crop in pixels and 2.6x its bands.
        I = _load_envi(os.path.join(data_dir, "WHU-Hi-HongHu.bsq"))
        return _normalize_bands(np.clip(I, 0, None))
    raise ValueError(f"Unknown dataset: {name}")


def _load_path(path):
    """Load an arbitrary .mat/.npy/ENVI cube → (H, W, C), normalized per band."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        cube = np.load(path).astype(float)
    elif ext == ".mat":
        import scipy.io
        mat = scipy.io.loadmat(path)
        arrays = [v for k, v in mat.items() if not k.startswith("__") and hasattr(v, "ndim")]
        cubes = [a for a in arrays if a.ndim == 3]
        if not cubes:
            raise ValueError(f"No 3D array found in {path}")
        cube = max(cubes, key=lambda a: a.size).astype(float)  # pick the largest cube
    elif ext in (".bsq", ".bil", ".bip", ".img", ".dat"):
        cube = _load_envi(path)  # raw ENVI data file with a sidecar .hdr
    else:
        raise ValueError(f"Unsupported cube extension: {ext} (use .mat, .npy, or ENVI)")
    cube = np.clip(cube, 0, None)
    return _normalize_bands(cube)


def load_cube(dataset_or_path, data_dir=DEFAULT_DATA_DIR):
    """Load a hyperspectral cube as ``(H, W, C)`` float32 normalized per band.

    ``dataset_or_path`` may be a known dataset name (urban|salinas|jasperridge|paviau)
    or a path to a ``.mat`` / ``.npy`` file.
    """
    if str(dataset_or_path).lower() in _KNOWN:
        return _load_known(dataset_or_path, data_dir)
    if os.path.exists(dataset_or_path):
        return _load_path(dataset_or_path)
    raise FileNotFoundError(
        f"'{dataset_or_path}' is neither a known dataset {sorted(_KNOWN)} nor an existing file"
    )


def load_endmember(name, rank, init_dir=DEFAULT_INIT_DIR):
    """Load the NMF endmember matrix (rank, channels) for a dataset."""
    path = os.path.join(init_dir, f"{name.lower()}_endmember_rank_{rank}_NMF.npy")
    return np.load(path).astype(np.float32)


def load_abundance(name, rank, init_dir=DEFAULT_INIT_DIR):
    """Load the NMF abundance map for entropy-guided init, or None if absent."""
    path = os.path.join(init_dir, f"{name.lower()}_abundance_rank_{rank}_NMF.npy")
    if os.path.exists(path):
        return np.load(path).astype(np.float32)
    return None


def to_tensor(cube):
    """Convert a (H, W, C) cube to the (1, C, H, W) tensor the model expects, clamped to [0, 1]."""
    import torch
    gt = torch.tensor(cube)
    gt = gt.view(-1, gt.size(0), gt.size(1), gt.size(2)).permute(0, 3, 1, 2).contiguous()
    return torch.clamp(gt, 0, 1)
