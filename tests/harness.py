"""Core logic for the ATLAS self-test, shared by ``tests/selftest.py`` and the pytest
GPU smoke. Pure helpers + one ``run_case`` that drives the real package end to end.

Heavy/optional imports (torch, atlas.train, atlas.codec, the CUDA kernel) are deferred
into the functions that need them, so importing this module stays light enough for the
CPU-only CI to collect the pytest GPU test without a GPU present.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import testcases

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "raw"
INIT_DIR = REPO / "data" / "init"
KERNEL_DIR = REPO / "atlas-gsplat"
GOLDEN_PATH = Path(__file__).resolve().parent / "golden.json"

# Make the in-repo `atlas/` package importable when driven from a script, independent of
# CWD or whether the wheel is pip-installed (pytest happens to add the root itself).
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Raw filenames for the integrated benchmark datasets (mirrors atlas.data._load_known).
_KNOWN_FILES = {
    "urban": "Urban_R162.mat",
    "salinas": "Salinas.mat",
    "jasperridge": "jasperRidge2_R198.mat",
    "paviau": "PaviaU.mat",
    "paviac": "Pavia.mat",
    "whu_honghu": "WHU-Hi-HongHu.bsq",
}


# --- case lookup ------------------------------------------------------------------------
def all_cases() -> list[testcases.Case]:
    return list(testcases.CASES)


def get_case(name: str) -> testcases.Case:
    for c in testcases.CASES:
        if c.name == name:
            return c
    raise KeyError(f"no such test case: {name!r}")


# --- environment ------------------------------------------------------------------------
def _ensure_kernel_on_path() -> None:
    """Make the local atlas-gsplat checkout importable when the wheel isn't installed
    (the dev box ships a prebuilt csrc.so there; on a cluster you'd pip install it)."""
    if importlib.util.find_spec("atlas_gsplat") is None and KERNEL_DIR.is_dir():
        sys.path.insert(0, str(KERNEL_DIR))


def gpu_ready() -> bool:
    """True iff torch sees a CUDA device and the atlas_gsplat kernel imports."""
    _ensure_kernel_on_path()
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        import atlas_gsplat  # noqa: F401
        return True
    except Exception:
        return False


# --- datasets ---------------------------------------------------------------------------
def _is_known(case: testcases.Case) -> bool:
    return case.dataset.lower() in _KNOWN_FILES


def dataset_ref(case: testcases.Case) -> str:
    """What to hand atlas.data.load_cube: a known name, or an absolute cube path."""
    if _is_known(case):
        return case.dataset
    return str((REPO / case.dataset).resolve())


def dataset_available(case: testcases.Case) -> bool:
    if _is_known(case):
        return (DATA_DIR / _KNOWN_FILES[case.dataset.lower()]).exists()
    return (REPO / case.dataset).exists()


def init_key(case: testcases.Case) -> str:
    from atlas import data
    return data.init_key(dataset_ref(case))


def rank_for(case: testcases.Case, cfg: testcases.Config) -> int:
    if cfg.rank is not None:
        return cfg.rank
    from atlas import data
    return data.rank_for(init_key(case))


def ensure_init(case: testcases.Case, rank: int, auto: bool = True) -> str:
    """Return the NMF init key, generating the endmember/abundance .npy files if missing."""
    from atlas import data
    key = init_key(case)
    endmember_npy = INIT_DIR / f"{key}_endmember_rank_{rank}_NMF.npy"
    if endmember_npy.exists():
        return key
    if not auto:
        raise FileNotFoundError(
            f"missing NMF init for {key} (rank {rank}); run `atlas init --dataset "
            f"{case.dataset} --rank {rank}`")
    from atlas.endmember import nmf_initialization
    cube = data.load_cube(dataset_ref(case), data_dir=str(DATA_DIR))
    mat = np.transpose(cube, (2, 0, 1)).reshape(cube.shape[2], -1)  # (C, H*W)
    nmf_initialization(mat, rank, key, init_dir=str(INIT_DIR))
    return key


# --- metrics ----------------------------------------------------------------------------
def mean_psnr(ref: np.ndarray, rec: np.ndarray) -> float:
    """Mean per-band PSNR between two (H, W, C) cubes in [0, 1]."""
    ref = np.clip(ref, 0, 1).astype(np.float32)
    rec = np.clip(rec, 0, 1).astype(np.float32)
    mse = ((ref - rec) ** 2).mean(axis=(0, 1))
    return float(np.mean(10 * np.log10(1.0 / np.maximum(mse, 1e-12))))


# --- the end-to-end run -----------------------------------------------------------------
def run_case(case: testcases.Case, mode: str = "full", auto_init: bool = True,
             output_dir: str | None = None) -> dict:
    """Train an ATLAS model for one cube, serialize it to an .atlas bitstream, decode it
    back, and measure the round-trip. Returns a metrics dict; raises on any failure.

    The PSNR/bpppb reported are the *compressed-file* metrics (decoded through the real
    bitstream), so this exercises training, quantization, the entropy codec, and the
    CUDA reconstruction in one shot.
    """
    import torch

    from atlas import codec, data
    from atlas.train import SimpleTrainerHSI

    _ensure_kernel_on_path()
    cfg = case.full if mode == "full" else case.quick
    rank = rank_for(case, cfg)

    torch.manual_seed(0)
    np.random.seed(0)

    key = ensure_init(case, rank, auto=auto_init)
    ref = dataset_ref(case)
    cube = data.load_cube(ref, data_dir=str(DATA_DIR))
    endmember = data.load_endmember(key, rank, init_dir=str(INIT_DIR))
    a0 = data.load_abundance(key, rank, init_dir=str(INIT_DIR))
    gt = data.to_tensor(cube)

    out_dir = output_dir or tempfile.mkdtemp(prefix="atlas_selftest_")
    t0 = time.time()
    trainer = SimpleTrainerHSI(
        ground_truth=gt, endmember=endmember, image_name=key,
        num_points=cfg.num_points, max_num_points=cfg.max_num_points,
        grow_interval=1000, grow_stop_iter=cfg.grow_stop_iter,
        iterations=cfg.iterations, qat_start=cfg.qat_start,
        prune_interval=cfg.prune_interval, endmember_lr=cfg.endmember_lr,
        sam_weight=cfg.sam_weight, A0=a0,
        abundance_bits=cfg.abundance_bits, cholesky_bits=cfg.cholesky_bits,
        output_dir=out_dir)
    trainer.train()
    model = trainer.gaussian_model

    blob = Path(out_dir) / f"{key}.atlas"
    nbytes = codec.write_bitstream(model, blob)          # encode the trained (QAT) model
    meta, arrays = codec.read_bitstream(blob)            # pure-numpy parse
    bpppb = codec.payload_bpppb(meta, arrays)
    recon = codec.decompress(blob)                       # CUDA reconstruction
    psnr = mean_psnr(cube, recon)
    elapsed = time.time() - t0

    if tuple(recon.shape) != tuple(cube.shape):
        raise ValueError(f"recon shape {tuple(recon.shape)} != cube {tuple(cube.shape)}")

    return {
        "psnr": float(psnr), "bpppb": float(bpppb), "file_bytes": int(nbytes),
        "n_gaussians": int(meta["N"]), "shape": list(cube.shape), "rank": int(rank),
        "iterations": int(cfg.iterations), "seconds": float(elapsed), "mode": mode,
    }


# --- evaluation -------------------------------------------------------------------------
def evaluate(case: testcases.Case, metrics: dict, mode: str, golden: dict):
    """Check hard floors (always) and golden drift (full mode, if recorded).

    Returns ``(floor_ok, floor_msgs, drift_msgs)``. Floor failures fail the run; drift is
    a warning unless the caller treats it as strict.
    """
    psnr, bpppb = metrics["psnr"], metrics["bpppb"]
    floor_ok, floor_msgs = True, []

    psnr_floor = case.quick_psnr_min if mode == "quick" else case.psnr_min
    if psnr < psnr_floor:
        floor_ok = False
        floor_msgs.append(f"PSNR {psnr:.2f} < floor {psnr_floor:.2f} dB")
    if bpppb > case.bpppb_max:
        floor_ok = False
        floor_msgs.append(f"bpppb {bpppb:.4f} > ceiling {case.bpppb_max:.4f}")

    drift_msgs = []
    g = golden.get(case.name) if mode == "full" else None
    if g:
        if abs(psnr - g["psnr"]) > case.psnr_tol:
            drift_msgs.append(
                f"PSNR {psnr:.2f} vs golden {g['psnr']:.2f} (tol {case.psnr_tol} dB)")
        if abs(bpppb - g["bpppb"]) > g["bpppb"] * case.bpppb_tol_frac:
            drift_msgs.append(
                f"bpppb {bpppb:.4f} vs golden {g['bpppb']:.4f} (tol {case.bpppb_tol_frac:.0%})")
    return floor_ok, floor_msgs, drift_msgs


# --- golden store -----------------------------------------------------------------------
def load_golden() -> dict:
    if GOLDEN_PATH.exists():
        return json.loads(GOLDEN_PATH.read_text())
    return {}


def save_golden(golden: dict) -> None:
    GOLDEN_PATH.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
