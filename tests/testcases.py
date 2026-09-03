"""Declarative registry of ATLAS self-test cases (one source of truth).

Both ``tests/selftest.py`` and the pytest GPU smoke consume this registry.

Each case fully specifies how to train+compress one cube and the bar it must clear:

* **Hard floors** (``psnr_min`` / ``bpppb_max``) are enforced on every run. They catch
  gross breakage -- a dead kernel, a broken codec, a quality collapse -- and are
  deliberately loose.
* **Golden values** are recorded per case in ``tests/golden.json`` (written by
  ``selftest.py --record``) and checked against ``psnr_tol`` / ``bpppb_tol_frac``. They
  catch *drift*: a change that quietly costs a dB or inflates the rate. Until a golden
  value is recorded for a case, only the floors apply.

A case whose cube is absent is SKIPPED (not failed), so a new dataset lights up the
moment its file lands in ``data/raw/``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    """The full-pipeline training/compression configuration for one case."""

    num_points: int
    iterations: int
    qat_start: int
    max_num_points: int | None = None
    grow_stop_iter: int = 2000
    prune_interval: int = 100
    endmember_lr: float = 1e-4
    sam_weight: float = 0.05
    abundance_bits: int = 6
    cholesky_bits: int = 8
    rank: int | None = None  # None -> atlas.data.rank_for(key)


def _quick(cfg: Config) -> Config:
    """A fast, dataset-agnostic variant for --quick / the pytest smoke: short schedule,
    no D3 growth. Proves the pipeline runs and the bitstream round-trips, not quality."""
    return replace(
        cfg, num_points=min(cfg.num_points, 1200), max_num_points=None,
        iterations=500, qat_start=250, grow_stop_iter=200,
    )


@dataclass(frozen=True)
class Case:
    name: str
    dataset: str          # known dataset name, or a cube path relative to the repo root
    full: Config
    psnr_min: float       # hard floor, always enforced (full mode)
    bpppb_max: float      # hard ceiling, always enforced
    psnr_tol: float = 1.5         # golden tolerance, dB
    bpppb_tol_frac: float = 0.20  # golden tolerance, fraction of the golden rate
    quick_psnr_min: float = 12.0  # loose floor for --quick (just "it ran and isn't garbage")
    forthcoming: bool = False     # dataset not integrated yet; here so it auto-activates
    min_vram_gb: float = 0.0      # skip (don't OOM/spill) if the GPU has less total VRAM

    @property
    def quick(self) -> Config:
        return _quick(self.full)


# --- synthetic smoke: always runnable, no data download, seconds to run -----------------
_SYNTHETIC = [
    Case(
        name="tiny_cube", dataset="examples/tiny_cube.npy",
        # 48x48 = 2304 px, so keep N well below the pixel count: at ~1 Gaussian/px QAT +
        # pruning collapse the fit. 1000 pts is a stable smoke (1200 reached ~36 dB).
        full=Config(num_points=1000, iterations=800, qat_start=400, grow_stop_iter=200, rank=4),
        # Tiny voxel count means per-Gaussian overhead dominates the rate, so bpppb is high by
        # construction. This is a smoke test (does it train + round-trip?), not a rate
        # benchmark: the ceiling is loose and PSNR is the real signal.
        psnr_min=25.0, bpppb_max=6.0, quick_psnr_min=10.0,
    ),
]

# --- integrated benchmark datasets (present in data/raw + data/init) --------------------
# Floors are derived from the known two-stage-QAT results with a wide margin; the recorded
# golden values (selftest.py --record) provide the tight drift check.
_BENCHMARKS = [
    Case(
        name="paviau", dataset="paviau",
        full=Config(num_points=7250, max_num_points=14500, grow_stop_iter=2000,
                    iterations=10500, qat_start=5250, rank=12),
        psnr_min=34.0, bpppb_max=0.25,
    ),
    Case(
        name="salinas", dataset="salinas",
        full=Config(num_points=2500, max_num_points=5000, grow_stop_iter=1500,
                    iterations=7000, qat_start=3500, rank=12),
        psnr_min=38.0, bpppb_max=0.15,
    ),
    Case(
        name="jasperridge", dataset="jasperridge",
        full=Config(num_points=1375, max_num_points=2750, grow_stop_iter=1500,
                    iterations=8000, qat_start=4000, rank=10),
        psnr_min=32.0, bpppb_max=0.25,
    ),
    Case(
        name="urban", dataset="urban",
        full=Config(num_points=14500, max_num_points=29000, grow_stop_iter=1500,
                    iterations=7000, qat_start=3500, rank=12),
        # The 29k-Gaussian operating point trades rate for quality (~43.8 dB @ ~0.236 bpppb),
        # higher than the 14.5k two-stage-QAT point; ceiling brackets it with headroom.
        psnr_min=33.0, bpppb_max=0.32,
    ),
    Case(
        name="paviac", dataset="paviac",
        # Pavia Centre: same ROSIS sensor/spectral family as PaviaU, ~6.8x the pixels. N is
        # scaled from PaviaU's Gaussian density; floors are provisional until confirmed with a
        # --record run on the real cube.
        full=Config(num_points=36000, max_num_points=72000, grow_stop_iter=2500,
                    iterations=12000, qat_start=6000, rank=12),
        psnr_min=32.0, bpppb_max=0.25,
    ),
    Case(
        name="whu_honghu", dataset="whu_honghu",
        # WHU-Hi-HongHu: 940x475x270 (ENVI). ~3.9x PaviaU px and 2.6x its bands, so the
        # spectral low-rank assumption (rank 12) is the real test. N scaled to PaviaU density;
        # floors provisional until confirmed with a --record run.
        # Training's HxWxC activation footprint exceeds a 10 GB 3080: on WSL2 the driver
        # spills to shared host RAM instead of raising OOM and the VM froze (2026-08-18).
        # Peak unmeasured; 12 GB is a lower-bound gate, measure before trusting a 12 GB card.
        full=Config(num_points=28000, max_num_points=56000, grow_stop_iter=2500,
                    iterations=12000, qat_start=6000, rank=12),
        psnr_min=28.0, bpppb_max=0.35, min_vram_gb=12.0,
    ),
    Case(
        name="chikusei", dataset="chikusei",
        # Chikusei (ENVI), cropped to the largest all-valid rectangle 2372x2196x128 -- ~45x
        # the PaviaU crop, the scalability stress test. The whole cube sits in VRAM (no
        # tiling); the forward's full H*W*C reconstruction + autograd copies need ~16 GB
        # *independent of N*, so this does NOT fit a 10 GB 3080 (measured 15.8 GB, spills to
        # shared RAM). min_vram_gb gates it to a big GPU (A100/H100 on SharcNet). Because
        # memory is ~N-independent here, N is bounded by time, not VRAM -- raise it on an
        # 80 GB H100. Config + floors provisional until a --record run on a big GPU.
        full=Config(num_points=80000, max_num_points=160000, grow_stop_iter=3000,
                    iterations=10000, qat_start=5000, rank=16),
        psnr_min=28.0, bpppb_max=0.40, min_vram_gb=24.0,
    ),
]

# --- forthcoming "bigger" datasets ------------------------------------------------------
# Not-yet-integrated datasets go here, referenced by their expected raw-cube path so they
# auto-skip until downloaded and auto-activate afterwards (configs provisional). See
# docs/findings for the dataset survey. (Empty: PaviaU/Salinas/JR/Urban/Pavia Centre/
# WHU-Hi-HongHu/Chikusei are all integrated above.)
_FORTHCOMING = []

CASES: list[Case] = [*_SYNTHETIC, *_BENCHMARKS, *_FORTHCOMING]
