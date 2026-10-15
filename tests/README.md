# Tests

ATLAS has two test layers. The pytest suite checks individual parts of the package: the
entropy codec, data loading, `atlas doctor`, `atlas init`, the plots, and the CUDA kernel.
The self-test runs the whole pipeline (train, write an `.atlas` file, decode it) on every
dataset you have downloaded and checks the PSNR and rate it reaches.

The tests live in the repository and are not part of the PyPI package, so run them from
a clone.

## Quickstart

```bash
pip install -e ".[viz,dev]"          # from the repository root
pytest tests/ -m "not gpu"           # CPU tests only
python tests/selftest.py --quick     # every layer, with a short training schedule
```

## Unit tests

```bash
pytest tests/                        # all 38 tests; the 6 GPU tests need a GPU
pytest tests/ -m "not gpu"           # the 32 CPU tests
```

Tests marked `gpu` skip themselves when there is no CUDA device or the kernel cannot load
(see `conftest.py`). CI (`.github/workflows/ci.yml`) runs `pytest tests/` on every push to
`main` and every pull request. It installs no torch, so only the CPU tests run there.

| File | Tests | GPU | What it checks |
|------|------:|:---:|----------------|
| `test_codec.py` | 5 | | ANS entropy-codec round trips and the `.atlas` container |
| `test_data.py` | 4 | | Cube loading, default ranks, NMF init keys |
| `test_diagnostics.py` | 18 | | `atlas doctor`: version matching, toolkit and ninja discovery |
| `test_cli_init.py` | 1 | | `atlas init` writes NMF files under the key `atlas train` reads |
| `test_viz.py` | 4 | | Plot helpers, on matplotlib's headless Agg backend |
| `test_roundtrip_gpu.py` | 1 | yes | The tiny cube trains, writes an `.atlas` file, and decodes |
| `kernel/test_map_gaussians.py` | 1 | yes | CUDA Gaussian-to-tile mapping against a PyTorch reference |
| `kernel/test_get_tile_bin_edges.py` | 2 | yes | Tile bins against the reference, including fewer intersections than tiles |
| `kernel/test_tile_bins_regression.py` | 2 | yes | 600 small Gaussians on a 760x760 image, forward and backward |

The PyTorch reference for the kernel tests is in `kernel/_reference.py`. The last two kernel
files guard a fixed bug. The kernel used to size its tile-bin buffer by the number of
intersections, so an image with more tiles than intersections read and wrote past the end of
the buffer.

## Self-test

```bash
python tests/selftest.py                      # full schedule on every downloaded dataset
python tests/selftest.py --quick              # short schedule (seconds to minutes per case)
python tests/selftest.py --list               # list the cases and which datasets are present
python tests/selftest.py --only paviau salinas
python tests/selftest.py --only paviau --record   # refresh golden.json for paviau
```

The self-test runs three stages in order:

1. Unit: the CPU pytest tests above.
2. Integration: the synthetic 48x48x20 cube in `examples/tiny_cube.npy`, so a GPU check runs
   even with no datasets downloaded.
3. Datasets: every benchmark case whose cube is in `data/raw/`.

Each case trains a model, writes it to an `.atlas` file, decodes the file through the real
codec, and reports the mean per-band PSNR and the bits per pixel per band (`bpppb`) of that
file. The GPU stages are skipped when there is no CUDA device or the kernel cannot load; run
`atlas doctor` to see why.

| Flag | Effect |
|------|--------|
| `--quick` | Short schedule: at most 1200 Gaussians, 500 iterations, QAT from 250, no growth |
| `--only NAME ...` | Run only these cases |
| `--record` | Save the PSNR and `bpppb` of each passing case to `golden.json` (full schedule only) |
| `--strict` | Fail the run on golden drift, not only warn |
| `--no-cpu` | Skip the pytest stage |
| `--verbose` | Show the trainer's output |
| `--list` | List the cases and exit |

The exit code is 1 when a case fails its floor or errors, and 0 otherwise. Drift alone
returns 0 unless you pass `--strict`.

## Test cases

`testcases.py` defines every case, and `harness.py` holds the code that runs them. A case
names a cube, the training configuration, and two kinds of checks.

Floors fail the run. Each case has a minimum PSNR and a maximum `bpppb`, set well below the
known results so they catch real breakage (a dead kernel, a broken codec, a quality
collapse) rather than noise. In `--quick` mode the PSNR floor drops to 12 dB (10 dB for the
tiny cube), since a 500-iteration run only shows that the pipeline works.

Golden values flag drift. `golden.json` records the PSNR and `bpppb` of one full-schedule
run per case. A later run that moves more than 1.5 dB in PSNR or 20% in `bpppb` gets a
drift warning. Golden values are not checked in `--quick` mode.

| Case | File in `data/raw/` | Gaussians (start to max) | Iterations | PSNR floor (dB) | bpppb ceiling | Golden PSNR / bpppb |
|------|---------------------|--------------------------|-----------:|----------------:|--------------:|---------------------|
| `tiny_cube` | (`examples/tiny_cube.npy`) | 1000 | 800 | 25 | 6.0 | 35.96 / 1.987 |
| `paviau` | `PaviaU.mat` | 7250 to 14500 | 10500 | 34 | 0.25 | 41.13 / 0.151 |
| `salinas` | `Salinas.mat` | 2500 to 5000 | 7000 | 38 | 0.15 | 42.82 / 0.099 |
| `jasperridge` | `jasperRidge2_R198.mat` | 1375 to 2750 | 8000 | 32 | 0.25 | 41.47 / 0.179 |
| `urban` | `Urban_R162.mat` | 14500 to 29000 | 7000 | 33 | 0.32 | 43.75 / 0.236 |
| `paviac` | `Pavia.mat` | 36000 to 72000 | 12000 | 32 | 0.25 | 38.94 / 0.110 |
| `whu_honghu` | `WHU-Hi-HongHu.bsq` and `.hdr` | 28000 to 56000 | 12000 | 28 | 0.35 | 35.31 / 0.058 |
| `chikusei` | `HyperspecVNIR_Chikusei_20140729.bsq` and `.hdr` | 80000 to 160000 | 10000 | 28 | 0.40 | 40.09 / 0.034 |

The tiny cube's rate is high because the per-Gaussian overhead dominates a 2304-pixel
image. Its ceiling is loose on purpose, and PSNR is the check that matters for it.

A case is skipped, not failed, when its cube is missing, or when the GPU has less memory
than the case needs: 12 GB for `whu_honghu` and 24 GB for `chikusei`. Both train the whole
cube in GPU memory. On WSL2 the driver spills to host RAM instead of raising an
out-of-memory error, which froze the machine once, so the self-test checks the card's memory
before it starts. If the NMF initialization for a case is missing from `data/init/`, the
self-test computes it first.

## Adding a case

1. Add a `Case` to `_BENCHMARKS` in `testcases.py`. For a new named dataset, also add its
   raw filename to `_KNOWN_FILES` in `harness.py` (it mirrors `atlas.data`). A cube outside
   the named datasets can instead be given as a path relative to the repository root.
2. Set the floors well below the results you expect.
3. Run `python tests/selftest.py --only NAME --record` to save its golden values, and commit
   the updated `golden.json`.

After a change that is meant to move the numbers, rerun with `--record` for the affected
cases. `--record` only rewrites the cases that ran and passed their floors.
