# ATLAS

ATLAS (Adaptive Transfer of Low-rank Abundance Splatting) compresses hyperspectral images
(HSI) with low-rank spectral Gaussian splatting.

It first decomposes a data cube with NMF into a small set of endmember spectra and low-rank
abundance maps, then represents those abundance maps as a compact set of optimizable 2D
Gaussians. Each Gaussian stores only `k` abundance coefficients (`k` is about 10-12) rather
than the full `l` spectral bands, which is what makes Gaussian splatting tractable for HSI.
The trained representation is itself the compressed file for that cube.

ATLAS builds on LoR-SGS (IEEE TGRS 2025) and ports the GaussianImage++ (AAAI 2026)
techniques that transfer to the HSI domain: distortion-driven densification, two-stage LSQ+
quantization, learnable endmembers, and gradient-guided initialization. It drops the ones
that do not (content-aware filters, position and progressive quantization, spatial delta
coding). On four benchmarks it beats published LoR-SGS on 6 of 8 rate-distortion targets, by
+1.20 dB PSNR on average.

ATLAS is a per-image representation, like an INR. You train it on the cube you want to
compress, and there is no pretrained model to download.

## Installation

ATLAS needs a CUDA GPU. It installs as two packages: the pure-Python `atlas-hsi` and the
CUDA kernel `atlas-gsplat` (see [its README](atlas-gsplat/README.md)).

```bash
# 1. PyTorch matching your CUDA (if not already installed)
pip install torch

# 2. The CUDA rasterization kernel (builds from source; needs nvcc and the torch above).
#    --no-build-isolation lets the build see your installed torch (see Troubleshooting).
pip install --no-build-isolation "git+https://github.com/jacobh3ll0/atlas-gsplat"

# 3. ATLAS
pip install atlas-hsi
# optional extras:
pip install "atlas-hsi[viz]"    # matplotlib plotting helpers
pip install "atlas-hsi[demo]"   # Streamlit demo app
```

For local development:

```bash
pip install -e ".[viz,dev]"
```

### Troubleshooting the install

If the kernel will not build, or training cannot find CUDA, run:

```bash
atlas doctor
```

It checks your torch build, the detected CUDA toolkit (`nvcc`), and whether `atlas-gsplat`
is installed, then prints what is wrong. The most common fresh-install trap is a CUDA
major-version mismatch: torch built for one CUDA major (for example 13.x) while your toolkit
is another (for example 12.8). The build fails until they agree. `atlas doctor` reports this
and prints the exact fix. The toolkit is auto-discovered from `CUDA_HOME`, `nvcc` on `PATH`,
or `/usr/local/cuda*`. Add `--build` to force a real kernel compile and load as a final check.

A second trap shows up while building `atlas-gsplat`: the build fails with
`ModuleNotFoundError: No module named 'torch'` even though torch is installed. That is pip's
build isolation hiding your torch from the extension build. Install the kernel with
`--no-build-isolation` (as shown above) so the compile can see it.

## Quickstart

```bash
# 1. NMF initialization, once per dataset/rank. Writes data/init/*.npy
atlas init --input paviau --rank 12

# 2. Train (FP32). The checkpoint in atlas_runs/ is the compressed representation.
atlas train --input paviau --num_points 14500 --iterations 7000

# 3. Recommended pipeline: D3 densification, two-stage QAT, learnable endmembers, SAM
atlas train --input paviau --num_points 7250 --max_num_points 14500 \
  --grow_stop_iter 2000 --iterations 10500 --qat_start 5250 \
  --prune_interval 100 --endmember_lr 1e-4 --sam_weight 0.05 \
  --cholesky_bits 8
# A quantized (QAT) run also writes a compressed bitstream at <logdir>/compressed.atlas

# 4. Decompress a bitstream back into a cube (.npy). --reference reports reconstruction PSNR
atlas decompress atlas_runs/HSI/.../compressed.atlas -o recon.npy --reference paviau
```

You can also pass a path to your own cube instead of a dataset name:

```bash
atlas init  --input /path/to/cube.mat --rank 12
atlas train --input /path/to/cube.mat --num_points 8000 --iterations 7000
```

### Python API

```python
import atlas

cube = atlas.load_cube("paviau")                 # (H, W, C), normalized
E    = atlas.load_endmember("paviau", rank=12)   # (rank, channels)
A0   = atlas.load_abundance("paviau", rank=12)    # for gradient-guided init
gt   = atlas.to_tensor(cube)                      # (1, C, H, W)

result = atlas.train_nd(gt, endmember=E, image_name="paviau",
                        iterations=7000, num_points=14500, A0=A0)
print(result["psnr"], result["bpppb"])
```

### Visualization (`[viz]` extra)

The optional `[viz]` extra adds matplotlib helpers under `atlas.viz`. They operate on plain
numpy arrays and need no GPU, so you can inspect cubes, abundance maps, and endmembers on a
machine without CUDA:

```python
from atlas import viz

cube = atlas.load_cube("jasperridge")
E    = atlas.load_endmember("jasperridge", rank=10)
A0   = atlas.load_abundance("jasperridge", rank=10)   # flattened (H*W, k)
H, W = cube.shape[:2]

viz.show_pseudo_rgb(cube, show=True)                       # quick RGB preview
viz.show_abundance_maps(A0, shape=(H, W), show=True)       # grid of abundance maps
viz.plot_endmembers(E, show=True)                          # endmember spectra
viz.plot_spectra(cube, recon, n=3, show=True)              # orig vs recon signatures
viz.show_error_map(cube, recon, metric="sam", show=True)   # per-pixel error map
viz.plot_rd_curve({"ATLAS": [(0.10, 37.2), (0.06, 31.5)]}, show=True)  # RD curve
```

Every helper returns `(fig, ax)` so you can compose or save them; pass `show=True` to display
immediately. `show_abundance_maps` accepts `(H, W, k)`, `(k, H, W)`, or flattened `(H*W, k)`
(the NMF init layout, with `shape=(H, W)`).

## CLI reference (`atlas train`)

| Flag | Default | Description |
|------|---------|-------------|
| `--input` / `--dataset` | required | Dataset name (`urban`/`salinas`/`jasperridge`/`paviau`) or `.mat`/`.npy` path |
| `--iterations` | 8000 | Training iterations |
| `--num_points` | required | Number of Gaussians (start count if D3 enabled) |
| `--max_num_points` | =num_points | D3: grow to this many Gaussians |
| `--grow_stop_iter` | 5000 | D3: stop densifying at this iter |
| `--qat_start` | 0 | Iteration to begin two-stage QAT (0 = FP32, no quantization) |
| `--prune_interval` | 100 | Prune degenerate Gaussians every N iters during QAT |
| `--endmember_lr` | 0 | Learnable-endmember LR (0 = frozen; 1e-4 recommended) |
| `--sam_weight` | 0.0 | SAM loss weight (0.05 recommended) |
| `--rank` | per-dataset | NMF rank (10 for JasperRidge, else 12) |
| `--output-dir` | `atlas_runs` | Where checkpoints and logs are written |
| `--abundance_bits` / `--cholesky_bits` | 6 / 10 | advanced: LSQ+ bit-widths (used under QAT) |
| `--lsq-grad-scale` | off | experimental: LSQ+ gradient scaling for the quantizer scale and offset (rate-distortion-neutral with the Adan optimizer) |

Techniques the paper found do not transfer to HSI are not exposed: content-aware filters,
spatial delta coding, and progressive QAT.

## Compressed bitstream

A QAT run writes `compressed.atlas` next to its checkpoints. This is the compressed file for
the cube: a binary container holding positions and the endmember as FP16, and the cholesky
covariance and abundances as LSQ+ integer codes, ANS entropy-coded. `atlas decompress` reads
it back and reconstructs the cube with the same forward pass, so the decoded result matches
the trained model:

```bash
atlas decompress run/compressed.atlas -o recon.npy --reference paviau
```

The trainer's reported `bpppb` is the entropy payload of this file (the on-disk file adds a
small fixed container header). For programmatic access, use
`atlas.codec.write_bitstream(model, path)` and `atlas.codec.decompress(path)`.

## Datasets

| Dataset | Shape | Rank | Typical N | Iterations |
|---------|-------|------|-----------|------------|
| PaviaU | 340x340x103 | 12 | 10000-23000 | 7000 |
| Salinas | 180x180x204 | 12 | 5000 | 7000 |
| JasperRidge | 100x100x198 | 10 | 600-2750 | 8000 |
| Urban | 307x307x162 | 12 | 14500-29000 | 7000 |

Place raw `.mat` files in `data/raw/`; `atlas init` writes NMF outputs to `data/init/`.
Override with `--data-dir` and `--init-dir`.

## How it works

1. Decompose. NMF factors the cube `X ≈ A·E` into abundance maps `A` and endmembers `E`
   (rank `k ≪ l`).
2. Splat. Fit `N` 2D Gaussians (position, Cholesky covariance, `k` abundances) to reconstruct
   `A`. D3 grows `N` by placing Gaussians at high-error pixels.
3. Quantize and encode. Two-stage QAT applies LSQ+ (FP16 positions, signed Cholesky, unsigned
   abundances), degenerate Gaussians are pruned, and parameters are entropy-coded.

## Citation

```bibtex
@inproceedings{rempel2026atlas,
  title  = {ATLAS: Hyperspectral Image Compression via Adaptive Transfer of Low-rank Abundance Splatting},
  author = {Rempel, Jacob},
  year   = {2026}
}
```

Built on LoR-SGS (Li et al., IEEE TGRS 2025) and GaussianImage++ (Li et al., AAAI 2026).

## License

MIT (see `LICENSE`). The `atlas-gsplat` kernel is Apache-2.0 (derived from nerfstudio gsplat).
