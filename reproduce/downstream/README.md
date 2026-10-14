# Downstream classification (paper Appendix G)

This folder holds everything behind the downstream evaluation in Appendix G of the ATLAS
paper: the scripts, the exact list of commands, and the per-sample records for both
classifiers. Every number in the paper's downstream table reproduces from `records/`.

## Contents

| File | Purpose |
|------|---------|
| `run_all.sh` | Every command that produced a file in `records/`, in order |
| `make_recons.py` | Trains ATLAS at the paper's operating points, writes the `.atlas` bitstream, and decodes it to `recon.npy` (full cube) and `abund.npy` (splatted abundances) |
| `make_jp2k.py` | JPEG2000 at the same bits per pixel per band as ATLAS |
| `make_blur.py` | HongHu blur control: the original cube blurred spatially with sigma 1.5 px |
| `downstream_eval.py` | Per-pixel SVM-RBF and random forest, random or spatially blocked splits |
| `records/*.json` | One file per condition, with all 10 per-sample results |
| `records/*.log` | Console output of the original runs (local paths removed) |

## Reading the records

Each JSON holds one condition: the scene, the feature set, the split protocol and block
size, 100 training pixels per class, the base seed, a `summary`, and `per_draw`. The
`per_draw` list has the 10 seed-paired samples: sample `k` uses seed `k` for the training
pixels and the random forest in every condition, so differences between conditions are
paired. `summary` gives the mean and standard deviation (`numpy.std`) of overall accuracy
(OA), average accuracy (AA), and kappa for the SVM (`svm_*`) and the random forest
(`rf_*`). Values are fractions; the paper prints them as percentages. The table reports the
SVM; random-forest numbers quoted in the text come from the `rf_*` entries.

File names follow `<scene>_<features>[_blocked][_<tag>].json`, with scenes `paviau`,
`paviac` (Pavia Centre), and `whu_honghu`.

| Paper table row | Random split | Blocked split (32 px) |
|-----------------|--------------|-----------------------|
| Original | `<scene>_original` | `<scene>_original_blocked` |
| ATLAS | `<scene>_recon` | `<scene>_recon_blocked` |
| JPEG2000 | `<scene>_recon_jp2` | `<scene>_recon_blocked_jp2` |
| Gaussian blur (no codec), HongHu | `whu_honghu_recon_blur` | `whu_honghu_recon_blocked_blur` |
| Splatted abundances (12-d) | `<scene>_abundance` | `<scene>_abundance_blocked` |
| NMF A0 control (12-d) | `<scene>_a0` | `<scene>_a0_blocked` |
| Splatted abundances (16-d), HongHu | `whu_honghu_abundance_rank16` | `whu_honghu_abundance_blocked_rank16` |
| NMF A0 control (16-d), HongHu | `whu_honghu_a0_rank16` | `whu_honghu_a0_blocked_rank16` |

The 96 px block diagnostic in the leakage-control paragraph is
`whu_honghu_original_blocked_b96` and `whu_honghu_recon_blocked_jp2_b96`. Only those two
conditions were run at 96 px.

## Reproducing

Install ATLAS and the kernel as in the top-level README, then the evaluation extras:

```bash
pip install scikit-learn scipy glymur   # glymur needs the OpenJPEG library
```

Place the cubes and ground truth in `data/raw/`: `PaviaU.mat` and `PaviaU_gt.mat`,
`Pavia.mat` and `Pavia_gt.mat`, and the ENVI pairs `WHU-Hi-HongHu.bsq`/`.hdr` and
`WHU-Hi-HongHu_gt.bsq`/`.hdr`. Then, from the repository root:

```bash
bash reproduce/downstream/run_all.sh
```

Stage 1 trains and decodes on the GPU and writes inputs to `reproduce/downstream/recons/`
(ignored by git, about 2 GB). WHU-Hi-HongHu needs at least 12 GB of VRAM; we trained it on
an A100. Stage 2 is CPU only and writes new files to `records/`, overwriting the shipped
ones, so copy them aside first if you want to compare.

Training is not bit-for-bit deterministic, because the rasterizer's backward pass
accumulates gradients with atomic adds. A rerun lands within run-to-run variance of the
recorded operating points (about 0.4 dB on PaviaU, 0.05 dB on the other scenes), and
accuracies move by a similarly small amount. Given the same input cubes, Stage 2 is
deterministic. The shipped records are the exact outputs reported in the paper.

## Notes

- Pavia University uses the bottom 340 rows of the scene (rows 270 to 609), the crop used
  throughout the paper. Class 5 (painted metal sheets) lies outside it, so the scene has 8
  classes and its accuracies are not comparable to published full-scene numbers.
- The original condition is the band-max-normalized cube that ATLAS trains on and
  reconstructs, not raw radiance.
- `abund.npy` is the compressed-domain representation: `atlas.codec.decompress(path,
  abundance=True)` stops before the endmember mixing step, so the spectral cube is never
  formed.
- JPEG2000 rates are measured on the whole `.jp2` file, while ATLAS reports its entropy
  payload. The difference is small next to the PSNR gap between the two codecs.
