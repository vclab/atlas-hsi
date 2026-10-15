#!/bin/bash
# Regenerate every record in records/ (paper Appendix G). Run from the repository root.
# Stage 1 needs a CUDA GPU (the kernel compiles on first use); WHU-Hi-HongHu needs >= 12 GB VRAM
# (we trained it on an A100). Stage 2 is CPU only. Datasets go in data/raw/ (see README.md).
set -euo pipefail
D=reproduce/downstream
R=$D/recons
E="python $D/downstream_eval.py"

# --- Stage 1: inputs -------------------------------------------------------------
atlas init --input whu_honghu --rank 16               # NMF init for the rank-16 ablation
python $D/make_recons.py paviau paviac whu_honghu     # ATLAS recon.npy + abund.npy
python $D/make_recons.py whu_honghu --rank 16         # rank-16 abundances
python $D/make_jp2k.py                                # JPEG2000 at the matched rate
python $D/make_blur.py                                # HongHu blur control

# --- Stage 2: classification, random and 32 px blocked splits ----------------------
for split in random blocked; do
  for s in paviau paviac whu_honghu; do
    $E --split $split --scene $s --features original
    $E --split $split --scene $s --features recon --recon $R/$s/recon.npy
    $E --split $split --scene $s --features recon --recon $R/$s/jp2_recon.npy --tag jp2
    $E --split $split --scene $s --features abundance --abundance $R/$s/abund.npy
    $E --split $split --scene $s --features a0
  done
  # HongHu only: blur control and the rank-16 ablation
  $E --split $split --scene whu_honghu --features recon --recon $R/whu_honghu/blur_recon.npy --tag blur
  $E --split $split --scene whu_honghu --features abundance --abundance $R/whu_honghu_rank16/abund.npy --tag rank16
  $E --split $split --scene whu_honghu --features a0 --rank 16 --tag rank16
done

# --- 96 px blocks: the border-leakage diagnostic (HongHu original and JPEG2000 only) --
$E --split blocked --block-size 96 --scene whu_honghu --features original --tag b96
$E --split blocked --block-size 96 --scene whu_honghu --features recon --recon $R/whu_honghu/jp2_recon.npy --tag jp2_b96
