"""JPEG2000 matched-rate condition for the downstream eval (paper Appendix G).

Encodes each scene's band-normalized cube (the exact tensor ATLAS trains on) as a
single multi-component JPEG2000 Part 1 codestream at the ATLAS operating point's
achieved bpppb (read from recons/<scene>/metrics.json), decodes it, and saves
jp2_recon.npy next to the ATLAS recon for the same classification eval.

Configuration notes for the paper: input scaled to uint16 (the 16 bpppb baseline the
rate convention assumes); one codestream with C components, so OpenJPEG allocates rate
globally across bands; no spectral decorrelation (Part 1 has none beyond 3 components),
which is the standard JPEG2000 baseline in HSI compression papers.

    python reproduce/downstream/make_jp2k.py [scene ...]
"""

import json
import sys
from pathlib import Path

import glymur
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from atlas.data import load_cube  # noqa: E402

RECONS = Path(__file__).resolve().parent / "recons"


def main():
    for scene in (sys.argv[1:] or ["paviau", "paviac", "whu_honghu"]):
        cube = load_cube(scene, data_dir=str(REPO / "data" / "raw")).astype(np.float32)
        bpppb = json.loads((RECONS / scene / "metrics.json").read_text())["bpppb"]
        cratio = 16.0 / bpppb  # ratio vs the uint16 raw size = 16 bpppb baseline

        q = np.clip(np.round(cube * 65535.0), 0, 65535).astype(np.uint16)
        jp2_path = RECONS / scene / "jp2.jp2"
        glymur.Jp2k(str(jp2_path), data=q, cratios=[cratio])

        rec = glymur.Jp2k(str(jp2_path))[:].astype(np.float32) / 65535.0
        np.save(RECONS / scene / "jp2_recon.npy", rec)

        mse = ((cube - rec) ** 2).mean(axis=(0, 1))
        psnr = float(np.mean(10 * np.log10(1.0 / np.maximum(mse, 1e-12))))
        achieved = jp2_path.stat().st_size * 8 / cube.size
        print(f"{scene}: target {bpppb:.4f} bpppb (cratio {cratio:.0f}) -> "
              f"file {achieved:.4f} bpppb, mean per-band PSNR {psnr:.2f} dB", flush=True)


if __name__ == "__main__":
    main()
