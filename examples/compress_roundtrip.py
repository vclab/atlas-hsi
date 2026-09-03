"""Round-trip demo: load a dataset, compress it to an ATLAS bitstream, decompress it back.

ATLAS compression means training a 2D Gaussian-splat representation of one hyperspectral
cube, then serializing that trained model to a self-describing binary file (the bitstream).
This script runs the whole loop end to end on one dataset and reports the file size, the
rate (bpppb), and the reconstruction PSNR.

Needs a CUDA GPU and the atlas-gsplat kernel, and a dataset that `atlas init` has already
processed (its NMF files under data/init/). Run from the repo root so the data/ paths
resolve:

    python examples/compress_roundtrip.py            # default: jasperridge, short training
    python examples/compress_roundtrip.py paviau
"""
import sys
from pathlib import Path

import numpy as np

from atlas import codec, data
from atlas.train import SimpleTrainerHSI


def mean_psnr(ref, rec):
    """Mean per-band PSNR between two ``(H, W, C)`` cubes in [0, 1]."""
    ref = np.clip(ref, 0, 1).astype(np.float32)
    rec = np.clip(rec, 0, 1).astype(np.float32)
    mse = ((ref - rec) ** 2).mean(axis=(0, 1))  # per-band MSE
    return float(np.mean(10 * np.log10(1.0 / np.maximum(mse, 1e-12))))


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "jasperridge"
    key = data.init_key(dataset)
    rank = data.rank_for(key)

    # --- load ---
    cube = data.load_cube(dataset)              # (H, W, C), normalized to [0, 1]
    endmember = data.load_endmember(key, rank)  # NMF endmembers written by `atlas init`
    A0 = data.load_abundance(key, rank)         # NMF abundance init
    gt = data.to_tensor(cube)
    print(f"Loaded {dataset}: cube {cube.shape}, rank {rank}")

    # --- compress: train a quantization-aware (QAT) model, then serialize it ---
    # Short schedule for a quick demo. qat_start must be < iterations so the model ends
    # quantized; codec.encode_model requires a QAT model.
    trainer = SimpleTrainerHSI(
        ground_truth=gt, endmember=endmember, image_name=key,
        num_points=2000, iterations=600, qat_start=300,
        prune_interval=100, endmember_lr=1e-4, sam_weight=0.05, A0=A0,
        abundance_bits=6, cholesky_bits=8,
    )
    trainer.train()
    model = trainer.gaussian_model

    out_path = Path("roundtrip.atlas")
    nbytes = codec.write_bitstream(model, out_path)
    meta, arrays = codec.read_bitstream(out_path)
    print(f"\nCompressed   -> {out_path}  "
          f"({nbytes} bytes on disk, payload {codec.payload_bpppb(meta, arrays):.4f} bpppb)")

    # --- uncompress: read the bitstream back into a cube ---
    recon = codec.decompress(out_path)          # (H, W, C); needs the CUDA kernel
    print(f"Decompressed -> cube {recon.shape}")
    print(f"Reconstruction mean per-band PSNR: {mean_psnr(cube, recon):.4f} dB")


if __name__ == "__main__":
    main()
