"""Produce ATLAS reconstructions for the downstream eval (paper Appendix G).

For each scene, trains at the harness full config (the operating points recorded in
tests/golden.json), writes the .atlas bitstream, decodes it twice (the full cube and,
by partial decoding, the splatted abundance map), and saves recon.npy, abund.npy, and
a metrics JSON under recons/<scene>/ next to this script. Reuses
tests/harness.run_case so the training and codec path is identical to the one that
recorded the goldens; achieved PSNR should land within run-to-run variance of
golden.json (PaviaU spreads ~0.4 dB, the others ~0.05 dB).

Needs the CUDA kernel. Scenes already holding a recon.npy are skipped, so the job
can be re-run after an interruption:

    python reproduce/downstream/make_recons.py [scene ...]
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))  # harness.py uses flat imports (import testcases)

import harness  # noqa: E402
import testcases  # noqa: E402

from atlas import codec  # noqa: E402

OUT = Path(__file__).resolve().parent / "recons"
DEFAULT_SCENES = ("paviau", "paviac", "whu_honghu")


def main():
    ap = argparse.ArgumentParser(description="Train + decode recons for the downstream eval")
    ap.add_argument("scenes", nargs="*", help=f"default: {' '.join(DEFAULT_SCENES)}")
    ap.add_argument("--rank", type=int, default=None,
                    help="override the case's NMF rank (ablation); outputs go to "
                         "<scene>_rank<R>/ and the golden comparison is skipped")
    args = ap.parse_args()
    scenes = args.scenes or list(DEFAULT_SCENES)
    golden = json.loads((REPO / "tests" / "golden.json").read_text())
    for name in scenes:
        case = next(c for c in testcases.CASES if c.name == name)
        if args.rank:
            case = dataclasses.replace(
                case, full=dataclasses.replace(case.full, rank=args.rank))
            out_dir = OUT / f"{name}_rank{args.rank}"
        else:
            out_dir = OUT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / "recon.npy").exists():
            print(f"=== {name}: recon.npy exists, skipping ===", flush=True)
            continue
        # Same VRAM gate as selftest.py; run_case itself does not check it, and on WSL2
        # exceeding VRAM spills to shared host RAM and can freeze the VM instead of OOMing.
        if case.min_vram_gb and harness.gpu_vram_gb() < case.min_vram_gb:
            print(f"=== {name}: SKIP, needs >= {case.min_vram_gb:.0f} GB VRAM "
                  f"(have {harness.gpu_vram_gb():.1f}); train on SharcNet ===", flush=True)
            continue

        print(f"=== {name}: training at full config ===", flush=True)
        t0 = time.time()
        metrics = harness.run_case(case, mode="full", output_dir=str(out_dir))
        blob = next(out_dir.glob("*.atlas"))
        recon = codec.decompress(blob)
        np.save(out_dir / "recon.npy", recon.astype(np.float32))
        # Partial decode: stop before endmember mixing (the compressed-domain features).
        abund = codec.decompress(blob, abundance=True)
        np.save(out_dir / "abund.npy", abund.astype(np.float32))
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        ref = None if args.rank else golden.get(name, {}).get("psnr")
        delta = f" (golden {ref:.2f}, delta {metrics['psnr'] - ref:+.2f} dB)" if ref else ""
        print(f"=== {name}: {metrics['psnr']:.2f} dB @ {metrics['bpppb']:.4f} bpppb, "
              f"N={metrics['n_gaussians']}, {time.time() - t0:.0f}s{delta} ===", flush=True)


if __name__ == "__main__":
    main()
