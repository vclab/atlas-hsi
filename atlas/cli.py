"""Command-line interface for ATLAS.

Two subcommands:
  atlas init   — NMF endmember/abundance initialization for a dataset
  atlas train  — train an ATLAS representation (the compressed cube) for one image

The CLI exposes a stable, trimmed flag set. Techniques that the paper found do not
transfer to HSI are not exposed: content-aware filters (hard off), spatial delta coding
and progressive QAT (removed). Gradient-guided / entropy init is always on.
"""
import argparse

from . import data


def _add_train_args(p):
    p.add_argument("--input", "--dataset", dest="input", required=True,
                   help="Dataset name (urban|salinas|jasperridge|paviau) or a .mat/.npy cube path")
    p.add_argument("--iterations", type=int, default=8000, help="Training iterations")
    p.add_argument("--num_points", type=int, required=True,
                   help="Number of Gaussians (starting count when D3 is enabled)")
    p.add_argument("--max_num_points", type=int, default=None,
                   help="D3: max Gaussians to grow to (enables D3 when > num_points)")
    p.add_argument("--grow_stop_iter", type=int, default=5000,
                   help="D3: stop growing after this iter")
    p.add_argument("--qat_start", type=int, default=0,
                   help="Iteration to start two-stage QAT (0 = FP32, no quantization)")
    p.add_argument("--prune_interval", type=int, default=100,
                   help="Prune degenerate Gaussians every N iters during QAT (0 = disabled)")
    p.add_argument("--endmember_lr", type=float, default=0,
                   help="Learnable-endmember LR (0 = frozen; 1e-4 recommended)")
    p.add_argument("--sam_weight", type=float, default=0.0,
                   help="SAM loss weight (0 = off; 0.05 recommended)")
    p.add_argument("--rank", type=int, default=None,
                   help="NMF rank (default: 10 for JasperRidge, else 12)")
    p.add_argument("--data-dir", type=str, default=data.DEFAULT_DATA_DIR,
                   help="Directory holding raw dataset files")
    p.add_argument("--init-dir", type=str, default=data.DEFAULT_INIT_DIR,
                   help="Directory holding NMF .npy init files")
    p.add_argument("--output-dir", type=str, default="atlas_runs",
                   help="Directory to write checkpoints and logs")
    # Advanced (quantization bit-widths; only matter once QAT is enabled).
    p.add_argument("--abundance_bits", type=int, default=6, help="[advanced] abundance LSQ+ bits")
    p.add_argument("--cholesky_bits", type=int, default=10, help="[advanced] cholesky LSQ+ bits")
    p.add_argument("--lsq-grad-scale", dest="lsq_grad_scale", action="store_true",
                   help="[experimental] enable LSQ+ gradient scaling for the quantizer "
                        "scale/offset params (default off = validated behavior)")


def cmd_train(args):
    from .train import train_nd  # imported here so `atlas init` doesn't require CUDA/atlas_gsplat

    name = str(args.input)
    key = data.init_key(name)  # matches what `atlas init` wrote (basename stem for paths)
    rank = args.rank if args.rank is not None else data.rank_for(key)
    max_num_points = args.max_num_points if args.max_num_points is not None else args.num_points
    d3_enabled = max_num_points > args.num_points

    grow_stop_iter = args.grow_stop_iter
    if args.qat_start > 0 and d3_enabled and grow_stop_iter >= args.qat_start:
        grow_stop_iter = args.qat_start - 1
        print(f"WARNING: grow_stop_iter >= qat_start; D3 must finish before QAT. "
              f"Setting grow_stop_iter = {grow_stop_iter}")

    cube = data.load_cube(name, data_dir=args.data_dir)
    endmember = data.load_endmember(key, rank, init_dir=args.init_dir)
    A0 = data.load_abundance(key, rank, init_dir=args.init_dir)  # entropy/gradient init (always on)
    gt = data.to_tensor(cube)

    print("\n=== ATLAS training ===")
    print(f"Input: {name}  ({cube.shape[0]}x{cube.shape[1]}x{cube.shape[2]}, rank {rank})")
    print(f"Gaussians: {args.num_points}" + (f" -> {max_num_points} (D3)" if d3_enabled else ""))
    if args.qat_start > 0:
        print(f"QAT @ {args.qat_start}: abundance={args.abundance_bits}-bit, "
              f"cholesky={args.cholesky_bits}-bit")
    if A0 is None:
        print("Entropy-guided init: no abundance .npy found, using random init")
    print()

    train_nd(
        gt, endmember=endmember, image_name=key, iterations=args.iterations,
        num_points=args.num_points, max_num_points=max_num_points,
        grow_interval=1000, grow_stop_iter=grow_stop_iter,
        qat_start=args.qat_start,
        prune_interval=args.prune_interval, endmember_lr=args.endmember_lr,
        sam_weight=args.sam_weight, A0=A0,
        abundance_bits=args.abundance_bits, cholesky_bits=args.cholesky_bits,
        lsq_grad_scale=args.lsq_grad_scale,
        output_dir=args.output_dir,
    )


def cmd_doctor(args):
    from . import diagnostics
    raise SystemExit(diagnostics.run(build=args.build))


def cmd_init(args):
    import numpy as np

    from .endmember import nmf_initialization

    cube = data.load_cube(args.input, data_dir=args.data_dir)
    I = np.transpose(cube, (2, 0, 1)).reshape(cube.shape[2], -1)  # (C, H*W)
    key = data.init_key(args.input)
    rank = args.rank if args.rank is not None else data.rank_for(key)
    nmf_initialization(I, rank, key, init_dir=args.init_dir)


def cmd_decompress(args):
    import numpy as np

    from . import codec

    meta, arrays = codec.read_bitstream(args.input)
    bpppb = codec.payload_bpppb(meta, arrays)
    cube = codec.reconstruct(meta, arrays)  # (H, W, C); needs CUDA
    np.save(args.output, cube)
    print(f"Decompressed {args.input} -> {args.output}  "
          f"shape={tuple(cube.shape)}  payload bpppb={bpppb:.4f}")
    if args.reference:
        ref = data.load_cube(args.reference, data_dir=args.data_dir)
        ref = np.clip(ref, 0, 1).astype(np.float32)
        mse = ((ref - cube) ** 2).mean(axis=(0, 1))  # per-band MSE
        mpsnr = float(np.mean(10 * np.log10(1.0 / np.maximum(mse, 1e-12))))
        print(f"Reference {args.reference}: mean per-band PSNR = {mpsnr:.4f} dB")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="atlas",
        description="ATLAS — Adaptive Transfer of Low-rank Abundance Splatting (HSI compression)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="NMF endmember/abundance initialization")
    p_init.add_argument("--input", "--dataset", dest="input", required=True,
                        help="Dataset name or path to a .mat/.npy cube")
    p_init.add_argument("--rank", type=int, default=None, help="NMF rank (default per dataset)")
    p_init.add_argument("--data-dir", type=str, default=data.DEFAULT_DATA_DIR)
    p_init.add_argument("--init-dir", type=str, default=data.DEFAULT_INIT_DIR)
    p_init.set_defaults(func=cmd_init)

    p_train = sub.add_parser("train", help="Train an ATLAS representation for one cube")
    _add_train_args(p_train)
    p_train.set_defaults(func=cmd_train)

    p_doctor = sub.add_parser(
        "doctor", help="Diagnose the torch / CUDA toolkit / atlas-gsplat environment")
    p_doctor.add_argument(
        "--build", action="store_true",
        help="Also force-build/load the atlas-gsplat CUDA kernel (slow; needs a GPU)")
    p_doctor.set_defaults(func=cmd_doctor)

    p_dec = sub.add_parser(
        "decompress", help="Reconstruct a cube from an ATLAS bitstream (.atlas)")
    p_dec.add_argument("input", help="Path to a .atlas bitstream file")
    p_dec.add_argument("-o", "--output", required=True, help="Output cube .npy path")
    p_dec.add_argument("--reference", default=None,
                       help="Optional original cube (name or path) to report reconstruction PSNR")
    p_dec.add_argument("--data-dir", type=str, default=data.DEFAULT_DATA_DIR)
    p_dec.set_defaults(func=cmd_decompress)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
