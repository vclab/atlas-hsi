"""Build the prebuilt ATLAS CUDA kernel for the installed torch and CUDA toolkit.

Writes ``atlas/_kernel/_C_cu<major>.so`` (``.pyd`` on Windows). Release builds run this
once per CUDA major, against the oldest supported torch (2.10): the kernel uses only the
LibTorch stable ABI, so that one build loads on every later torch. Compiling needs the
CUDA toolkit and ninja (and MSVC on Windows) but no GPU. On Linux, run it through
``tools/build_kernel_linux.sh`` so the result works on older distributions too.

    python tools/build_kernel.py [--verbose]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "atlas" / "_kernel" / "cuda" / "csrc"
OUT = REPO / "atlas" / "_kernel"

# GPU generations (compute capabilities) compiled for each CUDA major, plus PTX for the
# newest so later GPUs can still run it. CUDA 13 dropped support below 7.5.
ARCHS = {
    "12": "7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX",
    "13": "7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX",
}


def toolkit_major(cuda_home):
    """Major version of the toolkit torch will compile with, from ``nvcc --version``."""
    nvcc = os.path.join(cuda_home, "bin", "nvcc.exe" if os.name == "nt" else "nvcc")
    out = subprocess.check_output([nvcc, "--version"], text=True)
    return re.search(r"release (\d+)\.\d+", out).group(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verbose", action="store_true", help="show the compiler commands")
    args = ap.parse_args()

    import torch
    from torch.utils import cpp_extension

    torch_major = (torch.version.cuda or "").split(".")[0]
    if not torch_major:
        sys.exit("torch has no CUDA build; install a CUDA build of torch first")
    if not cpp_extension.CUDA_HOME:
        sys.exit("no CUDA toolkit found; set CUDA_HOME")
    major = toolkit_major(cpp_extension.CUDA_HOME)
    if major != torch_major:
        sys.exit(f"the toolkit is CUDA {major} but torch is built for CUDA {torch_major}; "
                 f"build each kernel with a matching pair")
    if major not in ARCHS:
        sys.exit(f"no GPU target list for CUDA {major}; add one to ARCHS")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", ARCHS[major])

    name = f"_C_cu{major}"
    ext = ".pyd" if os.name == "nt" else ".so"
    build_dir = Path(tempfile.mkdtemp(prefix="atlas_kernel_build_"))
    built = build_dir / f"{name}{ext}"
    try:
        # Builds, then loads the library to check it registers; loading needs no GPU.
        cpp_extension.load(
            name=name,
            sources=sorted(str(p) for p in SRC.glob("*.cu")),
            extra_cuda_cflags=["-O3"],
            build_directory=str(build_dir),
            is_python_module=False,
            verbose=args.verbose,
        )
    finally:
        if built.exists():
            shutil.copy2(built, OUT / built.name)
        shutil.rmtree(build_dir, ignore_errors=True)
    if not (OUT / built.name).exists():
        sys.exit("the build produced no library")

    ops = [op for op in ("project_gaussians_2d_forward", "map_gaussian_to_intersects",
                         "nd_rasterize_sum_backward_general") if hasattr(torch.ops.atlas, op)]
    size_mb = (OUT / built.name).stat().st_size / 1e6
    print(f"built {OUT / built.name} ({size_mb:.1f} MB): torch {torch.__version__}, "
          f"CUDA {major}, GPU targets {os.environ['TORCH_CUDA_ARCH_LIST']}, "
          f"{len(ops)}/3 sample ops registered")


if __name__ == "__main__":
    main()
