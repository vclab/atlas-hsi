"""Assemble the release files in dist/ from prebuilt kernels.

Expects kernels from tools/build_kernel.py in kernels/linux/ (``_C_cu*.so``) and
kernels/windows/ (``_C_cu*.pyd``); a missing platform is skipped. Produces:

- the sdist (sources only; compiles on first use),
- ``py3-none-any`` wheel (no kernels; compiles on first use, for any other platform),
- ``py3-none-manylinux_2_28_x86_64`` wheel with the Linux kernels,
- ``py3-none-win_amd64`` wheel with the Windows kernels.

pip picks the most specific wheel, so Linux x86_64 and Windows users get the prebuilt
kernels and everyone else falls back to compiling.

    python tools/assemble_wheels.py
"""
import glob
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KERNEL_DIR = REPO / "atlas" / "_kernel"
DIST = REPO / "dist"
PLATFORMS = [  # (kernels subfolder, file pattern, wheel platform tag)
    ("linux", "_C_cu*.so", "manylinux_2_28_x86_64"),
    ("windows", "_C_cu*.pyd", "win_amd64"),
]


def run(*cmd):
    subprocess.run(cmd, cwd=REPO, check=True)


def clear_kernels():
    for pattern in ("_C_cu*.so", "_C_cu*.pyd"):
        for f in KERNEL_DIR.glob(pattern):
            f.unlink()
    shutil.rmtree(REPO / "build", ignore_errors=True)  # setuptools reuses stale files


def main():
    shutil.rmtree(DIST, ignore_errors=True)
    clear_kernels()
    run(sys.executable, "-m", "build", "--sdist", "--wheel", "--outdir", str(DIST))

    for folder, pattern, tag in PLATFORMS:
        kernels = sorted(glob.glob(str(REPO / "kernels" / folder / pattern)))
        if not kernels:
            print(f"no {folder} kernels in kernels/{folder}/; skipping the {tag} wheel")
            continue
        clear_kernels()
        for k in kernels:
            shutil.copy2(k, KERNEL_DIR)
        staging = REPO / "build" / f"wheel-{folder}"
        run(sys.executable, "-m", "build", "--wheel", "--outdir", str(staging))
        built = next(staging.glob("*-py3-none-any.whl"))
        run(sys.executable, "-m", "wheel", "tags", "--platform-tag", tag, "--remove", str(built))
        tagged = next(staging.glob(f"*-{tag}.whl"))
        shutil.move(str(tagged), DIST / tagged.name)
    clear_kernels()

    for f in sorted(DIST.iterdir()):
        kernels = []
        if f.suffix == ".whl":
            with zipfile.ZipFile(f) as z:
                kernels = [Path(n).name for n in z.namelist() if Path(n).name.startswith("_C_cu")]
        print(f"{f.name:60s} {f.stat().st_size / 1e6:6.1f} MB  {' '.join(kernels)}")


if __name__ == "__main__":
    main()
