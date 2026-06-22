import glob
import os
import shutil

from rich.console import Console
from torch.utils.cpp_extension import _get_build_directory, load

# Toolkit detection lives in a side-effect-free module so callers (ATLAS's
# `atlas doctor`, the model's import-time kernel check) can probe for nvcc without
# triggering the JIT build that importing *this* module performs. Re-exported here so
# `from atlas_gsplat.cuda._backend import cuda_toolkit_available` keeps working.
from ._detect import cuda_toolkit_available, cuda_toolkit_version  # noqa: F401

PATH = os.path.dirname(os.path.abspath(__file__))


name = "atlas_gsplat_cuda"
build_dir = _get_build_directory(name, verbose=False)
extra_include_paths = [os.path.join(PATH, "csrc/third_party/glm")]
extra_cflags = ["-O3"]
extra_cuda_cflags = ["-O3"]

_C = None
sources = list(glob.glob(os.path.join(PATH, "csrc/*.cu"))) + list(
    glob.glob(os.path.join(PATH, "csrc/*.cpp"))
)

try:
    # try to import the compiled module (via setup.py)
    from atlas_gsplat import csrc as _C
except ImportError:
    # if failed, try with JIT compilation
    if cuda_toolkit_available():
        # If JIT is interrupted it might leave a lock in the build directory.
        # We dont want it to exist in any case.
        try:
            os.remove(os.path.join(build_dir, "lock"))
        except OSError:
            pass

        if os.path.exists(os.path.join(build_dir, f"{name}.so")) or os.path.exists(
            os.path.join(build_dir, f"{name}.lib")
        ):
            # If the build exists, we assume the extension has been built
            # and we can load it.
            _C = load(
                name=name,
                sources=sources,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_include_paths=extra_include_paths,
            )
        else:
            # Build from scratch. Remove the build directory just to be safe: pytorch jit might stuck
            # if the build directory exists with a lock file in it.
            shutil.rmtree(build_dir)
            with Console().status(
                "[bold yellow]atlas-gsplat: Setting up CUDA (this may take a few minutes the first time)",
                spinner="bouncingBall",
            ):
                _C = load(
                    name=name,
                    sources=sources,
                    extra_cflags=extra_cflags,
                    extra_cuda_cflags=extra_cuda_cflags,
                    extra_include_paths=extra_include_paths,
                )
    else:
        Console().print(
            "[yellow]atlas-gsplat: No CUDA toolkit found. atlas-gsplat will be disabled.[/yellow]"
        )


__all__ = ["_C"]
