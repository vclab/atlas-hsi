import glob
import os
import shutil

from rich.console import Console
from torch.utils.cpp_extension import _get_build_directory, load

# Toolkit detection lives in a side-effect-free module so callers (ATLAS's
# `atlas doctor`, the model's import-time kernel check) can probe for nvcc without
# triggering the JIT build that importing *this* module performs. Re-exported here so
# `from atlas._kernel.cuda._backend import cuda_toolkit_available` keeps working.
from ._detect import cuda_toolkit_available, cuda_toolkit_version, find_ninja  # noqa: F401

PATH = os.path.dirname(os.path.abspath(__file__))


name = "atlas_kernel_cuda"
build_dir = _get_build_directory(name, verbose=False)
extra_cflags = ["-O3"]
extra_cuda_cflags = ["-O3"]

_C = None

# torch's first-use compile runs `ninja` from PATH; make the pip-installed one visible
# even when the environment was not activated.
_ninja = find_ninja()
if _ninja and not shutil.which("ninja"):
    os.environ["PATH"] = os.path.dirname(_ninja) + os.pathsep + os.environ.get("PATH", "")

sources = list(glob.glob(os.path.join(PATH, "csrc/*.cu"))) + list(
    glob.glob(os.path.join(PATH, "csrc/*.cpp"))
)

try:
    # try to import the compiled module (via setup.py)
    from .. import csrc as _C
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
            )
        else:
            # Build from scratch. Remove the build directory just to be safe: pytorch jit might stuck
            # if the build directory exists with a lock file in it.
            shutil.rmtree(build_dir)
            with Console().status(
                "[bold yellow]atlas: compiling the CUDA kernel (first run only, about a minute)",
                spinner="bouncingBall",
            ):
                _C = load(
                    name=name,
                    sources=sources,
                    extra_cflags=extra_cflags,
                    extra_cuda_cflags=extra_cuda_cflags,
                )
    else:
        Console().print(
            "[yellow]atlas: no CUDA toolkit (nvcc) found, so the CUDA kernel cannot be compiled.[/yellow]"
        )


__all__ = ["_C"]
