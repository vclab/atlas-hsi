"""Load the ATLAS CUDA kernel, which registers ``torch.ops.atlas.*``.

Uses the prebuilt library for this torch's CUDA major version when one ships with the
package. Otherwise it compiles the same stable-ABI sources on first use (cached by torch
under ``~/.cache/torch_extensions``) and loads that. Importing this module does the load.
"""
import glob
import os
import shutil

import torch
from rich.console import Console
from torch.utils.cpp_extension import _get_build_directory, load

# Detection lives in a side-effect-free module so callers (`atlas doctor`, the model's
# import-time check) can probe without triggering the load that importing *this* does.
from ._detect import (  # noqa: F401
    cuda_toolkit_available,
    cuda_toolkit_version,
    find_ninja,
    prebuilt_kernel_path,
)

PATH = os.path.dirname(os.path.abspath(__file__))
name = "atlas_kernel_cuda"
sources = sorted(glob.glob(os.path.join(PATH, "csrc", "*.cu")))
extra_cflags = ["-O3"]
extra_cuda_cflags = ["-O3"]


def _compile_and_load():
    """Compile the sources (first use only; torch caches the build) and load the library."""
    # torch runs `ninja` from PATH; make the pip-installed one visible even when the
    # environment was not activated.
    found = find_ninja()
    if found and not shutil.which("ninja"):
        os.environ["PATH"] = os.path.dirname(found) + os.pathsep + os.environ.get("PATH", "")

    build_dir = _get_build_directory(name, verbose=False)
    library = os.path.join(build_dir, f"{name}.so")
    # An interrupted build can leave a lock that makes the next build hang.
    try:
        os.remove(os.path.join(build_dir, "lock"))
    except OSError:
        pass

    def build():
        load(name=name, sources=sources, extra_cflags=extra_cflags,
             extra_cuda_cflags=extra_cuda_cflags, is_python_module=False)

    if os.path.exists(library):
        build()  # cached; torch rebuilds only if the sources or flags changed
    else:
        shutil.rmtree(build_dir, ignore_errors=True)
        with Console().status(
            "[bold yellow]atlas: compiling the CUDA kernel (first run only, about a minute)",
            spinner="bouncingBall",
        ):
            build()
    return library


_C = None  # path of the loaded library, or None when the kernel is unavailable
_prebuilt = prebuilt_kernel_path(torch.version.cuda)
if _prebuilt:
    torch.ops.load_library(_prebuilt)
    _C = _prebuilt
elif cuda_toolkit_available():
    _C = _compile_and_load()
else:
    Console().print(
        "[yellow]atlas: no prebuilt kernel for this torch and no CUDA toolkit (nvcc) "
        "to compile one.[/yellow]"
    )


__all__ = ["_C"]
