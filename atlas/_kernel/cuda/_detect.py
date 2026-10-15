"""Side-effect-free CUDA toolkit detection for atlas-gsplat.

Kept separate from ``_backend.py`` so callers (and ATLAS's ``atlas doctor``, and the
model's import-time kernel check) can probe for a toolkit *without* triggering the JIT
build that importing ``_backend`` performs.

Detection order, matching what the build then uses: an explicit ``CUDA_HOME`` /
``CUDA_PATH``, then ``nvcc`` already on ``PATH``, then a scan of standard install
locations (``/usr/local/cuda``, ``/usr/local/cuda-*``). No hardcoded CUDA versions.
"""
import glob
import json
import os
import re
import shutil
from subprocess import DEVNULL, STDOUT, call, check_output

_RELEASE_RE = re.compile(r"release (\d+\.\d+)")


def _has_nvcc(cuda_home):
    """True if ``<cuda_home>/bin/nvcc`` exists."""
    return bool(cuda_home) and os.path.isfile(os.path.join(cuda_home, "bin", "nvcc"))


def _version_key(path):
    """Sort key from a ``cuda-XX.Y`` path → ``(XX, Y)``; unparseable → ``(0, 0)``."""
    m = re.search(r"cuda-(\d+)(?:\.(\d+))?", os.path.basename(path.rstrip("/")))
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)


def discover_cuda_home():
    """Find a CUDA toolkit in standard install locations, or return ``None``.

    Prefers the ``/usr/local/cuda`` symlink (the machine default), then the
    highest-versioned ``/usr/local/cuda-XX.Y``. Returns ``None`` off Linux (the globs
    simply match nothing).
    """
    if _has_nvcc("/usr/local/cuda"):
        return "/usr/local/cuda"
    versioned = sorted(
        (p for p in glob.glob("/usr/local/cuda-*") if _has_nvcc(p)),
        key=_version_key,
        reverse=True,
    )
    return versioned[0] if versioned else None


def activate_cuda_home(cuda_home):
    """Prepend ``<cuda_home>/bin`` to ``PATH`` and export ``CUDA_HOME``.

    Idempotent. Ensures both ``nvcc`` and torch's extension build can find the toolkit.
    """
    bindir = os.path.join(cuda_home, "bin")
    if bindir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("CUDA_HOME", cuda_home)


def cuda_toolkit_available():
    """Whether ``nvcc`` is reachable; activates the toolkit on ``PATH`` when found.

    Order: ``CUDA_HOME``/``CUDA_PATH``, then ``nvcc`` on ``PATH``, then a scan of
    standard install locations. Safe to call repeatedly. No hardcoded versions.
    """
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if _has_nvcc(cuda_home):
        activate_cuda_home(cuda_home)
        return True
    try:
        call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
        return True
    except OSError:
        pass
    discovered = discover_cuda_home()
    if discovered:
        activate_cuda_home(discovered)
        return True
    return False


def cuda_toolkit_version():
    """Toolkit version string (e.g. ``'12.8'``), or ``None`` if it can't be determined.

    Reads ``version.json`` / ``version.txt`` next to the toolkit, then falls back to
    parsing ``nvcc --version``. Returns ``None`` rather than raising when nvcc is absent.
    """
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return None
    cuda_home = os.path.join(os.path.dirname(nvcc), "..")
    version_txt = os.path.join(cuda_home, "version.txt")
    version_json = os.path.join(cuda_home, "version.json")
    if os.path.exists(version_txt):
        with open(version_txt) as f:
            return f.read().strip().split()[-1]
    if os.path.exists(version_json):
        with open(version_json) as f:
            return json.load(f)["cuda"]["version"]
    try:
        out = check_output([nvcc, "--version"], stderr=STDOUT, text=True)
    except Exception:
        return None
    m = _RELEASE_RE.search(out)
    return m.group(1) if m else None
