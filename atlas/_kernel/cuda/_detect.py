"""Side-effect-free CUDA toolkit detection for the ATLAS kernel.

Kept separate from ``_backend.py`` so callers (and ATLAS's ``atlas doctor``, and the
model's import-time kernel check) can probe for a toolkit *without* triggering the JIT
build that importing ``_backend`` performs.

Detection order, matching what the build then uses: an explicit ``CUDA_HOME`` /
``CUDA_PATH``, then ``nvcc`` already on ``PATH``, then a scan of standard install
locations (``/usr/local/cuda``, ``/usr/local/cuda-*``, and on Windows
``C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*``). No hardcoded CUDA versions.
"""
import glob
import json
import os
import re
import shutil
from subprocess import DEVNULL, STDOUT, call, check_output

_RELEASE_RE = re.compile(r"release (\d+\.\d+)")

IS_WINDOWS = os.name == "nt"
NVCC = "nvcc.exe" if IS_WINDOWS else "nvcc"
# The suffix torch gives compiled extension libraries on this platform.
KERNEL_EXT = ".pyd" if IS_WINDOWS else ".so"


def _has_nvcc(cuda_home):
    """True if ``<cuda_home>/bin/nvcc`` (``nvcc.exe`` on Windows) exists."""
    return bool(cuda_home) and os.path.isfile(os.path.join(cuda_home, "bin", NVCC))


def _version_key(path):
    """Sort key from a ``cuda-XX.Y`` (Linux) or ``vXX.Y`` (Windows) path → ``(XX, Y)``;
    unparseable → ``(0, 0)``."""
    name = os.path.basename(path.rstrip("/\\"))
    m = re.search(r"(?:cuda-|^v)(\d+)(?:\.(\d+))?", name)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)


def discover_cuda_home():
    """Find a CUDA toolkit in standard install locations, or return ``None``.

    Prefers the ``/usr/local/cuda`` symlink (the machine default), then the
    highest-versioned ``/usr/local/cuda-XX.Y`` or, on Windows, the highest-versioned
    toolkit under ``Program Files``.
    """
    if _has_nvcc("/usr/local/cuda"):
        return "/usr/local/cuda"
    patterns = ["/usr/local/cuda-*"]
    if IS_WINDOWS:
        root = os.environ.get("ProgramFiles", r"C:\Program Files")
        patterns.append(os.path.join(root, "NVIDIA GPU Computing Toolkit", "CUDA", "v*"))
    versioned = sorted(
        (p for pattern in patterns for p in glob.glob(pattern) if _has_nvcc(p)),
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


def find_ninja():
    """Path to the ``ninja`` build tool, or ``None``.

    Looks on PATH first, then in the directory where the pip ``ninja`` package installs
    its binary (the environment's ``bin``), which is not on PATH when Python runs without
    activating the environment, as cluster jobs and notebook kernels often do.
    """
    found = shutil.which("ninja")
    if found:
        return found
    try:
        import ninja  # the pip package that ships the binary
    except ImportError:
        return None
    return shutil.which("ninja", path=ninja.BIN_DIR)


# Prebuilt kernels ship in atlas/_kernel/ as _C_cu<CUDA major>.so (.pyd on Windows), one
# per CUDA major version.
KERNEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prebuilt_kernel_path(cuda_version):
    """Path of the prebuilt kernel matching a torch CUDA version such as ``'13.0'``, or
    ``None`` when none ships with this install (torch-free: the caller passes
    ``torch.version.cuda``)."""
    major = str(cuda_version or "").split(".")[0]
    if not major.isdigit():
        return None
    path = os.path.join(KERNEL_DIR, f"_C_cu{major}{KERNEL_EXT}")
    return path if os.path.isfile(path) else None


def msvc_available():
    """Whether MSVC's ``cl`` is on PATH. Compiling the kernel on Windows needs it (use the
    "x64 Native Tools Command Prompt for VS"); a prebuilt kernel does not. Always True off
    Windows."""
    return (not IS_WINDOWS) or shutil.which("cl") is not None
