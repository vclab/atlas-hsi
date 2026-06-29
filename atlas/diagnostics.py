"""Environment diagnostics for ATLAS — powers the ``atlas doctor`` command.

Reports the pieces the CUDA kernel build depends on (torch's CUDA build, a CUDA
toolkit / ``nvcc``, whether ``atlas-gsplat`` is installed) and flags the most common
fresh-install trap: torch and the toolkit built for *different CUDA major versions*.

Runs with no GPU, and even without torch or atlas-gsplat installed, so a fresh user can
diagnose what's missing before attempting a (slow) build. Toolkit detection here mirrors
``atlas_gsplat.cuda._detect`` but is kept local so ``atlas doctor`` needs neither torch
nor atlas-gsplat just to find ``nvcc``.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import platform
import re
import shutil
import sys
from subprocess import STDOUT, check_output

_RELEASE_RE = re.compile(r"release (\d+\.\d+)")
_MARK = {"ok": "[ OK ]", "warn": "[WARN]", "error": "[FAIL]", "info": "[INFO]"}
GSPLAT_REPO = "https://github.com/jacobh3ll0/atlas-gsplat"


# --- CUDA toolkit detection (torch-free, atlas-gsplat-free) ------------------------

def _has_nvcc(cuda_home: str | None) -> bool:
    return bool(cuda_home) and os.path.isfile(os.path.join(cuda_home, "bin", "nvcc"))


def _version_key(path: str) -> tuple[int, int]:
    m = re.search(r"cuda-(\d+)(?:\.(\d+))?", os.path.basename(path.rstrip("/")))
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)


def find_nvcc() -> str | None:
    """Path to ``nvcc`` via CUDA_HOME/CUDA_PATH, then PATH, then ``/usr/local/cuda*``."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if _has_nvcc(cuda_home):
        return os.path.join(cuda_home, "bin", "nvcc")
    on_path = shutil.which("nvcc")
    if on_path:
        return on_path
    if _has_nvcc("/usr/local/cuda"):
        return "/usr/local/cuda/bin/nvcc"
    versioned = sorted(
        (p for p in glob.glob("/usr/local/cuda-*") if _has_nvcc(p)),
        key=_version_key,
        reverse=True,
    )
    return os.path.join(versioned[0], "bin", "nvcc") if versioned else None


def _parse_release(text: str) -> str | None:
    m = _RELEASE_RE.search(text or "")
    return m.group(1) if m else None


def nvcc_version(nvcc: str | None = None) -> str | None:
    """Toolkit version like ``'12.8'`` parsed from ``nvcc --version``, or ``None``."""
    nvcc = nvcc or find_nvcc()
    if not nvcc:
        return None
    try:
        out = check_output([nvcc, "--version"], stderr=STDOUT, text=True)
    except Exception:
        return None
    return _parse_release(out)


# --- version comparison ------------------------------------------------------------

def cuda_major(version: str | None) -> int | None:
    """Major CUDA version as int: ``'12.8'`` → 12, ``'13'`` → 13, junk/None → None."""
    if not version:
        return None
    m = re.match(r"\s*(\d+)", str(version))
    return int(m.group(1)) if m else None


def cu_tag(version: str | None) -> str | None:
    """PyTorch wheel CUDA tag: ``'12.8'`` → ``'cu128'``, ``'13.0'`` → ``'cu130'``."""
    if not version:
        return None
    digits = "".join(re.findall(r"\d+", str(version)))
    return f"cu{digits}" if digits else None


def match_status(built_cuda: str | None, toolkit_version: str | None) -> tuple[str, str]:
    """Compare torch's CUDA build to the toolkit. Returns ``(level, message)`` with
    ``level`` in ``{'ok', 'warn', 'error'}``. Only the *major* version must match."""
    tj, kj = cuda_major(built_cuda), cuda_major(toolkit_version)
    if tj is None:
        return ("warn",
                "torch has no CUDA build (CPU-only torch). The atlas-gsplat kernel needs "
                "a CUDA-enabled torch — reinstall it from a CUDA wheel index.")
    if kj is None:
        return ("error",
                f"No CUDA toolkit (nvcc) found, so the kernel cannot build. Install a "
                f"CUDA {built_cuda} toolkit to match torch (or set CUDA_HOME if one is "
                f"already installed in a non-standard location).")
    if tj == kj:
        return ("ok", f"torch CUDA {built_cuda} matches the toolkit ({toolkit_version}).")
    tag = cu_tag(toolkit_version) or "cuXYZ"
    return ("error",
            f"CUDA major mismatch: torch is built for {built_cuda} but the toolkit is "
            f"{toolkit_version}. Building the kernel will fail. Align them. Either:\n"
            f"      (a) match torch to your toolkit:\n"
            f"          pip install torch --index-url "
            f"https://download.pytorch.org/whl/{tag}\n"
            f"      (b) install a CUDA {tj}.x toolkit to match torch.")


# --- torch / atlas-gsplat probes ---------------------------------------------------

def torch_report() -> dict:
    """torch version, the CUDA it was built for, and runtime availability. torch-optional."""
    try:
        import torch
    except Exception as e:
        return {"installed": False, "error": str(e)}
    info = {
        "installed": True,
        "version": torch.__version__,
        "built_cuda": torch.version.cuda,           # e.g. '12.8', or None for a CPU build
        "cuda_available": bool(torch.cuda.is_available()),
        "device": None,
    }
    if info["cuda_available"]:
        try:
            info["device"] = torch.cuda.get_device_name(0)
        except Exception:
            info["device"] = None
    return info


def gsplat_installed() -> bool:
    """Whether ``atlas_gsplat`` is importable — via ``find_spec``, so no JIT build fires."""
    try:
        return importlib.util.find_spec("atlas_gsplat") is not None
    except Exception:
        return False


def kernel_prebuilt() -> bool:
    """Whether a prebuilt ``atlas_gsplat.csrc`` C extension is importable. A prebuilt
    extension runs without any CUDA toolkit, so the torch/toolkit build-match is moot
    (this mirrors the check in ``atlas.model``)."""
    try:
        return importlib.util.find_spec("atlas_gsplat.csrc") is not None
    except Exception:
        return False


def probe_kernel_build() -> tuple[str, str]:
    """Force the atlas-gsplat CUDA extension to load (compiling it if needed).

    Heavy: may compile for minutes on first run and needs torch + a GPU. Returns
    ``(level, message)``.
    """
    try:
        from atlas_gsplat.cuda import _backend
    except Exception as e:
        return ("error", f"atlas-gsplat kernel failed to build/load: {e}")
    if getattr(_backend, "_C", None) is None:
        return ("error", "atlas-gsplat kernel did not load (no CUDA toolkit, or build disabled).")
    return ("ok", "atlas-gsplat CUDA extension compiled and loaded.")


# --- assembly + reporting ----------------------------------------------------------

def collect() -> dict:
    """Gather the full diagnostic report as a plain dict."""
    nvcc = find_nvcc()
    toolkit_version = nvcc_version(nvcc)
    torch_info = torch_report()
    built = torch_info.get("built_cuda") if torch_info.get("installed") else None
    level, message = match_status(built, toolkit_version)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch_info,
        "nvcc": nvcc,
        "toolkit_version": toolkit_version,
        "gsplat_installed": gsplat_installed(),
        "kernel_prebuilt": kernel_prebuilt(),
        "match_level": level,
        "match_message": message,
    }


def overall_ok(report: dict) -> bool:
    """True when the environment can run the kernel.

    A prebuilt extension runs without a toolkit; otherwise torch and the toolkit must
    share a CUDA major version so the kernel can be built on first use.
    """
    t = report["torch"]
    if not (t.get("installed") and t.get("cuda_available") and report["gsplat_installed"]):
        return False
    return report.get("kernel_prebuilt", False) or report["match_level"] == "ok"


def install_hint(report: dict) -> list[str]:
    """Copy-pasteable command(s) to install the kernel for *this* environment.

    Leads with a CUDA build of torch when one is missing (the kernel compiles against
    torch's CUDA), then the kernel itself with ``--no-build-isolation`` so the source
    build can see that torch. The toolkit prerequisite is covered by ``match_message``.
    """
    t = report["torch"]
    built = t.get("built_cuda") if t.get("installed") else None
    hints: list[str] = []
    if built is None:
        hints.append("# install a CUDA build of torch first, e.g. for CUDA 12.4:")
        hints.append("pip install torch --index-url https://download.pytorch.org/whl/cu124")
    hints.append(f'pip install --no-build-isolation "git+{GSPLAT_REPO}"')
    return hints


def format_report(report: dict) -> str:
    """Render a report dict as human-readable text."""
    lines = ["ATLAS environment check", "=" * 23, ""]

    t = report["torch"]
    if t.get("installed"):
        dev = t["device"] or ("available" if t["cuda_available"] else "no CUDA device")
        mark = _MARK["ok"] if t["cuda_available"] else _MARK["warn"]
        lines.append(f"{mark} torch {t['version']} "
                     f"(built for CUDA {t['built_cuda'] or 'none / CPU'}); runtime GPU: {dev}")
    else:
        lines.append(f"{_MARK['error']} torch not importable: {t.get('error', '')}")

    if report["nvcc"]:
        ver = report["toolkit_version"] or "unknown version"
        lines.append(f"{_MARK['ok']} CUDA toolkit {ver} at {report['nvcc']}")
    else:
        lines.append(f"{_MARK['error']} no CUDA toolkit (nvcc) found on "
                     f"CUDA_HOME / PATH / /usr/local/cuda*")

    if report["gsplat_installed"]:
        extra = " (prebuilt kernel present)" if report.get("kernel_prebuilt") else ""
        lines.append(f"{_MARK['ok']} atlas-gsplat installed{extra}")
    else:
        lines.append(f"{_MARK['error']} atlas-gsplat NOT installed; install it with:")
        lines += [f"       {h}" for h in install_hint(report)]

    # A prebuilt kernel runs without the toolkit, so a build-time CUDA mismatch is not
    # fatal — demote it to an informational note instead of a failure.
    if report.get("kernel_prebuilt") and report["match_level"] != "ok":
        lines += ["", f"{_MARK['info']} {report['match_message']}",
                  f"{_MARK['ok']} a prebuilt atlas-gsplat kernel is installed, so the "
                  f"toolkit match above only matters if you rebuild from source.", ""]
    else:
        lines += ["", f"{_MARK[report['match_level']]} {report['match_message']}", ""]

    if overall_ok(report):
        lines.append("READY — your environment can run the atlas-gsplat kernel.")
    else:
        lines.append("NOT READY — resolve the items above, then re-run `atlas doctor`.")

    lines += ["", f"python {report['python']}  |  {report['platform']}"]
    return "\n".join(lines)


def run(build: bool = False) -> int:
    """Print the diagnostic report. Returns a shell exit code (0 = ready)."""
    report = collect()
    print(format_report(report))
    ok = overall_ok(report)
    if build:
        level, message = probe_kernel_build()
        print(f"\n{_MARK[level]} {message}")
        ok = ok and level == "ok"
    return 0 if ok else 1
