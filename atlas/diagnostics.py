"""Environment diagnostics for ATLAS, behind the ``atlas doctor`` command.

ATLAS compiles its CUDA kernel on first use. This reports what that needs: torch's CUDA
build, a CUDA toolkit (``nvcc``), and the ``ninja`` build tool, plus whether a prebuilt
kernel is present (which needs neither). It runs with no GPU and without torch, so a
fresh user can see what is missing before the first (slow) compile. Toolkit detection
mirrors ``atlas._kernel.cuda._detect`` but stays torch-free here.
"""
from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import sys
from subprocess import STDOUT, check_output

_RELEASE_RE = re.compile(r"release (\d+\.\d+)")
IS_WINDOWS = os.name == "nt"
NVCC = "nvcc.exe" if IS_WINDOWS else "nvcc"
KERNEL_EXT = ".pyd" if IS_WINDOWS else ".so"
_MARK = {"ok": "[ OK ]", "warn": "[WARN]", "error": "[FAIL]", "info": "[INFO]"}


# --- CUDA toolkit detection (torch-free) ------------------------------------------

def _has_nvcc(cuda_home: str | None) -> bool:
    return bool(cuda_home) and os.path.isfile(os.path.join(cuda_home, "bin", NVCC))


def _version_key(path: str) -> tuple[int, int]:
    name = os.path.basename(path.rstrip("/\\"))
    m = re.search(r"(?:cuda-|^v)(\d+)(?:\.(\d+))?", name)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)


def find_nvcc() -> str | None:
    """Path to ``nvcc`` via CUDA_HOME/CUDA_PATH, then PATH, then ``/usr/local/cuda*`` (or the
    standard Windows install location)."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if _has_nvcc(cuda_home):
        return os.path.join(cuda_home, "bin", NVCC)
    on_path = shutil.which("nvcc")
    if on_path:
        return on_path
    if _has_nvcc("/usr/local/cuda"):
        return "/usr/local/cuda/bin/nvcc"
    patterns = ["/usr/local/cuda-*"]
    if IS_WINDOWS:
        root = os.environ.get("ProgramFiles", r"C:\Program Files")
        patterns.append(os.path.join(root, "NVIDIA GPU Computing Toolkit", "CUDA", "v*"))
    versioned = sorted(
        (p for pattern in patterns for p in glob.glob(pattern) if _has_nvcc(p)),
        key=_version_key,
        reverse=True,
    )
    return os.path.join(versioned[0], "bin", NVCC) if versioned else None


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
                "torch has no CUDA build (CPU-only torch). The ATLAS CUDA kernel needs "
                "a CUDA-enabled torch — reinstall it from a CUDA wheel index.")
    if kj is None:
        return ("error",
                f"No CUDA toolkit (nvcc) found, so the kernel cannot build. Install a "
                f"CUDA {built_cuda} toolkit to match torch (or set CUDA_HOME if one is "
                f"already installed in a non-standard location).")
    if tj == kj:
        return ("ok", f"torch CUDA {built_cuda} matches the toolkit ({toolkit_version}).")
    tag = cu_tag(toolkit_version) or "cuXYZ"
    # torch enforces a matching major only for install-time builds (setup.py). The
    # first-use compile goes through cpp_extension.load, which does not check, and a
    # 12.8 toolkit with a CUDA 13.0 torch compiled and ran correctly in testing.
    return ("warn",
            f"CUDA major mismatch: torch is built for {built_cuda} but the toolkit is "
            f"{toolkit_version}. The first-use compile usually still works; if it fails, "
            f"align them. Either:\n"
            f"      (a) match torch to your toolkit:\n"
            f"          pip install torch --index-url "
            f"https://download.pytorch.org/whl/{tag}\n"
            f"      (b) install a CUDA {tj}.x toolkit to match torch.")


# --- torch / kernel probes ---------------------------------------------------------

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


def find_ninja() -> str | None:
    """Path to the ``ninja`` build tool, which the first-use kernel compile needs: on PATH,
    or where the pip ``ninja`` package installs it. Mirrors
    ``atlas._kernel.cuda._detect.find_ninja`` (kept here so doctor stays torch-free)."""
    found = shutil.which("ninja")
    if found:
        return found
    try:
        import ninja
    except ImportError:
        return None
    return shutil.which("ninja", path=ninja.BIN_DIR)


def find_msvc() -> str | bool | None:
    """On Windows, the path to MSVC's ``cl``, which compiling the kernel needs, or ``None``.
    Off Windows there is no MSVC requirement, so this returns ``True``."""
    return shutil.which("cl") if IS_WINDOWS else True


def kernel_prebuilt(built_cuda: str | None = None) -> bool:
    """Whether a prebuilt kernel for torch's CUDA major version ships with this install
    (``atlas/_kernel/_C_cu<major>.so``). It runs without a CUDA toolkit or ninja. Mirrors
    ``atlas._kernel.cuda._detect.prebuilt_kernel_path``, kept here torch-free."""
    major = str(built_cuda or "").split(".")[0]
    if not major.isdigit():
        return False
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.isfile(os.path.join(here, "_kernel", f"_C_cu{major}{KERNEL_EXT}"))


def probe_kernel_build() -> tuple[str, str]:
    """Force the ATLAS CUDA kernel to load (compiling it if needed).

    Heavy: may compile for minutes on first run and needs torch + a GPU. Returns
    ``(level, message)``.
    """
    try:
        from ._kernel.cuda import _backend
    except Exception as e:
        return ("error", f"the ATLAS CUDA kernel failed to build or load: {e}")
    if getattr(_backend, "_C", None) is None:
        return ("error", "the ATLAS CUDA kernel did not load (no CUDA toolkit found).")
    return ("ok", "the ATLAS CUDA kernel compiled and loaded.")


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
        "ninja": find_ninja(),
        "msvc": find_msvc(),
        "kernel_prebuilt": kernel_prebuilt(built),
        "match_level": level,
        "match_message": message,
    }


def overall_ok(report: dict) -> bool:
    """True when the environment can run the kernel.

    A prebuilt kernel needs only a CUDA torch and a GPU. Otherwise the kernel compiles on
    first use, which needs a toolkit and ninja; a CUDA major mismatch is only a warning.
    """
    t = report["torch"]
    if not (t.get("installed") and t.get("cuda_available")):
        return False
    if report.get("kernel_prebuilt", False):
        return True
    return (bool(report.get("nvcc")) and bool(report.get("ninja"))
            and report.get("msvc", True) is not None
            and report["match_level"] != "error")


def install_hint(report: dict) -> list[str]:
    """Copy-pasteable command(s) for what *this* environment is missing: a CUDA build
    of torch and the ninja build tool. The toolkit is covered by ``match_message``."""
    t = report["torch"]
    built = t.get("built_cuda") if t.get("installed") else None
    hints: list[str] = []
    if built is None:
        hints.append("# install a CUDA build of torch, e.g. for CUDA 13.0:")
        hints.append("pip install torch --index-url https://download.pytorch.org/whl/cu130")
    if not report.get("ninja") and not report.get("kernel_prebuilt"):
        hints.append("pip install ninja")
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

    prebuilt = report.get("kernel_prebuilt")
    if report["nvcc"]:
        ver = report["toolkit_version"] or "unknown version"
        lines.append(f"{_MARK['ok']} CUDA toolkit {ver} at {report['nvcc']}")
    elif prebuilt:
        lines.append(f"{_MARK['info']} no CUDA toolkit found; none is needed with the "
                     f"prebuilt kernel")
    else:
        lines.append(f"{_MARK['error']} no CUDA toolkit (nvcc) found on "
                     f"CUDA_HOME / PATH / /usr/local/cuda*")

    if report.get("kernel_prebuilt"):
        lines.append(f"{_MARK['ok']} prebuilt CUDA kernel installed, so nothing needs compiling")
    elif report.get("ninja"):
        lines.append(f"{_MARK['ok']} ninja at {report['ninja']}; the kernel compiles on first use")
    else:
        lines.append(f"{_MARK['error']} ninja not found; the kernel compiles on first use "
                     f"and needs it")
    if report.get("msvc") is None and not report.get("kernel_prebuilt"):
        lines.append(f"{_MARK['error']} MSVC (cl.exe) not on PATH; compiling the kernel on "
                     f"Windows needs it. Run from the \"x64 Native Tools Command Prompt for VS\".")
    hints = install_hint(report)
    if hints:
        lines.append("       Install what is missing with:")
        lines += [f"       {h}" for h in hints]

    # A prebuilt kernel runs without the toolkit, so a toolkit that does not match torch
    # only matters when compiling from source: one note, without the remedies.
    if prebuilt:
        if report["nvcc"] and report["match_level"] != "ok":
            lines += ["", f"{_MARK['info']} the toolkit does not match torch's CUDA build; "
                          f"this only matters if you compile the kernel from source."]
        lines.append("")
    else:
        lines += ["", f"{_MARK[report['match_level']]} {report['match_message']}", ""]

    if overall_ok(report):
        lines.append("READY — your environment can run the ATLAS CUDA kernel.")
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
