"""Tests for `atlas doctor` diagnostics — the version/match logic and toolkit scan.

All torch-free and GPU-free so they run anywhere (incl. CI).
"""
from atlas import diagnostics as dx


def test_cuda_major():
    assert dx.cuda_major("12.8") == 12
    assert dx.cuda_major("13.0") == 13
    assert dx.cuda_major("13") == 13
    assert dx.cuda_major(None) is None
    assert dx.cuda_major("") is None
    assert dx.cuda_major("cuda-12") is None  # no leading digit


def test_cu_tag():
    assert dx.cu_tag("12.8") == "cu128"
    assert dx.cu_tag("13.0") == "cu130"
    assert dx.cu_tag(None) is None
    assert dx.cu_tag("") is None


def test_match_ok_same_major_minor_differs():
    # Only the major version must match — 12.1 vs 12.8 is fine.
    level, _ = dx.match_status("12.1", "12.8")
    assert level == "ok"
    level, _ = dx.match_status("12.8", "12.8")
    assert level == "ok"


def test_match_major_mismatch_recommends_matching_tag():
    # Only a warning: the first-use compile does not enforce a matching major.
    level, msg = dx.match_status("13.0", "12.8")
    assert level == "warn"
    assert "mismatch" in msg.lower()
    assert "cu128" in msg  # recommend torch built for the installed toolkit


def test_match_no_toolkit_is_error():
    level, msg = dx.match_status("12.8", None)
    assert level == "error"
    assert "no cuda toolkit" in msg.lower()


def test_match_cpu_torch_is_warn():
    level, msg = dx.match_status(None, "12.8")
    assert level == "warn"
    assert "cpu" in msg.lower()


def test_parse_release():
    out = "Cuda compilation tools, release 12.8, V12.8.61"
    assert dx._parse_release(out) == "12.8"
    assert dx._parse_release("nothing here") is None
    assert dx._parse_release("") is None


def test_version_key_orders_by_version():
    assert dx._version_key("/usr/local/cuda-12.8") == (12, 8)
    assert dx._version_key("/usr/local/cuda-11") == (11, 0)
    assert dx._version_key("/usr/local/cuda") == (0, 0)


def test_has_nvcc(tmp_path):
    assert dx._has_nvcc(None) is False
    assert dx._has_nvcc(str(tmp_path)) is False  # no bin/nvcc yet
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "nvcc").write_text("")
    assert dx._has_nvcc(str(tmp_path)) is True


def test_find_nvcc_honors_cuda_home(tmp_path, monkeypatch):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "nvcc").write_text("")
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    found = dx.find_nvcc()
    assert found == str(tmp_path / "bin" / "nvcc")


def test_find_ninja_returns_path_or_none():
    found = dx.find_ninja()
    assert found is None or isinstance(found, str)


def test_overall_ok():
    ready = {
        "torch": {"installed": True, "cuda_available": True},
        "nvcc": "/usr/local/cuda/bin/nvcc", "ninja": "/usr/bin/ninja",
        "match_level": "ok",
    }
    assert dx.overall_ok(ready) is True
    assert dx.overall_ok(dict(ready, match_level="warn")) is True    # major mismatch
    assert dx.overall_ok(dict(ready, ninja=None)) is False
    assert dx.overall_ok(dict(ready, nvcc=None, match_level="error")) is False


def test_kernel_prebuilt_returns_bool():
    assert isinstance(dx.kernel_prebuilt(), bool)


def test_overall_ok_prebuilt_kernel_overrides_build_mismatch():
    # A prebuilt extension runs even when torch and the toolkit disagree on CUDA major.
    base = {
        "torch": {"installed": True, "cuda_available": True},
        "nvcc": None, "ninja": None,
        "match_level": "error",
    }
    assert dx.overall_ok(base) is False                       # no prebuilt kernel → must build
    assert dx.overall_ok(dict(base, kernel_prebuilt=True)) is True
    # but a prebuilt kernel cannot save a missing GPU / torch
    no_gpu = {"installed": True, "cuda_available": False}
    assert dx.overall_ok(dict(base, kernel_prebuilt=True, torch=no_gpu)) is False


def test_format_report_prebuilt_demotes_mismatch_to_ready():
    report = {
        "python": "3.11.0", "platform": "Linux-test",
        "torch": {"installed": True, "version": "2.11.0+cu130", "built_cuda": "13.0",
                  "cuda_available": True, "device": "RTX 3080"},
        "nvcc": "/usr/local/cuda-12.8/bin/nvcc", "toolkit_version": "12.8",
        "ninja": None, "kernel_prebuilt": True,
        "match_level": "error", "match_message": "CUDA major mismatch: ...",
    }
    text = dx.format_report(report)
    assert "READY" in text and "NOT READY" not in text
    assert "prebuilt" in text


def test_format_report_smoke():
    report = {
        "python": "3.11.0",
        "platform": "Linux-test",
        "torch": {"installed": True, "version": "2.9.0+cu128", "built_cuda": "12.8",
                  "cuda_available": True, "device": "RTX 3080"},
        "nvcc": "/usr/local/cuda-12.8/bin/nvcc",
        "toolkit_version": "12.8",
        "ninja": "/usr/bin/ninja",
        "match_level": "ok",
        "match_message": "torch CUDA 12.8 matches the toolkit (12.8).",
    }
    text = dx.format_report(report)
    assert isinstance(text, str)
    assert "RTX 3080" in text
    assert "READY" in text
    assert "[ OK ]" in text


def test_collect_has_expected_keys():
    report = dx.collect()
    for key in ("python", "platform", "torch", "nvcc", "toolkit_version",
                "ninja", "match_level", "match_message"):
        assert key in report
    assert report["match_level"] in ("ok", "warn", "error")
