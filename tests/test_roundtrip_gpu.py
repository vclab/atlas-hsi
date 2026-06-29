"""GPU smoke test: the synthetic tiny cube trains, serializes to an ATLAS bitstream, and
decodes back to a sane reconstruction. Auto-skipped without a CUDA GPU + atlas_gsplat
(see conftest.py). The full multi-dataset matrix lives in tests/selftest.py.
"""
import harness
import pytest


@pytest.mark.gpu
def test_tiny_cube_roundtrip():
    case = harness.get_case("tiny_cube")
    metrics = harness.run_case(case, mode="quick")

    assert tuple(metrics["shape"]) == (48, 48, 20)
    assert metrics["file_bytes"] > 0
    assert metrics["n_gaussians"] > 0
    assert metrics["psnr"] > case.quick_psnr_min
    assert metrics["bpppb"] < case.bpppb_max
