#!/usr/bin/env python3
"""ATLAS self-test -- one command to validate the atlas-hsi package end to end.

Three stages:

  1. unit (cpu)        the CPU-only pytest suite (codec, data, diagnostics, viz, cli-init)
  2. integration (gpu) the synthetic tiny-cube train -> compress -> decompress round-trip
  3. datasets (gpu)    every dataset whose cube is present, through the full pipeline

Hard floors (PSNR / bpppb) fail the run. Recorded golden values (tests/golden.json) flag
drift. Datasets whose cube is not downloaded are SKIPPED, so new datasets light up
automatically once their file lands in data/raw/.

Examples:
  python tests/selftest.py                   # full pipeline on every present dataset
  python tests/selftest.py --quick           # short configs, fast sanity (seconds-minutes)
  python tests/selftest.py --only paviau salinas
  python tests/selftest.py --record          # refresh golden.json from this (full) run
  python tests/selftest.py --no-cpu          # skip the pytest stage
  python tests/selftest.py --strict          # treat golden drift as failure too
  python tests/selftest.py --list            # list cases and exit

Run from the repo root so data/ paths resolve. Lives in tests/ (not scripts/, which this
repo gitignores) so it ships in the source tree and runs on a fresh clone or cluster.
"""
import argparse
import contextlib
import io
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness  # noqa: E402  (path set above)

REPO = Path(__file__).resolve().parents[1]


def _fmt_secs(s):
    return f"{int(s // 60)}m{int(s % 60):02d}s" if s >= 60 else f"{s:.1f}s"


def run_cpu_stage():
    """Run the CPU-only pytest tier in a subprocess. Returns (ok, summary_line)."""
    print("[1/3] unit (cpu)")
    cmd = [sys.executable, "-m", "pytest", "tests/", "-m", "not gpu", "-q"]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    summary = lines[-1] if lines else "(no pytest output)"
    ok = proc.returncode == 0
    print(f"      {'PASS' if ok else 'FAIL'}  {summary}")
    if not ok:
        print(proc.stdout[-2000:])
    return ok, summary


def run_case_quiet(case, mode, verbose):
    """Run one case, capturing the noisy trainer output unless --verbose. Returns
    (metrics, error, log)."""
    if verbose:
        return harness.run_case(case, mode=mode), None, ""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            metrics = harness.run_case(case, mode=mode)
        return metrics, None, buf.getvalue()
    except Exception as exc:
        return None, exc, buf.getvalue()


def report(name, metrics):
    return (f"  {name:<12} PSNR {metrics['psnr']:6.2f}  bpppb {metrics['bpppb']:.4f}  "
            f"N {metrics['n_gaussians']:>6}  ({_fmt_secs(metrics['seconds'])})")


def main():
    ap = argparse.ArgumentParser(description="ATLAS package self-test")
    ap.add_argument("--quick", action="store_true", help="short configs (fast sanity)")
    ap.add_argument("--record", action="store_true",
                    help="write golden.json from this run (full mode only)")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="run only these case names")
    ap.add_argument("--no-cpu", action="store_true", help="skip the pytest (cpu) stage")
    ap.add_argument("--strict", action="store_true", help="golden drift fails the run too")
    ap.add_argument("--verbose", action="store_true", help="show full trainer output")
    ap.add_argument("--list", action="store_true", help="list cases and exit")
    args = ap.parse_args()

    cases = harness.all_cases()
    if args.only:
        wanted = set(args.only)
        cases = [c for c in cases if c.name in wanted]
        missing = wanted - {c.name for c in cases}
        if missing:
            ap.error(f"unknown case(s): {', '.join(sorted(missing))}")

    if args.list:
        print(f"{'CASE':<12} {'DATASET':<28} STATUS")
        for c in cases:
            status = "available" if harness.dataset_available(c) else "no data"
            tag = " (forthcoming)" if c.forthcoming else ""
            print(f"{c.name:<12} {c.dataset:<28} {status}{tag}")
        return 0

    mode = "quick" if args.quick else "full"
    if args.record and args.quick:
        ap.error("--record requires full mode (drop --quick)")

    t_start = time.time()
    failures, drifts, skipped, ran = [], [], [], []
    golden = harness.load_golden()

    if not args.no_cpu:
        cpu_ok, _ = run_cpu_stage()
        if not cpu_ok:
            failures.append("unit (cpu)")

    if not harness.gpu_ready():
        print("\nNo CUDA GPU, or the CUDA kernel cannot load -> skipping GPU stages "
              "(2 integration, 3 datasets).")
        print("  Run `atlas doctor` to see what is missing (a CUDA toolkit or ninja).")
        _summary(failures, drifts, skipped, ran, t_start)
        return 1 if failures else 0

    synthetic = [c for c in cases if c.name == "tiny_cube"]
    datasets = [c for c in cases if c.name != "tiny_cube"]

    if synthetic:
        print(f"\n[2/3] integration (gpu){'  [quick]' if args.quick else ''}")
        for c in synthetic:
            _run_one(c, mode, args, golden, failures, drifts, skipped, ran)

    if datasets:
        print(f"\n[3/3] datasets (gpu){'  [quick]' if args.quick else '  [full pipeline]'}")
        for c in datasets:
            _run_one(c, mode, args, golden, failures, drifts, skipped, ran)

    if args.record and ran:
        for name, metrics in ran:
            golden[name] = {"psnr": round(metrics["psnr"], 3),
                            "bpppb": round(metrics["bpppb"], 5)}
        harness.save_golden(golden)
        print(f"\nRecorded golden values for {len(ran)} case(s) -> {harness.GOLDEN_PATH}")

    _summary(failures, drifts, skipped, ran, t_start)
    if failures:
        return 1
    if drifts and args.strict:
        return 1
    return 0


def _run_one(case, mode, args, golden, failures, drifts, skipped, ran):
    if not harness.dataset_available(case):
        note = " (forthcoming)" if case.forthcoming else ""
        print(f"  {case.name:<12} SKIP (no data){note}")
        skipped.append(case.name)
        return
    if case.min_vram_gb and harness.gpu_vram_gb() < case.min_vram_gb:
        print(f"  {case.name:<12} SKIP (needs >= {case.min_vram_gb:.0f} GB VRAM, "
              f"have {harness.gpu_vram_gb():.1f})")
        skipped.append(case.name)
        return
    metrics, err, log = run_case_quiet(case, mode, args.verbose)
    if err is not None:
        print(f"  {case.name:<12} ERROR  {type(err).__name__}: {err}")
        if log:
            print("    --- last trainer output ---")
            print("\n".join("    " + ln for ln in log.splitlines()[-12:]))
        failures.append(case.name)
        return
    floor_ok, floor_msgs, drift_msgs = harness.evaluate(case, metrics, mode, golden)
    line = report(case.name, metrics)
    verdict = "OK" if floor_ok else "FAIL"
    print(f"{line}  {verdict}")
    for m in floor_msgs:
        print(f"      ! {m}")
    for m in drift_msgs:
        print(f"      ~ drift: {m}")
    if not floor_ok:
        failures.append(case.name)
    else:
        ran.append((case.name, metrics))
    if drift_msgs:
        drifts.append(case.name)


def _summary(failures, drifts, skipped, ran, t_start):
    print(f"\nSUMMARY: {len(ran)} passed, {len(skipped)} skipped, {len(failures)} failed"
          + (f", {len(drifts)} drift" if drifts else "")
          + f"   ({_fmt_secs(time.time() - t_start)})")
    if failures:
        print("  failed: " + ", ".join(failures))
    if drifts:
        print("  drift:  " + ", ".join(drifts))


if __name__ == "__main__":
    sys.exit(main())
