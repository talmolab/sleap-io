#!/usr/bin/env python
"""Performance benchmarks for lazy SLP loading.

This script compares eager vs lazy loading performance for SLP files.

Usage:
    python scripts/benchmark_lazy.py [--file PATH] [--runs N]

If no file is specified, uses the test fixture file.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path


def benchmark_load(path: str, lazy: bool, n_runs: int = 5) -> dict:
    """Benchmark load time.

    Args:
        path: Path to SLP file.
        lazy: Whether to use lazy loading.
        n_runs: Number of runs for averaging.

    Returns:
        Dictionary with timing results.
    """
    import sleap_io as sio

    times = []
    for _ in range(n_runs):
        gc.collect()
        start = time.perf_counter()
        labels = sio.load_slp(path, lazy=lazy)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        del labels

    return {
        "avg": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
        "times": times,
    }


def benchmark_load_numpy(path: str, lazy: bool, n_runs: int = 5) -> dict:
    """Benchmark load + numpy() time.

    Args:
        path: Path to SLP file.
        lazy: Whether to use lazy loading.
        n_runs: Number of runs for averaging.

    Returns:
        Dictionary with timing results.
    """
    import sleap_io as sio

    times = []
    for _ in range(n_runs):
        gc.collect()
        start = time.perf_counter()
        labels = sio.load_slp(path, lazy=lazy)
        _ = labels.numpy()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        del labels

    return {
        "avg": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
        "times": times,
    }


def benchmark_iteration(path: str, lazy: bool, n_runs: int = 3) -> dict:
    """Benchmark iteration over all frames.

    Args:
        path: Path to SLP file.
        lazy: Whether to use lazy loading.
        n_runs: Number of runs for averaging.

    Returns:
        Dictionary with timing results.
    """
    import sleap_io as sio

    times = []
    for _ in range(n_runs):
        gc.collect()
        labels = sio.load_slp(path, lazy=lazy)
        start = time.perf_counter()
        for lf in labels:
            _ = lf.frame_idx
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        del labels

    return {
        "avg": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
        "times": times,
    }


def get_file_info(path: str) -> dict:
    """Get information about the SLP file.

    Args:
        path: Path to SLP file.

    Returns:
        Dictionary with file information.
    """
    import sleap_io as sio

    labels = sio.load_slp(path)
    n_frames = len(labels)
    n_instances = sum(len(lf) for lf in labels)
    n_nodes = len(labels.skeletons[0].nodes) if labels.skeletons else 0
    file_size_mb = Path(path).stat().st_size / 1024 / 1024

    info = {
        "path": str(path),
        "file_size_mb": file_size_mb,
        "n_frames": n_frames,
        "n_instances": n_instances,
        "n_nodes": n_nodes,
        "n_videos": len(labels.videos),
        "n_tracks": len(labels.tracks),
    }
    del labels
    return info


def print_results(
    name: str,
    eager_result: dict,
    lazy_result: dict,
    target_speedup: float | None = None,
) -> float:
    """Print benchmark results comparison.

    Args:
        name: Name of the benchmark.
        eager_result: Eager timing results.
        lazy_result: Lazy timing results.
        target_speedup: Expected minimum speedup.

    Returns:
        Actual speedup achieved.
    """
    speedup = eager_result["avg"] / lazy_result["avg"]
    print(f"\n{name}:")
    print(f"  Eager: {eager_result['avg']:.4f}s (min={eager_result['min']:.4f}s)")
    print(f"  Lazy:  {lazy_result['avg']:.4f}s (min={lazy_result['min']:.4f}s)")
    print(f"  Speedup: {speedup:.1f}x")

    if target_speedup is not None:
        if speedup >= target_speedup:
            print(f"  [PASS] Meets target of {target_speedup}x")
        else:
            print(f"  [WARN] Below target of {target_speedup}x")

    return speedup


def run_benchmarks(path: str, n_runs: int = 5) -> dict:
    """Run all benchmarks on a file.

    Args:
        path: Path to SLP file.
        n_runs: Number of runs for averaging.

    Returns:
        Dictionary with all results.
    """
    print("=" * 60)
    print("LAZY LOADING PERFORMANCE BENCHMARKS")
    print("=" * 60)

    # File info
    info = get_file_info(path)
    print(f"\nFile: {info['path']}")
    print(f"Size: {info['file_size_mb']:.2f} MB")
    print(f"Frames: {info['n_frames']}")
    print(f"Instances: {info['n_instances']}")
    print(f"Nodes: {info['n_nodes']}")
    print(f"Videos: {info['n_videos']}")
    print(f"Tracks: {info['n_tracks']}")

    results = {"info": info}

    # Load only benchmark
    print(f"\nRunning benchmarks with {n_runs} runs each...")
    print("-" * 40)

    eager_load = benchmark_load(path, lazy=False, n_runs=n_runs)
    lazy_load = benchmark_load(path, lazy=True, n_runs=n_runs)
    results["load_speedup"] = print_results(
        "Load only", eager_load, lazy_load, target_speedup=10.0
    )

    # Load + numpy() benchmark
    eager_numpy = benchmark_load_numpy(path, lazy=False, n_runs=n_runs)
    lazy_numpy = benchmark_load_numpy(path, lazy=True, n_runs=n_runs)
    results["numpy_speedup"] = print_results(
        "Load + numpy()", eager_numpy, lazy_numpy, target_speedup=5.0
    )

    # Iteration benchmark (lazy is slower here - expected)
    eager_iter = benchmark_iteration(path, lazy=False, n_runs=min(n_runs, 3))
    lazy_iter = benchmark_iteration(path, lazy=True, n_runs=min(n_runs, 3))
    iter_speedup = print_results("Full iteration (lazy slower)", eager_iter, lazy_iter)
    results["iter_speedup"] = iter_speedup

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Load speedup: {results['load_speedup']:.1f}x")
    print(f"  Load + numpy() speedup: {results['numpy_speedup']:.1f}x")
    print(f"  Iteration speedup: {results['iter_speedup']:.2f}x (< 1 expected)")

    targets_met = results["load_speedup"] >= 10.0 and results["numpy_speedup"] >= 5.0
    if targets_met:
        print("\n  [SUCCESS] All performance targets met!")
    else:
        print("\n  [WARNING] Some performance targets not met")

    print("\n" + "=" * 60)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark lazy vs eager SLP loading performance"
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Path to SLP file to benchmark",
    )
    parser.add_argument(
        "--runs",
        "-n",
        type=int,
        default=5,
        help="Number of runs for averaging (default: 5)",
    )
    args = parser.parse_args()

    # Find benchmark file
    if args.file:
        slp_path = Path(args.file)
    else:
        # Try scratch folder first (larger benchmark file)
        scratch_dir = Path(__file__).parent.parent / "scratch"
        scratch_file = (
            scratch_dir
            / "2026-01-05-lazy-slp-loading"
            / "clip_01_24_12_20_11_38_16_cam08.slp"
        )
        if scratch_file.exists():
            slp_path = scratch_file
        else:
            # Fall back to test fixtures
            fixtures_dir = Path(__file__).parent.parent / "tests" / "data"
            slp_files = list(fixtures_dir.glob("*.slp"))
            if slp_files:
                slp_path = slp_files[0]
            else:
                print("ERROR: No SLP file found for benchmarking.")
                print("  Provide a file with --file or add test fixtures.")
                sys.exit(1)

    if not slp_path.exists():
        print(f"ERROR: File not found: {slp_path}")
        sys.exit(1)

    run_benchmarks(str(slp_path), n_runs=args.runs)


if __name__ == "__main__":
    main()
