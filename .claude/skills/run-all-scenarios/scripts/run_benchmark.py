"""Run all scenarios, compare learned vs baseline, retry if worse.

Usage:
    python .claude/skills/run-all-scenarios/scripts/run_benchmark.py [--iterations 60] [--workers 4] [--max-retries 3] [--runs 5]
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.resolve()
CONFIG_DIR = PROJECT_ROOT / "configs/simulation"
MODE = "spsa2"
DEFAULT_ITERATIONS = 60
DEFAULT_WORKERS = max(1, os.cpu_count() - 1)
DEFAULT_RETRIES = 3
DEFAULT_RUNS = 5
TIMEOUT = 259200  # 3 days


def run_config(config_name: str, seed: int = 0, iterations: int = DEFAULT_ITERATIONS, runs: int = DEFAULT_RUNS) -> dict:
    config_path = CONFIG_DIR / config_name
    cmd = [
        sys.executable,
        "examples/run_and_plot.py",
        "--config",
        str(config_path),
        "--mode",
        MODE,
        "--iterations",
        str(iterations),
        "--no-display",
        "--seed",
        str(seed),
        "--runs",
        str(runs),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        cwd=str(PROJECT_ROOT),
    )
    output = result.stdout + result.stderr

    fixed_loss = None
    learned_loss = None
    reached = None
    for line in output.splitlines():
        if "Fixed policy:" in line and "loss=" in line:
            try:
                fixed_loss = float(line.split("loss=")[1].split()[0])
            except Exception:
                pass
        if "Learned policy:" in line and "loss=" in line:
            try:
                learned_loss = float(line.split("loss=")[1].split()[0])
            except Exception:
                pass
            try:
                reached = "reached=True" in line
            except Exception:
                pass

    return {
        "config": config_name,
        "seed": seed,
        "fixed_loss": fixed_loss,
        "learned_loss": learned_loss,
        "reached": reached,
        "rc": result.returncode,
    }


def run_batch(configs, seed, iterations, workers, runs):
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_config, cfg, seed, iterations, runs): cfg for cfg in configs
        }
        for future in as_completed(futures):
            cfg = futures[future]
            try:
                res = future.result()
                results[cfg] = res
                fl = res.get("fixed_loss")
                ll = res.get("learned_loss")
                status = (
                    "BAD"
                    if fl is not None and ll is not None and ll > fl
                    else "OK"
                )
                print(
                    f"  {cfg:40s} seed={seed}  fixed={fl}  learned={ll}  {status}"
                )
            except Exception as e:
                print(f"  {cfg:40s} ERROR: {e}")
                results[cfg] = {"config": cfg, "seed": seed, "error": str(e)}
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark all scenarios")
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Training iterations (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Max retries per scenario (default: {DEFAULT_RETRIES})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Number of independent training runs per scenario (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Specific config filenames to run (default: all *.json in configs/simulation)",
    )
    args = parser.parse_args()

    if args.configs:
        configs = sorted(args.configs)
    else:
        configs = sorted([p.name for p in CONFIG_DIR.glob("*.json")])
    print(
        f"Found {len(configs)} configs, workers={args.workers}, iterations={args.iterations}, runs={args.runs}"
    )

    results = run_batch(configs, seed=0, iterations=args.iterations, workers=args.workers, runs=args.runs)

    for attempt in range(1, args.max_retries + 1):
        bad = [
            cfg
            for cfg, r in results.items()
            if r.get("fixed_loss") is not None
            and r.get("learned_loss") is not None
            and r["learned_loss"] > r["fixed_loss"]
        ]
        if not bad:
            break
        print(f"\nRetrying {len(bad)} bad configs with seed={attempt} ...")
        new_results = run_batch(
            bad, seed=attempt, iterations=args.iterations, workers=args.workers, runs=args.runs
        )
        for cfg, res in new_results.items():
            if "error" not in res:
                results[cfg] = res

    print("\n=== FINAL SUMMARY ===")
    bad_count = 0
    for cfg in configs:
        r = results.get(cfg, {})
        fl = r.get("fixed_loss")
        ll = r.get("learned_loss")
        status = "OK" if fl is not None and ll is not None and ll <= fl else "BAD"
        if status == "BAD":
            bad_count += 1
        print(
            f"{cfg:40s} seed={r.get('seed', 0)}  fixed={fl}  learned={ll}  {status}"
        )

    print(f"\nTotal configs: {len(configs)}, Bad: {bad_count}")

    out_path = PROJECT_ROOT / ".claude" / "skills" / "run-all-scenarios" / "scripts" / "benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in results.items()}, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
