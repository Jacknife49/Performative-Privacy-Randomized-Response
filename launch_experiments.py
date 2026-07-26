"""Launch independent compact v2 q experiments in parallel.

Example:
    python launch_experiments_v2.py --workers 4
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "perf_priv_mia_par.py"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
q_values = np.linspace(0.01, 0.99, 20)


def run_one(q, results_dir, force):
    result_file = results_dir / f"results_q_{q:.8f}.pkl"
    if result_file.exists() and not force:
        return q, "skipped (result already exists)", None

    started_at = time.perf_counter()
    command = [
        sys.executable,
        str(RUNNER),
        "--q",
        f"{q:.17g}",
        "--results-dir",
        str(results_dir),
    ]
    subprocess.run(command, check=True)
    return q, "completed", time.perf_counter() - started_at


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Maximum simultaneous q experiments (default: 3)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Result directory (default: ./results beside this script)",
    )
    parser.add_argument(
        "--force", action="store_true", help="rerun q values with existing results"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    durations = []
    finished = 0
    total = len(q_values)
    launch_started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, float(q), results_dir, args.force): float(q)
            for q in q_values
        }
        for future in as_completed(futures):
            q = futures[future]
            try:
                _, status, duration = future.result()
                finished += 1
                if duration is not None:
                    durations.append(duration)

                message = f"q={q:.8f}: {status}"
                if duration is not None:
                    message += f" in {format_duration(duration)}"
                message += f" | progress {finished}/{total}"

                remaining_jobs = total - finished
                if durations and remaining_jobs:
                    mean_duration = sum(durations) / len(durations)
                    estimated_remaining = mean_duration * remaining_jobs / args.workers
                    message += f" | estimated remaining {format_duration(estimated_remaining)}"
                print(message, flush=True)
            except Exception as exc:
                finished += 1
                failures.append((q, exc))
                print(
                    f"q={q:.8f}: FAILED: {exc} | progress {finished}/{total}",
                    file=sys.stderr,
                    flush=True,
                )

    if failures:
        raise SystemExit(f"{len(failures)} q experiment(s) failed")
    elapsed = time.perf_counter() - launch_started_at
    print(f"ALL q EXPERIMENTS DONE in {format_duration(elapsed)}", flush=True)


if __name__ == "__main__":
    main()
