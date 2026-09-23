"""Aggregate results and generate the heatmap.

Example:
    python plot_heatmap.py --results-dir /path/to/your/custom_folder
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results_regression",
        help="Directory containing the .pkl result files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir.resolve()

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Discover all pickle files in the directory
    pkl_files = list(results_dir.glob("result_q_*_r_*.pkl"))
    if not pkl_files:
        raise ValueError(f"No .pkl files found in {results_dir}")

    # Load the first file to extract grid dimensions and config.
    # config["taus"] is a list (there can be more than one tau), so we build
    # one heatmap per tau rather than assuming a single config["tau"].
    with open(pkl_files[0], "rb") as f:
        first_result = pickle.load(f)

    q_vals = first_result["config"]["q_values"]
    r_vals = first_result["config"]["r_values"]
    gammas = first_result["config"]["gammas"]
    taus = [float(t) for t in first_result["config"]["taus"]]

    grid_size_q = len(q_vals)
    grid_size_r = len(r_vals)

    print(f"Found {len(pkl_files)} result files. Building {grid_size_q}x{grid_size_r} heatmap(s) for {len(taus)} tau value(s)...")

    # One matrix per tau, initialized with NaN so missing/off-grid pairs show up blank.
    optimal_gamma_matrices = {tau: np.full((grid_size_q, grid_size_r), np.nan) for tau in taus}

    skipped_off_grid = 0
    skipped_missing_tau = 0

    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        q_idx = data["q_index"]
        r_idx = data["r_index"]
        if q_idx is None or r_idx is None:
            # q or r wasn't exactly on the grid for this file; nothing to place.
            skipped_off_grid += 1
            continue

        for tau in taus:
            tau_results = data["results_by_tau"].get(tau)
            if tau_results is None:
                skipped_missing_tau += 1
                continue

            # best_gamma isn't stored directly - recover it from the error curve.
            gamma_errors = tau_results["gamma_errors"]
            best_idx = int(np.argmin(gamma_errors))
            best_gamma = float(gammas[best_idx])

            optimal_gamma_matrices[tau][q_idx, r_idx] = best_gamma

    if skipped_off_grid:
        print(f"Note: skipped {skipped_off_grid} file(s) with q/r not exactly on the stored grid.")
    if skipped_missing_tau:
        print(f"Note: skipped {skipped_missing_tau} (file, tau) entries with no matching results_by_tau key.")

    for tau in taus:
        optimal_gamma_matrix = optimal_gamma_matrices[tau]

        plt.figure(figsize=(8, 6))

        # We transpose (.T) the matrix so q is on the x-axis (columns) and r is on the y-axis (rows).
        # extent ensures the axes reflect the actual probability values rather than array indices.
        # origin='lower' places the minimum (q, r) at the bottom left.
        img = plt.imshow(
            optimal_gamma_matrix.T,
            extent=[q_vals.min(), q_vals.max(), r_vals.min(), r_vals.max()],
            origin='lower',
            cmap='viridis',
            aspect='auto',
            vmin=0.2,
            vmax=12
        )

        # Configure the colorbar
        cbar = plt.colorbar(img)
        cbar.set_label(r'Optimal $\gamma$')

        # Apply exactly matching labels and title
        plt.title(f'Optimal Noise Scale $\\gamma$ ($\\tau = {tau}$)', fontsize=13)
        plt.xlabel('Leave probability q', fontsize=12)
        plt.ylabel('Join probability r', fontsize=12)
        plt.tight_layout()

        # Save and display
        output_filename = f'heatmap_tau_{tau}.png'
        plt.savefig(output_filename, dpi=300)
        print(f"Plot saved successfully as '{output_filename}' in your current working directory.")
        plt.show()


if __name__ == "__main__":
    main()
