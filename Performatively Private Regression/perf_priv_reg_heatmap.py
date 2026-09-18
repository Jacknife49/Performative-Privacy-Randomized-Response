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

    # Load the first file to extract grid dimensions and config
    with open(pkl_files[0], "rb") as f:
        first_result = pickle.load(f)
    
    q_vals = first_result["config"]["q_values"]
    r_vals = first_result["config"]["r_values"]
    tau = first_result["config"]["tau"]
    
    grid_size_q = len(q_vals)
    grid_size_r = len(r_vals)
    
    # Initialize the matrix with NaNs (so missing data shows up as blank/distinct)
    optimal_gamma_matrix = np.full((grid_size_q, grid_size_r), np.nan)
    
    print(f"Found {len(pkl_files)} result files. Building {grid_size_q}x{grid_size_r} heatmap...")

    # Populate the matrix
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            
        q_idx = data["q_index"]
        r_idx = data["r_index"]
        best_gamma = data["results"]["best_gamma"]
        
        # Populate based on indices
        optimal_gamma_matrix[q_idx, r_idx] = best_gamma

    # Plot the Heatmap matching the reference style
    plt.figure(figsize=(8, 6))
    
    # We transpose (.T) the matrix so q is on the x-axis (columns) and r is on the y-axis (rows).
    # extent ensures the axes reflect the actual probability values rather than array indices.
    # origin='lower' places the minimum (q, r) at the bottom left.
    img = plt.imshow(
        optimal_gamma_matrix.T, 
        extent=[q_vals.min(), q_vals.max(), r_vals.min(), r_vals.max()],
        origin='lower',
        cmap='viridis',
        aspect='auto'
    )
    
    # Configure the colorbar
    cbar = plt.colorbar(img)
    cbar.set_label(r'Optimal $\gamma$')
    
    # Apply exactly matching labels and title
    plt.title(f'Optimal Noise Scale $\gamma$ ($\\tau = {tau}$)', fontsize=13)
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
