import os
import glob
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ============================================================
# Plot Aesthetics
# ============================================================
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 14,
    "axes.grid": True,
    "grid.color": "#E0E0E0",
    "grid.linestyle": "-",
    "grid.alpha": 0.7
})

import math

def extract_empirical_gammas(input_folder, target_tau=0.02, output_txt=None):
    """
    Reads all empirical pickle files, extracts the optimal gamma for each (q, r),
    returns a dictionary structured as data[q][r] = gamma, and optionally saves to text.
    """
    empirical_data = {}
    pkl_files = glob.glob(os.path.join(input_folder, "*.pkl"))
    
    if not pkl_files:
        print(f"Warning: No .pkl files found in '{input_folder}'.")
        return empirical_data

    print(f"Processing {len(pkl_files)} empirical .pkl files...")
    flat_results = []

    for filepath in pkl_files:
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            # 1. Extract q and r (Top-level keys)
            if 'q' in data and 'r' in data:
                q, r = data['q'], data['r']
            else:
                # Fallback: parse q and r directly from the filename
                match = re.search(r'q_([\d.]+)_r_([\d.]+)', os.path.basename(filepath))
                if match:
                    q, r = float(match.group(1)), float(match.group(2))
                else:
                    print(f"Skipping {filepath}: Could not extract q and r.")
                    continue
            
            # 2. Extract the gammas array (Nested inside 'config')
            gammas = data['config']['gammas']
            
            # 3. Match the tau key safely using math.isclose to avoid float precision mismatch
            results_by_tau = data.get('results_by_tau', {})
            matching_tau_key = None
            
            for key in results_by_tau.keys():
                if math.isclose(key, target_tau, rel_tol=1e-5, abs_tol=1e-8):
                    matching_tau_key = key
                    break
            
            if matching_tau_key is not None:
                gamma_errors = results_by_tau[matching_tau_key]['gamma_errors']
                
                # Identify optimal empirical gamma ignoring NaNs
                opt_gamma_idx = np.nanargmin(gamma_errors)
                opt_gamma = gammas[opt_gamma_idx]
                
                if q not in empirical_data:
                    empirical_data[q] = {}
                empirical_data[q][r] = opt_gamma
                
                flat_results.append((q, r, opt_gamma))
            else:
                print(f"Skipping {filepath}: tau={target_tau} missing. Available taus: {list(results_by_tau.keys())}")
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    if output_txt and flat_results:
        flat_results.sort(key=lambda x: (x[0], x[1]))
        with open(output_txt, 'w') as f:
            for q, r, opt_gamma in flat_results:
                f.write(f"Optimal gamma for (q, r) = ({q:.2f}, {r:.2f}): {opt_gamma:.4f}\n")
        print(f"Saved empirical summaries to '{output_txt}'.")

    return empirical_data
def parse_theory_gammas(filepath):
    """
    Parses the theoretical text file and returns a dictionary data[q][r] = gamma.
    """
    theory_data = {}
    if not os.path.exists(filepath):
        print(f"Warning: Theoretical file '{filepath}' not found.")
        return theory_data
        
    pattern = re.compile(r"Optimal gamma for \(q, r\) = \(([\d.]+), ([\d.]+)\): ([\d.]+)")
    
    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                q, r, gamma = float(match.group(1)), float(match.group(2)), float(match.group(3))
                if q not in theory_data:
                    theory_data[q] = {}
                theory_data[q][r] = gamma
                
    print(f"Loaded theoretical data from '{filepath}'.")
    return theory_data

def plot_gamma_comparison(theory_data, empirical_data, output_image):
    """
    Generates a line plot comparing theoretical and empirical optimal gamma
    values across join probabilities (r), color-coded by leave probability (q).
    """
    # Extract sorted q values present in either dataset
    all_qs = sorted(list(set(theory_data.keys()).union(set(empirical_data.keys()))))
    
    if not all_qs:
        print("No data available to plot.")
        return

    # Select a visually readable subset of q values (e.g., 8 lines)
    num_lines = min(8, len(all_qs))
    q_indices = np.linspace(0, len(all_qs) - 1, num_lines, dtype=int)
    selected_qs = [all_qs[i] for i in q_indices]
    
    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    # Plot lines mapping q to color
    for q in selected_qs:
        color = cmap(norm(q))
        
        # Plot Theory (Solid)
        if q in theory_data:
            r_vals_th = sorted(list(theory_data[q].keys()))
            gamma_vals_th = [theory_data[q][r] for r in r_vals_th]
            ax.plot(r_vals_th, gamma_vals_th, color=color, linestyle='-', linewidth=3.5, alpha=0.9)
            
        # Plot Empirical (Dashed)
        if q in empirical_data:
            r_vals_emp = sorted(list(empirical_data[q].keys()))
            gamma_vals_emp = [empirical_data[q][r] for r in r_vals_emp]
            ax.plot(r_vals_emp, gamma_vals_emp, color=color, linestyle='--', linewidth=3.5, alpha=0.9)

    ax.set_yscale('log')
    ax.set_xlabel("Join probability ($r$)")
    ax.set_ylabel("Optimal $\gamma$")
    ax.set_xlim(-0.05, 1.05)
    
    legend_elements = [
        Line2D([0], [0], color='black', lw=3.5, linestyle='-', label='Theory'),
        Line2D([0], [0], color='black', lw=3.5, linestyle='--', label='Empirical')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=1.0)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.04)
    cbar.set_label('Leave probability ($q$)', rotation=90, labelpad=15, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_image, bbox_inches='tight')
    plt.show()
    print(f"Comparison plot saved to '{output_image}'.")

if __name__ == "__main__":
    # Define IO paths
    PKL_DIRECTORY = "results_regression"
    THEORY_TXT_FILE = "results_0.02_theory.txt"
    EMPIRICAL_TXT_FILE = "results_0.02_empirical.txt" # Set to None to skip writing
    OUTPUT_PLOT = "gamma_line_comparison_ols.png"
    TAU_VALUE = 0.02
    
    # Execute combined workflow
    emp_data = extract_empirical_gammas(PKL_DIRECTORY, target_tau=TAU_VALUE, output_txt=EMPIRICAL_TXT_FILE)
    th_data = parse_theory_gammas(THEORY_TXT_FILE)
    plot_gamma_comparison(th_data, emp_data, OUTPUT_PLOT)
