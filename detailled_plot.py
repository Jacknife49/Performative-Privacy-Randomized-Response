"""Create either heatmaps or two detail plots for one saved (q, r) pair.

Examples:
    python detailled_plot.py --heatmap
    python detailled_plot.py --q 0.01 --r 0.01
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_PLOTS_DIR = SCRIPT_DIR / "plots"


def load_result(path):
    with path.open("rb") as handle:
        result = pickle.load(handle)
    if result.get("format_version") != 4:
        raise ValueError(f"Unsupported result format in {path}")
    return result


def load_results(results_dir):
    paths = sorted(results_dir.glob("result_q_*_r_*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No result_q_*_r_*.pkl files found in {results_dir}")

    loaded = []
    for path in paths:
        loaded.append(load_result(path))

    loaded.sort(key=lambda result: (result["q_index"], result["r_index"]))
    reference = loaded[0]["config"]
    for result in loaded[1:]:
        config = result["config"]
        for key in ("taus", "gammas", "q_values", "r_values"):
            if not np.array_equal(config[key], reference[key]):
                raise ValueError(
                    f"Inconsistent {key} in result for "
                    f"(q={result['q']}, r={result['r']})"
                )
    return loaded, reference


def load_pair_result(results_dir, requested_q, requested_r):
    path = (
        results_dir
        / f"result_q_{requested_q:.8f}_r_{requested_r:.8f}.pkl"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"No result file found for q={requested_q:.8f}, "
            f"r={requested_r:.8f} in {results_dir}. "
            "Run single_q_r.py for this pair first."
        )

    result = load_result(path)
    if not (
        np.isclose(result["q"], requested_q, rtol=0.0, atol=5e-9)
        and np.isclose(result["r"], requested_r, rtol=0.0, atol=5e-9)
    ):
        raise ValueError(
            f"Result file {path} contains (q={result['q']}, r={result['r']}), "
            f"not (q={requested_q}, r={requested_r})"
        )
    return result, result["config"]


def plot_detail(result, config, plots_dir):
    q = result["q"]
    r = result["r"]
    gammas = np.asarray(config["gammas"])
    T = config["T"]
    saved_paths = []

    for tau, pair_result in result["results_by_tau"].items():
        empirical_C_gammas = np.asarray(pair_result["empirical_C_gammas"])
        empirical_C_std_gammas = np.asarray(pair_result["empirical_C_std_gammas"])
        Nt_means = np.asarray(pair_result["Nt_mean_by_gamma"])
        Nt_stds = np.asarray(pair_result["Nt_std_by_gamma"])

        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = cm.viridis
        norm = mcolors.LogNorm(vmin=min(gammas), vmax=max(gammas))
        time_steps = np.arange(T + 1)
        for gamma_index, gamma_value in enumerate(gammas):
            gamma = float(gamma_value)
            mean_Nt = Nt_means[gamma_index]
            std_Nt = Nt_stds[gamma_index]
            color = cmap(norm(gamma))
            ax.plot(time_steps, mean_Nt, color=color, linewidth=1.5)
            ax.fill_between(
                time_steps,
                np.maximum(0, mean_Nt - std_Nt),
                mean_Nt + std_Nt,
                color=color,
                alpha=0.3,
            )
        ax.set_xlabel("Time step $t$", fontsize=14)
        ax.set_ylabel("$N_t$ (Averaged across trials)", fontsize=14)
        ax.set_title(
            f"Population Trajectory ($\\tau={tau}, q={q:.2f}, r={r:.2f}$)",
            fontsize=16,
        )
        ax.grid(True, linestyle="--", alpha=0.5)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r"$\gamma$ (log scale)", fontsize=14)
        plt.tight_layout()
        destination = plots_dir / "population_trajectories"
        destination.mkdir(parents=True, exist_ok=True)
        trajectory_path = (
            destination / f"pop_traj_tau_{tau}_q_{q:.2f}_r_{r:.2f}.png"
        )
        fig.savefig(trajectory_path, dpi=150)
        plt.close(fig)
        saved_paths.append(trajectory_path)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            gammas,
            empirical_C_gammas,
            yerr=empirical_C_std_gammas,
            marker="o",
            markersize=6,
            color="dodgerblue",
            linewidth=2,
            capsize=4,
            label=r"Empirical $C(\gamma)$",
        )
        ax.axhline(
            1 - q,
            color="firebrick",
            linestyle="--",
            linewidth=1.8,
            label=rf"$1-q={1-q:.2f}$",
        )
        ax.axhline(
            1 + r,
            color="darkgreen",
            linestyle="--",
            linewidth=1.8,
            label=rf"$1+r={1+r:.2f}$",
        )
        ax.set_xlabel(r"Noise Scale $\gamma$", fontsize=12)
        ax.set_ylabel(r"$C(\gamma)$", fontsize=12)
        ax.set_title(
            rf"Empirical Population Multiplier ($\tau={tau}, q={q:.2f}, r={r:.2f}$)",
            fontsize=14,
        )
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xscale("log")
        plt.tight_layout()
        destination = plots_dir / "c_gamma"
        destination.mkdir(parents=True, exist_ok=True)
        c_gamma_path = destination / f"C_gamma_tau_{tau}_q_{q:.2f}_r_{r:.2f}.png"
        fig.savefig(c_gamma_path, dpi=150)
        plt.close(fig)
        saved_paths.append(c_gamma_path)

    return saved_paths


def plot_heatmaps(results, config, plots_dir):
    q_values = np.asarray(config["q_values"])
    r_values = np.asarray(config["r_values"])
    expected = {
        (q_index, r_index)
        for q_index in range(len(q_values))
        for r_index in range(len(r_values))
    }
    actual = {(result["q_index"], result["r_index"]) for result in results}
    missing = sorted(expected - actual)
    if missing:
        preview = ", ".join(
            f"({q_values[q_index]:.4f}, {r_values[r_index]:.4f})"
            for q_index, r_index in missing[:20]
        )
        suffix = " ..." if len(missing) > 20 else ""
        raise ValueError(
            f"Cannot create complete heatmaps; missing {len(missing)} "
            f"(q, r) result(s): {preview}{suffix}"
        )
    duplicates = len(results) - len(actual)
    if duplicates:
        raise ValueError(f"Found {duplicates} duplicate (q, r) result file(s)")

    heatmap_dir = plots_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for tau_value in config["taus"]:
        tau = float(tau_value)
        best_gamma_map = np.full((len(r_values), len(q_values)), np.nan)
        for result in results:
            q_index = result["q_index"]
            r_index = result["r_index"]
            pair_result = result["results_by_tau"][tau]
            errors = np.asarray(pair_result["gamma_errors"])
            best_gamma_map[r_index, q_index] = config["gammas"][np.argmin(errors)]

        fig, ax = plt.subplots(figsize=(8, 6))
        extent = [q_values.min(), q_values.max(), r_values.min(), r_values.max()]
        image = ax.imshow(
            best_gamma_map,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
        )
        ax.set_xlabel("Leave probability q", fontsize=12)
        ax.set_ylabel("Join probability r", fontsize=12)
        ax.set_title(f"Optimal Noise Scale $\\gamma$ ($\\tau$ = {tau})", fontsize=14)
        cbar = plt.colorbar(image)
        cbar.set_label("Optimal $\\gamma$")
        plt.tight_layout()
        heatmap_path = heatmap_dir / f"heatmap_tau_{tau}.png"
        fig.savefig(heatmap_path)
        plt.close(fig)
        saved_paths.append(heatmap_path)
    return saved_paths


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--heatmap",
        action="store_true",
        help="create only the heatmap (requires all grid-pair result files)",
    )
    mode.add_argument("--q", type=float, help="q value for the two detail plots")
    parser.add_argument("--r", type=float, help="r value for the two detail plots")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    args = parser.parse_args()
    if args.heatmap and args.r is not None:
        parser.error("--r cannot be used with --heatmap")
    if args.q is not None and args.r is None:
        parser.error("--r is required when --q is used")
    return args


def main():
    args = parse_args()
    results_dir = args.results_dir.resolve()
    plots_dir = args.plots_dir.resolve()

    if args.heatmap:
        results, config = load_results(results_dir)
        saved_paths = plot_heatmaps(results, config, plots_dir)
    else:
        result, config = load_pair_result(results_dir, args.q, args.r)
        saved_paths = plot_detail(result, config, plots_dir)

    for path in saved_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
