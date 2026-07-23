import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import warnings
import os

warnings.filterwarnings("ignore")

# -------------------------
# Base Parameters
# -------------------------
np.random.seed(42)

d = 12
mu = np.zeros(d)
Sigma = np.eye(d)

N0 = 1000
T = 50
R = 10.0

# Define intervals of t you want to snapshot
t_intervals = [0, 10, 20, 30, 49] 

# Define multiple thresholds for tau
taus = [0.02] 
#taus = [0.02, 0.05, 0.1, 0.2]

# Search over gamma
gammas = np.geomspace(0.2, 20.0, 10)

# Grid of q and r 
q_values = np.linspace(0.01, 0.99, 30)
r_values = np.linspace(0.01, 0.99, 30)

trials = 5

# Directories
heatmap_dir = "/home/uddalak/Performative_Privacy_likelihood/Performative_Privacy_MIA_heatmaps"
hist_base_dir = "/home/uddalak/Performative_Privacy_likelihood/Performative_Privacy_MIA_Histograms"
pop_traj_dir = "/home/uddalak/Performative_Privacy_likelihood/Performative_Privacy_MIA_PopTraj"
c_gamma_dir = "/home/uddalak/Performative_Privacy_likelihood/Performative_Privacy_MIA_C_gamma" # NEW DIR

os.makedirs(heatmap_dir, exist_ok=True)
os.makedirs(hist_base_dir, exist_ok=True)
os.makedirs(pop_traj_dir, exist_ok=True)
os.makedirs(c_gamma_dir, exist_ok=True)

# -------------------------
# Sweep over tau, q, and r
# -------------------------
for tau in taus:
    print(f"Processing for tau = {tau}...")
    
    best_gamma_map = np.zeros((len(r_values), len(q_values)))
    min_error_map = np.zeros_like(best_gamma_map)

    for i, r in enumerate(r_values):
        for j, q in enumerate(q_values):

            gamma_errors = []
            
            # Array to hold the empirical expected ratio for each gamma at this q, r
            empirical_C_gammas = []
            
            # Dictionary to store Lambdas for ALL gammas at this specific q, r combination
            lambdas_all_gammas = {g: {t: [] for t in t_intervals} for g in gammas}
            
            # Dictionary to store Nt trajectories across trials
            Nt_history_all_gammas = {g: np.zeros((trials, T + 1)) for g in gammas}

            for gamma in gammas:
                trial_errors = []
                ratios_this_gamma = [] # Track N_t / N_{t-1} for this specific gamma

                for trial in range(trials):
                    Nt = N0
                    Nt_history_all_gammas[gamma][trial, 0] = Nt
                    
                    valid_rounds = 0
                    cumulative_error = 0.0  # CHANGED: Now tracks the sum of errors

                    for t in range(T):
                        if Nt <= 1:
                            # Forward-fill the remaining steps if population collapses
                            Nt_history_all_gammas[gamma][trial, t+1:] = Nt
                            break

                        # Generate data
                        Z = np.random.multivariate_normal(mu, Sigma, Nt)
                        norms = np.linalg.norm(Z, axis=1)

                        clip = np.minimum(1.0, R / np.where(norms == 0, 1e-9, norms))
                        Z_bar = Z * clip[:, None]

                        mu_Z = np.mean(Z_bar, axis=0)

                        noise = np.random.normal(0, gamma / np.sqrt(Nt), size=d)
                        o_t = mu_Z + noise

                        # Accumulate squared error
                        cumulative_error += np.sum((o_t - mu) ** 2)
                        valid_rounds += 1

                        # Likelihood ratio
                        term1 = (d / 2) * np.log(Nt / (Nt - 1))
                        diff1 = o_t - (Nt * mu_Z - Z_bar) / (Nt - 1)
                        diff2 = o_t - mu_Z

                        term2 = (Nt - 1) * np.sum(diff1**2, axis=1) / (2 * gamma**2)
                        term3 = Nt * np.sum(diff2**2) / (2 * gamma**2)

                        Lambda = term1 + term2 - term3
                        
                        # Capture Lambda
                        if t in t_intervals:
                            lambdas_all_gammas[gamma][t].extend(Lambda)

                        leaves = np.sum((Lambda > tau) & (np.random.rand(Nt) < q))
                        joins = np.sum((Lambda <= tau) & (np.random.rand(Nt) < r))

                        Nt_next = min(20000, max(1, Nt - leaves + joins))
                        
                        # Track the ratio before overwriting Nt
                        ratios_this_gamma.append(Nt_next / Nt)
                        
                        Nt = Nt_next
                        Nt_history_all_gammas[gamma][trial, t+1] = Nt

                    if valid_rounds > 0:
                        # Normalize the error sum to the full horizon T so early crashes aren't "rewarded"
                        adjusted_error = (cumulative_error / valid_rounds) * T
                        trial_errors.append(adjusted_error)
                    else:
                        trial_errors.append(np.inf)

                # Store the mean error for this gamma
                gamma_errors.append(np.mean(trial_errors))
                
                # Store the mean ratio (empirical C(gamma)) for this gamma
                if len(ratios_this_gamma) > 0:
                    empirical_C_gammas.append(np.mean(ratios_this_gamma))
                else:
                    empirical_C_gammas.append(np.nan)

            gamma_errors = np.array(gamma_errors)
            idx_best = np.argmin(gamma_errors)

            best_gamma_map[i, j] = gammas[idx_best]
            min_error_map[i, j] = gamma_errors[idx_best]
            print(f"For tau = {tau}, q = {q:.2f} and r = {r:.2f}, min error gamma is {gammas[idx_best]:.2f}")

            # -------------------------------------------------------------
            # PLOTTING: Population Trajectory with Confidence Bands
            # -------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(10, 6))
            
            cmap = cm.viridis
            norm = mcolors.LogNorm(vmin=min(gammas), vmax=max(gammas))
            time_steps = np.arange(T + 1)
            
            for gamma in gammas:
                color = cmap(norm(gamma))
                
                mean_Nt = np.mean(Nt_history_all_gammas[gamma], axis=0)
                std_Nt = np.std(Nt_history_all_gammas[gamma], axis=0)
                
                ax.plot(time_steps, mean_Nt, color=color, linewidth=1.5)
                ax.fill_between(time_steps, 
                                np.maximum(0, mean_Nt - std_Nt), 
                                mean_Nt + std_Nt, 
                                color=color, alpha=0.3)
                
            ax.set_xlabel("Time step $t$", fontsize=14)
            ax.set_ylabel("$N_t$ (Averaged across trials)", fontsize=14)
            ax.set_title(f"Population Trajectory ($\\tau={tau}, q={q:.2f}, r={r:.2f}$)", fontsize=16)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(r'$\gamma$ (log scale)', fontsize=14)
            
            plt.tight_layout()
            traj_filename = os.path.join(pop_traj_dir, f"pop_traj_tau_{tau}_q_{q:.2f}_r_{r:.2f}.png")
            plt.savefig(traj_filename, dpi=150)
            plt.close(fig)

            # -------------------------------------------------------------
            # PLOTTING: Overlaid Histograms (Best, Worst, Average)
            # -------------------------------------------------------------
            specific_dir = os.path.join(
                hist_base_dir, 
                f"tau_{tau}", 
                f"q_{q:.2f}_r_{r:.2f}"
            )
            os.makedirs(specific_dir, exist_ok=True)
            
            best_gamma_val = gammas[idx_best]
            worst_gamma_val = gammas[np.argmax(gamma_errors)]
            avg_gamma_val = gammas[len(gammas) // 2]
            
            selected_gammas = sorted(list(set([best_gamma_val, worst_gamma_val, avg_gamma_val])), reverse=True)
            color_palette = ['#86C880', '#FCB272', '#E17876'] 
            
            for t_val in t_intervals:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                plot_min = min(-0.1, -tau)
                plot_max = max(0.4, tau * 2)
                bins = np.linspace(plot_min, plot_max, 100)
                
                for k, gamma in enumerate(selected_gammas):
                    data = np.array(lambdas_all_gammas[gamma][t_val])
                    
                    label_suffix = []
                    if gamma == best_gamma_val: label_suffix.append("Best")
                    if gamma == worst_gamma_val: label_suffix.append("Worst")
                    if gamma == avg_gamma_val and gamma not in (best_gamma_val, worst_gamma_val): label_suffix.append("Avg")
                    
                    label_str = f'$\\gamma = {gamma:.2f}$' + (f" ({', '.join(label_suffix)})" if label_suffix else "")
                    
                    if len(data) > 0:
                        ax.hist(data, bins=bins, density=True, 
                                color=color_palette[k % len(color_palette)], 
                                alpha=0.8, label=label_str)
                
                ax.axvline(tau, color='black', linestyle='--', linewidth=2, label=f'Privacy Threshold $\\tau = {tau}$')
                
                ax.set_xlim(plot_min, plot_max)
                ax.set_title("The Shifting Tail of Privacy Leakage", fontsize=14, pad=15)
                ax.set_xlabel("Log-Likelihood Ratio $\\Lambda_t$", fontsize=12)
                ax.set_ylabel("Density (Probability Mass)", fontsize=12)
                ax.legend(loc='upper right', fontsize=11)
                
                plt.tight_layout()
                hist_filename = os.path.join(specific_dir, f"hist_t_{t_val}.png")
                plt.savefig(hist_filename, dpi=150)
                plt.close(fig)

            # -------------------------------------------------------------
            # NEW PLOTTING: Empirical C(gamma)
            # -------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Plot the Empirical Data recorded from the simulation
            ax.plot(gammas, empirical_C_gammas, marker='o', markersize=6, color='dodgerblue', 
                    lw=2, label=r'Empirical Ratio $\mathbb{E}[N_t / N_{t-1}]$')
            
            ax.axhline(1.0, color='black', linestyle=':', alpha=0.6, label='Equilibrium (Ratio=1)')
            
            ax.set_xlabel(r"Noise Scale $\gamma$", fontsize=12)
            ax.set_ylabel(r"Empirical Multiplier $\mathbb{E}[N_t / N_{t-1}]$", fontsize=12)
            ax.set_title(rf"Empirical Population Multiplier ($\tau={tau}, q={q:.2f}, r={r:.2f}$)", fontsize=14)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xscale("log")
            
            plt.tight_layout()
            c_gamma_filename = os.path.join(c_gamma_dir, f"C_gamma_tau_{tau}_q_{q:.2f}_r_{r:.2f}.png")
            plt.savefig(c_gamma_filename, dpi=150)
            plt.close(fig)

    # -------------------------
    # Plot and Save Heatmap
    # -------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [q_values.min(), q_values.max(), r_values.min(), r_values.max()]

    im = ax.imshow(
        best_gamma_map, origin="lower", aspect="auto",
        extent=extent, cmap="viridis"
    )

    ax.set_xlabel("Leave probability q", fontsize=12)
    ax.set_ylabel("Join probability r", fontsize=12)
    ax.set_title(f"Optimal Noise Scale $\\gamma$ ($\\tau$ = {tau})", fontsize=14)

    cbar = plt.colorbar(im)
    cbar.set_label("Optimal $\\gamma$")
    plt.tight_layout()
    
    heatmap_filename = os.path.join(heatmap_dir, f"heatmap_tau_{tau}.png")
    plt.savefig(heatmap_filename)
    plt.close(fig)
