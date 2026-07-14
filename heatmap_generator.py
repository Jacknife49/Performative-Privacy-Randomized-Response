import os
import numpy as np
import matplotlib.pyplot as plt

def simulate_final_step_experiment(
    epsilon,
    time_horizon_T,
    initial_participants_N,
    join_prob,
    leave_prob,
    model_type="original",
    trials=10,
    nmax=20000,
    rng_seed=None
):
    rng = np.random.default_rng(rng_seed)
    p = np.exp(epsilon) / (1.0 + np.exp(epsilon))
    final_errors = []
    N_track = []
    
    for _ in range(trials):
        N = int(initial_participants_N)

        for t in range(time_horizon_T - 1):
            if N <= 0:
                N = 0
                break

            leaked_count = rng.binomial(N, p)

            if model_type == "original":
                unleaked_count = N - leaked_count
                joining = rng.binomial(unleaked_count, join_prob)
            elif model_type == "agnostic":
                joining = rng.binomial(N, join_prob)
            else:
                raise ValueError("model_type must be 'original' or 'agnostic'")

            leaving = rng.binomial(leaked_count, leave_prob)
            N = N + joining - leaving
            N = min(max(0, int(N)), nmax)

        if N <= 0:
            final_errors.append(10)
            N_track.append(1)
        else:
            mu_T = rng.uniform(0.5, 1.0)
            X = rng.binomial(1, mu_T, N)

            keep_mask = rng.binomial(1, p, N)
            Y = np.where(keep_mask == 1, X, 1 - X)

            nu_T = np.mean(Y)
            mu_hat = (nu_T - (1.0 - p)) / (2.0 * p - 1.0)
            mu_hat = np.clip(mu_hat, 0.0, 1.0)

            final_errors.append((mu_T - mu_hat)**2)
            N_track.append(N)

    return np.mean(final_errors), N_track

def generate_heatmap_for_model(model_type="original"):
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    T = 200
    N_init = 400
    grid_size = 50 
    r_s = np.linspace(0.0, 1.0, grid_size)
    q_s = np.linspace(0.0, 1.0, grid_size)
    epsilons = np.geomspace(0.01, 20, 10) 

    print(f"Running simulation for the {model_type.capitalize()} Model ({grid_size}x{grid_size} grid)...")
    best_eps_matrix = np.zeros((len(q_s), len(r_s)))

    for i, q in enumerate(q_s):
        for j, r in enumerate(r_s):
            errors = []
            for eps in epsilons:
                avg_error, _ = simulate_final_step_experiment(
                    epsilon=eps,
                    time_horizon_T=T,
                    initial_participants_N=N_init,
                    join_prob=r,
                    leave_prob=q,
                    model_type=model_type,
                    trials=10, 
                    nmax=20000 
                )
                errors.append(avg_error)
            
            best_eps_matrix[i, j] = epsilons[np.argmin(errors)]

    plt.figure(figsize=(10, 8))
    log_eps_matrix = np.log10(best_eps_matrix)

    mesh = plt.pcolormesh(
        r_s, q_s, log_eps_matrix, 
        shading='auto', cmap='viridis', vmin=-2.0, vmax=0.7
    )

    cbar = plt.colorbar(mesh)
    cbar.set_label(r"$\log_{10}(\epsilon^*)$", fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    model_title = "Original Model" if model_type == "original" else "Agnostic Model"
    plt.title(f"{model_title} Optimal Epsilon Landscape", fontsize=18, pad=15)
    plt.xlabel(r"$r$ (Join Rate)", fontsize=16)
    plt.ylabel(r$q$ (Leave Rate)", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    file_path = os.path.join(output_dir, f"{model_type}_model_heatmap.pdf")
    plt.tight_layout()
    plt.savefig(file_path, format='pdf', dpi=300)
    plt.close()
    
    print(f" -> Saved: {file_path}")
