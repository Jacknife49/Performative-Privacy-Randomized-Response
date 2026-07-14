import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def compute_beta_from_eps(eps):
    return 0.5 * np.tanh(eps / 2.0)

def apply_randomized_response(states, beta, gen):
    return np.where(gen.random(len(states)) < (0.5 + beta), states, 1 - states)

def estimate_unbiased_prevalence(responses, beta):
    return (np.mean(responses) - (0.5 - beta)) / (2 * beta)

def causal_moving_average(data, window):
    cum_sum = np.cumsum(data)
    res = np.zeros_like(data, dtype=float)
    for t in range(len(data)):
        if t < window:
            res[t] = cum_sum[t] / (t + 1)
        else:
            res[t] = (cum_sum[t] - cum_sum[t - window]) / window
    return res

def run_population_simulation(
    growth_model, eps_array, max_time_steps,
    leave_rate, join_rate, initial_pop_size,
    num_trials=20, seed_val=0, min_pop_size=1,
    bound_estimates=True, MAX_POPULATION=20000 
):
    random_gen = np.random.default_rng(seed_val)
    beta_array = compute_beta_from_eps(eps_array)

    pop_history = np.zeros((num_trials, eps_array.size, max_time_steps), dtype=float)
    abs_err_history = np.zeros((num_trials, eps_array.size, max_time_steps), dtype=float)
    cum_err_history = np.zeros((num_trials, eps_array.size, max_time_steps), dtype=float)

    for trial in range(num_trials):
        for idx_eps, beta_val in enumerate(beta_array):
            current_pop = int(initial_pop_size)
            running_err = 0.0

            for step in range(max_time_steps):
                actual_prevalence = random_gen.random()
                actual_states = (random_gen.random(current_pop) < actual_prevalence).astype(int)
                noisy_responses = apply_randomized_response(actual_states, beta_val, random_gen)

                prevalence_est = estimate_unbiased_prevalence(noisy_responses, beta_val)
                if bound_estimates:
                    prevalence_est = np.clip(prevalence_est, 0.0, 1.0)

                current_abs_err = float(abs(prevalence_est - actual_prevalence))
                running_err += current_abs_err

                pop_history[trial, idx_eps, step] = current_pop
                abs_err_history[trial, idx_eps, step] = current_abs_err
                cum_err_history[trial, idx_eps, step] = running_err

                exposed_mask = (actual_states == noisy_responses)
                num_exposed = int(np.sum(exposed_mask))
                num_hidden = current_pop - num_exposed

                departures = random_gen.binomial(num_exposed, leave_rate) if num_exposed > 0 else 0

                if growth_model == "unleaked":
                    arrivals = random_gen.binomial(num_hidden, join_rate) if num_hidden > 0 else 0
                elif growth_model == "total":
                    arrivals = random_gen.binomial(current_pop, join_rate) if current_pop > 0 else 0
                else:
                    raise ValueError("growth_model must be 'unleaked' or 'total'.")

                current_pop = current_pop - departures + arrivals
                current_pop = min(max(min_pop_size, int(current_pop)), MAX_POPULATION)

    return beta_array, pop_history, abs_err_history, cum_err_history

def plot_population_history_only(
    leave_rate=0.3, join_rate=0.6, growth_model="total",
    max_time_steps=200, initial_pop_size=400, num_trials=30,
    output_filename="population_history.pdf"
):
    print(f" -> Generating plot (q={leave_rate}, r={join_rate})...")
    eps_array = np.geomspace(0.01, 5, 10)

    _, pop_history, _, _ = run_population_simulation(
        growth_model, eps_array, max_time_steps, leave_rate, join_rate,
        initial_pop_size, num_trials=num_trials, seed_val=42
    )

    mean_pop_history = np.mean(pop_history, axis=0)

    time_steps = np.arange(max_time_steps)
    fig, ax = plt.subplots(figsize=(8, 5))

    norm = LogNorm(vmin=float(np.min(eps_array)), vmax=float(np.max(eps_array)))
    cmap = plt.cm.viridis
    color_mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    color_mapper.set_array([])

    for i, eps in enumerate(eps_array):
        ax.plot(time_steps, mean_pop_history[i], color=cmap(norm(eps)), alpha=0.8)

    ax.set_title(rf"Population History for $r$={join_rate}, $q$={leave_rate}")
    ax.set_xlabel(r"Time step $t$", fontsize=12)
    ax.set_ylabel(r"$N_t$ (Averaged across trials)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    cbar = fig.colorbar(color_mapper, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\varepsilon$ (log scale)", fontsize=12)

    fig.tight_layout()
    fig.savefig(output_filename, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {output_filename}")
