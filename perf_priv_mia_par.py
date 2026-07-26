
import argparse
import os
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np


# Base parameters (kept identical to the original experiment).
BASE_SEED = 42
d = 12
mu = np.zeros(d)
Sigma = np.eye(d)
N0 = 1000
T = 50
R = 10.0
taus = [0.02]
gammas = np.geomspace(0.2, 20.0, 30)
q_values = np.linspace(0.01, 0.99, 20)
r_values = np.linspace(0.01, 0.99, 20)
trials = 20

# All paths are relative to this script, so the scripts work from any cwd.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"


def resolve_q(q):
    """Return the canonical grid value and index for a requested q."""
    matches = np.flatnonzero(np.isclose(q_values, q, rtol=0.0, atol=5e-9))
    if len(matches) != 1:
        values = ", ".join(f"{value:.8f}" for value in q_values)
        raise ValueError(f"q={q} is not on the configured q grid. Choose one of: {values}")
    index = int(matches[0])
    return float(q_values[index]), index


def run_for_q(q, q_index):
    # A q-specific generator makes results deterministic regardless of launch order.
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, q_index]))
    results_by_tau = {}

    for tau in taus:
        print(f"Processing tau={tau}, q={q:.8f}...")
        r_results = []

        for r in r_values:
            gamma_errors = []
            empirical_C_gammas = []
            Nt_history_all_gammas = {
                float(gamma): np.zeros((trials, T + 1)) for gamma in gammas
            }

            for gamma_value in gammas:
                gamma = float(gamma_value)
                trial_errors = []
                ratios_this_gamma = []

                for trial in range(trials):
                    Nt = N0
                    Nt_history_all_gammas[gamma][trial, 0] = Nt
                    valid_rounds = 0
                    cumulative_error = 0.0

                    for t in range(T):
                        if Nt <= 1:
                            Nt_history_all_gammas[gamma][trial, t + 1 :] = Nt
                            break

                        Z = rng.multivariate_normal(mu, Sigma, Nt)
                        norms = np.linalg.norm(Z, axis=1)
                        clip = np.minimum(1.0, R / np.where(norms == 0, 1e-9, norms))
                        Z_bar = Z * clip[:, None]
                        mu_Z = np.mean(Z_bar, axis=0)

                        noise = rng.normal(0, gamma / np.sqrt(Nt), size=d)
                        o_t = mu_Z + noise
                        cumulative_error += np.sum((o_t - mu) ** 2)
                        valid_rounds += 1

                        term1 = (d / 2) * np.log(Nt / (Nt - 1))
                        diff1 = o_t - (Nt * mu_Z - Z_bar) / (Nt - 1)
                        diff2 = o_t - mu_Z
                        term2 = (Nt - 1) * np.sum(diff1**2, axis=1) / (2 * gamma**2)
                        term3 = Nt * np.sum(diff2**2) / (2 * gamma**2)
                        Lambda = term1 + term2 - term3

                        leaves = np.sum((Lambda > tau) & (rng.random(Nt) < q))
                        joins = np.sum((Lambda <= tau) & (rng.random(Nt) < r))
                        Nt_next = min(20000, max(1, Nt - leaves + joins))
                        ratios_this_gamma.append(Nt_next / Nt)
                        Nt = Nt_next
                        Nt_history_all_gammas[gamma][trial, t + 1] = Nt

                    if valid_rounds > 0:
                        trial_errors.append((cumulative_error / valid_rounds) * T)
                    else:
                        trial_errors.append(np.inf)

                gamma_errors.append(np.mean(trial_errors))
                empirical_C_gammas.append(
                    np.mean(ratios_this_gamma) if ratios_this_gamma else np.nan
                )

            gamma_errors = np.asarray(gamma_errors)
            idx_best = int(np.argmin(gamma_errors))
            Nt_mean_by_gamma = np.stack(
                [
                    np.mean(Nt_history_all_gammas[float(gamma)], axis=0)
                    for gamma in gammas
                ]
            )
            Nt_std_by_gamma = np.stack(
                [
                    np.std(Nt_history_all_gammas[float(gamma)], axis=0)
                    for gamma in gammas
                ]
            )
            print(
                f"tau={tau}, q={q:.2f}, r={r:.2f}: "
                f"best gamma={gammas[idx_best]:.2f}",
                flush=True,
            )
            r_results.append(
                {
                    "r": float(r),
                    "gamma_errors": gamma_errors,
                    "empirical_C_gammas": np.asarray(empirical_C_gammas),
                    "Nt_mean_by_gamma": Nt_mean_by_gamma,
                    "Nt_std_by_gamma": Nt_std_by_gamma,
                }
            )

        results_by_tau[float(tau)] = r_results

    return {
        "format_version": 2,
        "q": q,
        "q_index": q_index,
        "config": {
            "base_seed": BASE_SEED,
            "d": d,
            "N0": N0,
            "T": T,
            "R": R,
            "taus": np.asarray(taus),
            "gammas": gammas,
            "q_values": q_values,
            "r_values": r_values,
            "trials": trials,
        },
        "results_by_tau": results_by_tau,
    }


def save_atomically(result, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", type=float, required=True, help="q value from q_values")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Result directory (default: ./results beside this script)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    q, q_index = resolve_q(args.q)
    started_at = time.perf_counter()
    result = run_for_q(q, q_index)
    destination = args.results_dir.resolve() / f"results_q_{q:.8f}.pkl"
    save_atomically(result, destination)
    elapsed = time.perf_counter() - started_at
    print(
        f"DONE q={q:.8f} in {elapsed / 60:.1f} minutes; saved {destination}",
        flush=True,
    )


if __name__ == "__main__":
    main()

