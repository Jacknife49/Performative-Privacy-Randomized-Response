"""Run the regression experiment for one (q, r) pair and save results.

Example:
    python single_q_r_regression.py --q 0.5 --r 0.5
"""

import argparse
import os
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm

# Base parameters matching the DP-SGD grid space
BASE_SEED = 42
d = 12
N0 = 1000
T = 50
R = 5.0
tau = 0.02
gammas = np.geomspace(0.2, 20.0, 30)
q_values = np.linspace(0.01, 0.99, 30)
r_values = np.linspace(0.01, 0.99, 30)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results_regression"

class RegressionNet(nn.Module):
    def __init__(self, input_dim):
        super(RegressionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_nn_dp_sgd(X, y, gamma, R, epochs=3, batch_size=32, lr=0.1):
    N, d = X.shape
    model = RegressionNet(d)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    final_grad_norms = torch.zeros(N)

    for epoch in range(epochs):
        indices = torch.randperm(N)
        X_shuf, y_shuf = X[indices], y[indices]
        
        for i in range(0, N, batch_size):
            X_batch = X_shuf[i : i + batch_size]
            y_batch = y_shuf[i : i + batch_size]
            B = len(X_batch)
            
            sum_clipped_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
            
            for j in range(B):
                x_j, y_j = X_batch[j:j+1], y_batch[j:j+1]
                
                model.zero_grad()
                pred_j = model(x_j)
                loss_j = criterion(pred_j, y_j)
                loss_j.backward()
                
                grad_norm = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()))
                
                if epoch == epochs - 1:
                    original_idx = indices[i + j]
                    final_grad_norms[original_idx] = grad_norm.item()
                
                clip_factor = max(1.0, grad_norm.item() / (R + 1e-10))
                
                for name, p in model.named_parameters():
                    sum_clipped_grads[name] += (p.grad / clip_factor)
            
            optimizer.zero_grad()
            for name, param in model.named_parameters():
                noise = torch.normal(mean=0.0, std=R * gamma, size=param.size())
                param.grad = (sum_clipped_grads[name] + noise) / B
                
            optimizer.step()
            
    return model, final_grad_norms.numpy()

def simulate_nn_performative_trajectory(gamma, q, r, N0=1000, T=50, tau=0.02, R=5.0, d=12):
    N_t = N0
    total_loss = 0.0
    theta_true = np.random.randn(d)
    
    def generate_data(n_samples):
        X_np = np.random.randn(n_samples, d)
        y_np = np.sin(X_np[:, 0]) + X_np.dot(theta_true) + np.random.randn(n_samples) * 0.5
        return torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
    
    X_test, y_test = generate_data(1000)
    
    for t in range(T):
        if N_t <= 1:
            total_loss += 1e6 * (T - t) 
            break
            
        X_train, y_train = generate_data(N_t)
        model, grad_norms = train_nn_dp_sgd(X_train, y_train, gamma, R, epochs=3)
        
        with torch.no_grad():
            preds_test = model(X_test)
            iter_loss = nn.MSELoss()(preds_test, y_test).item()
            total_loss += iter_loss
            
            preds_train = model(X_train)
            losses = nn.MSELoss(reduction='none')(preds_train, y_train).squeeze().numpy()
        
        influence = grad_norms / (R * gamma + 1e-5)
        mu_in = np.mean(losses) - influence
        mu_out = np.mean(losses) + influence
        sigma_loss = np.std(losses) + 0.1
        
        log_p_in = norm.logpdf(losses, loc=mu_in, scale=sigma_loss)
        log_p_out = norm.logpdf(losses, loc=mu_out, scale=sigma_loss)
        Lambda = log_p_in - log_p_out
        
        vuln_mask = Lambda > tau
        safe_mask = ~vuln_mask
        num_vuln = np.sum(vuln_mask)
        num_safe = np.sum(safe_mask)
        
        departures = np.random.binomial(num_vuln, q)
        recruits = np.random.binomial(num_safe, r)
        
        N_t = min(20000, max(1, N_t - departures + recruits))
        
    return total_loss, N_t

def optional_grid_index(value, grid):
    matches = np.flatnonzero(np.isclose(grid, value, rtol=0.0, atol=5e-9))
    return int(matches[0]) if len(matches) == 1 else None

def float_seed_words(value):
    bits = int(np.float64(value).view(np.uint64))
    return [bits & 0xFFFFFFFF, bits >> 32]

def run_for_pair(q, r):
    q_index = optional_grid_index(q, q_values)
    r_index = optional_grid_index(r, r_values)

    # Establish deterministic seeds for Torch and Numpy based on (q,r)
    seed_words = [BASE_SEED, *float_seed_words(q), *float_seed_words(r)]
    combined_seed = abs(hash(tuple(seed_words))) % (2**32 - 1)
    np.random.seed(combined_seed)
    torch.manual_seed(combined_seed)

    gamma_errors = []
    final_Nts = []

    for gamma_value in gammas:
        gamma = float(gamma_value)
        print(f"Processing q={q:.8f}, r={r:.8f}, gamma={gamma:.2f}...", flush=True)
        
        loss, final_Nt = simulate_nn_performative_trajectory(gamma, q, r, N0=N0, T=T, tau=tau, R=R, d=d)
        gamma_errors.append(loss)
        final_Nts.append(final_Nt)

    gamma_errors = np.asarray(gamma_errors)
    final_Nts = np.asarray(final_Nts)
    idx_best = int(np.argmin(gamma_errors))
    
    print(f"Best gamma={gammas[idx_best]:.2f} for (q={q:.2f}, r={r:.2f})", flush=True)

    return {
        "format_version": 1,
        "q": q,
        "q_index": q_index,
        "r": r,
        "r_index": r_index,
        "config": {
            "base_seed": BASE_SEED,
            "d": d,
            "N0": N0,
            "T": T,
            "R": R,
            "tau": tau,
            "gammas": gammas,
            "q_values": q_values,
            "r_values": r_values,
        },
        "results": {
            "gamma_errors": gamma_errors,
            "final_Nts": final_Nts,
            "best_gamma": float(gammas[idx_best]),
            "best_loss": float(gamma_errors[idx_best])
        }
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
    parser.add_argument("--q", type=float, required=True, help="q value")
    parser.add_argument("--r", type=float, required=True, help="r value")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Result directory (default: ./results_regression beside this script)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    q = float(args.q)
    r = float(args.r)
    started_at = time.perf_counter()
    result = run_for_pair(q, r)
    destination = args.results_dir.resolve() / f"result_q_{q:.8f}_r_{r:.8f}.pkl"
    save_atomically(result, destination)
    elapsed = time.perf_counter() - started_at
    print(f"DONE q={q:.8f}, r={r:.8f} in {elapsed / 60:.1f} minutes; saved {destination}", flush=True)

if __name__ == "__main__":
    main()
