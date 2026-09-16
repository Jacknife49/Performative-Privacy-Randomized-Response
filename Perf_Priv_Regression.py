import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# 1. The Neural Network
# ==========================================
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

# ==========================================
# 2. True DP-SGD Training Function
# ==========================================
def train_nn_dp_sgd(X, y, gamma, R, epochs=3, batch_size=32, lr=0.1):
    N, d = X.shape
    model = RegressionNet(d)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Track the gradient norm of each individual during the final epoch
    # to use in the LiRA attack (influence proxy).
    final_grad_norms = torch.zeros(N)

    for epoch in range(epochs):
        indices = torch.randperm(N)
        X_shuf, y_shuf = X[indices], y[indices]
        
        for i in range(0, N, batch_size):
            X_batch = X_shuf[i : i + batch_size]
            y_batch = y_shuf[i : i + batch_size]
            B = len(X_batch)
            
            sum_clipped_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
            
            # Compute PER-SAMPLE gradients manually
            for j in range(B):
                x_j, y_j = X_batch[j:j+1], y_batch[j:j+1]
                
                model.zero_grad()
                pred_j = model(x_j)
                loss_j = criterion(pred_j, y_j)
                loss_j.backward()
                
                # Compute L2 norm of this individual's gradient
                grad_norm = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()))
                
                # If it's the last epoch, save it for the LiRA evaluation
                if epoch == epochs - 1:
                    original_idx = indices[i + j]
                    final_grad_norms[original_idx] = grad_norm.item()
                
                # Clip
                clip_factor = max(1.0, grad_norm.item() / (R + 1e-10))
                
                for name, p in model.named_parameters():
                    sum_clipped_grads[name] += (p.grad / clip_factor)
            
            # Add DP Noise and Step
            optimizer.zero_grad()
            for name, param in model.named_parameters():
                noise = torch.normal(mean=0.0, std=R * gamma, size=param.size())
                param.grad = (sum_clipped_grads[name] + noise) / B
                
            optimizer.step()
            
    return model, final_grad_norms.numpy()

# ==========================================
# 3. The Explicit Performative Simulation
# ==========================================
def simulate_nn_performative_trajectory(gamma, q, r, N0=1000, T=50, tau=0.02, R=5.0, d=12):
    N_t = N0
    total_loss = 0.0
    
    # Ground truth (Non-linear so the NN actually has to work)
    theta_true = np.random.randn(d)
    
    def generate_data(n_samples):
        X_np = np.random.randn(n_samples, d)
        # y = sin(x1) + X*theta + noise
        y_np = np.sin(X_np[:, 0]) + X_np.dot(theta_true) + np.random.randn(n_samples) * 0.5
        return torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
    
    # Pristine test set for true generalization error
    X_test, y_test = generate_data(1000)
    
    for t in range(T):
        if N_t <= 1:
            total_loss += 1e6 * (T - t) 
            break  # Collapse
            
        X_train, y_train = generate_data(N_t)
        
        # 1. Train the NN using real DP-SGD
        model, grad_norms = train_nn_dp_sgd(X_train, y_train, gamma, R, epochs=3)
        
        # 2. Evaluate Generalization Error on Test Set
        with torch.no_grad():
            preds_test = model(X_test)
            iter_loss = nn.MSELoss()(preds_test, y_test).item()
            total_loss += iter_loss
            
            # 3. Evaluate LiRA (Likelihood Ratio) for the Training Set
            preds_train = model(X_train)
            losses = nn.MSELoss(reduction='none')(preds_train, y_train).squeeze().numpy()
        
        # The influence the person had on the NN parameters
        influence = grad_norms / (R * gamma + 1e-5)
        
        # Simulated Shadow Models (f_in and f_out densities)
        mu_in = np.mean(losses) - influence  # Inclusion lowers their loss
        mu_out = np.mean(losses) + influence # Exclusion raises their loss
        sigma_loss = np.std(losses) + 0.1
        
        log_p_in = norm.logpdf(losses, loc=mu_in, scale=sigma_loss)
        log_p_out = norm.logpdf(losses, loc=mu_out, scale=sigma_loss)
        Lambda = log_p_in - log_p_out
        
        # 4. Performative Branching (Coin Flips)
        vuln_mask = Lambda > tau
        safe_mask = ~vuln_mask
        
        num_vuln = np.sum(vuln_mask)
        num_safe = np.sum(safe_mask)
        
        departures = np.random.binomial(num_vuln, q)
        recruits = np.random.binomial(num_safe, r)
        
        N_t = min(20000, max(1, N_t - departures + recruits))
        
    return total_loss, N_t

# ==========================================
# 4. Multiprocessing Worker Function
# ==========================================
def _search_optimal_gamma_worker(args):
    """
    Worker function executed by each CPU core independently.
    """
    i, q, j, r, gamma_space, tau = args
    best_loss = np.inf
    best_gamma = gamma_space[-1]
    
    # Dictionary to store the (total_loss, final_Nt) for each gamma evaluated
    gamma_stats = {}
    
    # Set a unique seed per worker to ensure diversity in parallel execution
    np.random.seed(int((i+1) * (j+1) * 1000) % (2**32 - 1))
    torch.manual_seed(int((i+1) * (j+1) * 1000) % (2**32 - 1))
    
    for gamma in gamma_space:
        # Unpack both the loss and the final population size
        loss, final_Nt = simulate_nn_performative_trajectory(gamma, q, r, T=50, tau=tau)
        
        # Record the stats for this specific gamma
        gamma_stats[gamma] = (loss, final_Nt)
        
        if loss < best_loss:
            best_loss = loss
            best_gamma = gamma
            
    # Format the losses and N_t into a clean string for the console
    stats_str = ", ".join([f"γ={g:.2f}: [L={l:.1f}, N={n}]" for g, (l, n) in gamma_stats.items()])
    
    # Print the final summary for this (q, r) combination
    print(f"Worker (q={q:.2f}, r={r:.2f}) finished | Optimal γ*: {best_gamma:.2f} | Stats -> {stats_str}")
    
    return (i, j, best_gamma)
# ==========================================
# 5. Main Parallelized Grid Search
# ==========================================
def run_nn_heatmap_parallel():
    grid_size = 5 # 5x5 grid = 25 scenarios. Increase for higher resolution.
    tau = 0.02
    
    q_vals = np.linspace(0.1, 1.0, grid_size)
    r_vals = np.linspace(0.1, 1.0, grid_size)
    gamma_space = np.geomspace(0.2, 20.0, 5) # Evaluates 5 gamma values per grid point
    
    optimal_gamma_matrix = np.zeros((grid_size, grid_size))
    
    # 1. Package the tasks
    tasks = []
    for i, q in enumerate(q_vals):
        for j, r in enumerate(r_vals):
            tasks.append((i, q, j, r, gamma_space, tau))
            
    num_cores = multiprocessing.cpu_count() - 2
    print(f"Initializing Parallel Processing...")
    print(f"Distributing {len(tasks)} grid tasks across {num_cores} CPU cores.")
    print(f"Total model trainings required: {len(tasks) * len(gamma_space)} models * 10 rounds.")
    
    # 2. Execute pool
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = executor.map(_search_optimal_gamma_worker, tasks)
        
    # 3. Unpack results into the matrix
    for i, j, best_gamma in results:
        optimal_gamma_matrix[i, j] = best_gamma

    # 4. Plot the Heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        optimal_gamma_matrix, 
        xticklabels=np.round(r_vals, 2), 
        yticklabels=np.round(q_vals, 2),
        cmap="magma",
        cbar_kws={'label': r'Empirical Optimal DP Noise ($\gamma^*$)'}
    )
    ax.invert_yaxis() 
    plt.title(f'True NN DP-SGD: Optimal $\gamma^*$ ($\\tau$ = {tau})', fontsize=12)
    plt.ylabel('Departure Probability ($q$)', fontsize=11)
    plt.xlabel('Recruitment Probability ($r$)', fontsize=11)
    plt.tight_layout()
    
    # Save the figure so you can import it into your LaTeX poster
    plt.savefig(f'practice_heatmap_{tau}.jpeg', dpi=300)
    print(f"Plot saved successfully as 'practice_heatmap_{tau}.jpeg'")
    plt.show()

# ==========================================
# CRITICAL: Entry point required for multiprocessing
# ==========================================
if __name__ == "__main__":
    # Ensure PyTorch multiprocessing compatibility on Windows/macOS
    multiprocessing.freeze_support() 
    run_nn_heatmap_parallel()
