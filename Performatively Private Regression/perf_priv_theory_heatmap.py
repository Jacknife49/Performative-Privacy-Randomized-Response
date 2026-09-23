import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar

# ============================================================
# Plot Aesthetics
# ============================================================
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "image.cmap": "magma",
    "axes.grid": False
})

# ============================================================
# System Constants (Matched to Empirical Parameters)
# ============================================================
N0 = 1000                  # Population size
N = int(N0)
T = 50                     # Finite time horizon
d = 12                     # Data dimension
tau = 0.02                 # Privacy threshold
R = 10.0                   # L2 feature clipping radius
Y = 10.0                   # Label clipping bound (safety threshold for theory)

# Exact empirical values
RIDGE_EPS = 1e-8           # Tiny ridge for numerical stability of (X^T X)^{-1}
LEVERAGE_DENOM_FLOOR = 1e-9 # Guards the (1 - h_i) leave-one-out denominator
sigma_eps = 1.0            # std dev of the regression noise eps_i
tr_sigma = d               # Tr(Sigma_x) for Sigma_x = I
nu = d * (sigma_eps**2)    # Constant bounding non-private estimation error

# ============================================================
# Monte Carlo Parameters
# ============================================================
np.random.seed(42)

# Total number of focal records sampled to evaluate the expectation
NUM_SAMPLES = 50_000

# ============================================================
# Helper: L2 Clip
# ============================================================
def clip_l2(X, radius):
    """
    Clip each row/vector of X to have L2 norm at most 'radius'.
    """
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    scale = np.minimum(1.0, radius / np.maximum(norms, 1e-12))
    return X * scale

# ============================================================
# Exact Dataset-Level Monte Carlo for Ridge Regression
# ============================================================
effective_signals = np.empty(NUM_SAMPLES, dtype=np.float64)
num_generated = 0
batches = max(1, NUM_SAMPLES // N)

# True underlying parameter explicitly set to zeros matching empirical
theta_star = np.zeros(d)

print(f"Generating {NUM_SAMPLES:,} Monte Carlo samples for the OLS effective signals...")

for _ in range(batches):
    if num_generated >= NUM_SAMPLES:
        break
        
    # 1. Generate synthetic dataset matching empirical distribution
    X = np.random.randn(N, d)
    noise = np.random.randn(N) * sigma_eps
    y = X @ theta_star + noise

    # 2. Clip features and labels
    X_bar = clip_l2(X, R)
    y_bar = np.clip(y, -Y, Y)

    # 3. Compute Suff Stats (using RIDGE_EPS for stability instead of heavy regularization)
    A = X_bar.T @ X_bar + RIDGE_EPS * np.eye(d)
    A_inv = np.linalg.inv(A)
    theta = A_inv @ (X_bar.T @ y_bar)

    # 4. Leverage scores and residuals
    h = np.sum((X_bar @ A_inv) * X_bar, axis=1)
    e = y_bar - X_bar @ theta

    # 5. Effective Signal Si = ||Delta theta||_2
    M = np.sum((X_bar @ A_inv)**2, axis=1)
    
    # Safely compute S_i using empirical LEVERAGE_DENOM_FLOOR
    safe_denominator = np.maximum(1.0 - h, LEVERAGE_DENOM_FLOOR)
    S = (np.abs(e) * np.sqrt(M)) / safe_denominator

    # Save to global array
    start = num_generated
    end = min(start + N, NUM_SAMPLES)
    size = end - start
    effective_signals[start:end] = S[:size]
    num_generated += size

# Remove numerically zero values to prevent division by zero in the CDF
effective_signals = effective_signals[effective_signals > 1e-12]

print(f"Generated {len(effective_signals):,} valid signals for DP-OLS.")

# ============================================================
# Marginal Leakage Probability (DP-OLS)
# ============================================================
def p_tau_mc(gamma):
    """
    Computes the marginal leakage probability for DP-OLS.
    """
    S = effective_signals

    term1 = (tau * gamma) / (np.sqrt(N) * S)
    term2 = (np.sqrt(N) * S) / (2.0 * gamma)

    leakage_probability = 1.0 - norm.cdf(term1 - term2)

    return np.mean(leakage_probability)


# ============================================================
# Objective Function (Excess Test Risk)
# ============================================================
def cumulative_error(gamma, q, r):
    """
    Finite-horizon closed-form objective for DP-OLS.
    Minimizes cumulative expected excess test loss (MSE).
    """
    p_t = p_tau_mc(gamma)
    C_gamma = 1.0 + r - (q + r) * p_t
    num = (gamma**2 * tr_sigma) + nu

    if abs(C_gamma - 1.0) < 1e-8:
        return (num * T) / N

    denom = N * (C_gamma - 1.0)
    geometric_factor = 1.0 - (1.0 / (C_gamma ** T))

    return (num / denom) * geometric_factor


# ============================================================
# Grid Search Optimization
# ============================================================
GRID_RESOLUTION = 30

# Using exactly 30 resolution matching empirical setup
q_values = np.linspace(0.01, 0.99, GRID_RESOLUTION)
r_values = np.linspace(0.01, 0.99, GRID_RESOLUTION)

optimal_gammas = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION))

print(f"Running numerical optimization over {GRID_RESOLUTION}x{GRID_RESOLUTION} grid...")

for i, q in enumerate(q_values):
    for j, r in enumerate(r_values):
        
        # Optimize gamma explicitly bounded to the empirical geomspace bounds (0.2, 20.0)
        result = minimize_scalar(
            cumulative_error,
            bounds=(0.2, 20.0), 
            args=(q, r),
            method="bounded"
        )

        optimal_gammas[i, j] = result.x
        print(f"Optimal gamma for (q, r) = ({q:.2f}, {r:.2f}): {result.x:.4f}")


# ============================================================
# Plot Heatmap
# ============================================================
plt.figure(figsize=(9, 7))

heatmap = plt.pcolormesh(
    q_values,
    r_values,
    optimal_gammas.T,
    shading="nearest",
    cmap="viridis",
    vmin=0.2, 
    vmax=12  # Matched to referenced file colorbar scaling
)

cbar = plt.colorbar(heatmap)
cbar.set_label(
    r"Optimal $\gamma$",
    rotation=270,
    labelpad=20,
    fontsize=14
)
cbar.ax.tick_params(labelsize=11)

plt.title(
    rf"Optimal Noise Scale $\gamma$ ($\tau = {tau}$)",
    fontsize=16,
    pad=15
)

plt.xlabel("Leave probability q", fontsize=14)
plt.ylabel("Join probability r", fontsize=14)

plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.tight_layout()

plt.savefig("optimal_gamma_heatmap_ols.png", dpi=300, bbox_inches="tight")
plt.show()

print("Optimization complete. Heatmap saved as 'optimal_gamma_heatmap_ols_matched.png'.")
