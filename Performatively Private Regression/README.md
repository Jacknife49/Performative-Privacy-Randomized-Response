# Regression Grid Search & Analysis

This repository contains scripts for running individual regression experiments, launching parallel grid searches, and visualizing the results.

## Usage

### 1. Run a Single Regression
Run an individual regression experiment:
`python single_q_r_regression.py --q 0.5 --r 0.5`

### 2. Launch Grid Search
Launch the full grid search across multiple workers:
`python launch_all_grid_regression.py --workers 4`

### 3. Plot Results
Generate a performance/privacy heatmap from the completed grid search:
`python perf_priv_reg_heatmap.py --results-dir /path/to/your/custom_folder` (The default folder in which the .pkl files are saved is /results_regression/ as output from 2)

### 4. Hyperparameters & Trial Run Status

Based on the current configuration in `single_q_r_regression.py`, the following parameters and hyperparameters are being used:

**Model Architecture:**
*   **Network:** 2-layer MLP (Input -> 16 units -> ReLU -> 1 unit)
*   **Input Dimension (`d`):** 12

**Training Hyperparameters:**
*   **Optimizer:** SGD
*   **Learning Rate (`lr`):** 0.1
*   **Batch Size:** 32
*   **Epochs:** 3
*   **Loss Function:** MSELoss

**Performative & DP Parameters:**
*   **Initial Population (`N0`):** 1000
*   **Time Steps (`T`):** 50
*   **Clipping Bound (`R`):** 5.0
*   **Vulnerability Threshold (`tau`):** 0.02
*   **Base Seed:** 42
*   **Search Grids:** 3 values for `gamma` (0.2 to 20.0), `q`, and `r` (0.1 to 1.0)

*Note: Please note that this is currently a trial run. As such, the chosen parameters and hyperparameters listed above are fixed directly in the script for now. Because this is an initial test phase, we are not allowing the explicit tuning of things like the learning rate or batch size via command line arguments. If this experiment is successful, we shall incorporate those changes to allow for dynamic tuning in future updates.*
