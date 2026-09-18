# Regression Grid Search & Analysis

This repository contains scripts for running individual regression experiments, launching parallel grid searches, and visualizing the results.

## Usage

### 1. Run a Single Regression
Run an individual regression experiment:
`python single_q_r_regression.py`

### 2. Launch Grid Search
Launch the full grid search across multiple workers:
`python launch_all_grid_regression.py --workers 4`

### 3. Plot Results
Generate a performance/privacy heatmap from your completed grid search:
`python perf_priv_reg_heatmap.py --results-dir /path/to/your/custom_folder` (The default folder in which the .pkl files are saved is /results_regression/ as output from 2.
