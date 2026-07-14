# LDP Population Dynamics & Error Simulation

A Python simulation framework for analyzing Local Differential Privacy (LDP) estimation errors within dynamic populations. This tool models scenarios where participants join and leave an LDP protocol over time, tracking population evolution and identifying the optimal privacy budget ($\epsilon$) that minimizes estimation error at a final time horizon.

## Features

* **Dynamic Population Modeling:** Simulates users joining (rate $r$) and leaving (rate $q$) across hundreds of time steps.
* **Dual Protocol Models:**
* **Original Model:** New participants join based on the *unleaked* population count.
* **Agnostic Model:** New participants join based on the *total* population count.
* **Total/Unleaked Growth Models:** Configurable parameters to test variations in population dynamics.


* **Optimal $\epsilon$ Heatmaps:** Performs a grid search across $r$ and $q$ to find the privacy budget that minimizes mean squared error, visualized as a high-resolution log-scaled heatmap.
* **Variance Tracking:** Calculates moving-average variance over time, providing empirical standard error confidence bands.

## Project Structure

The codebase is split into three modular files:

* `main.py`: The central orchestrator. Run this file to execute the full suite of simulations. It automatically triggers the heatmap generation and outputs three specific population evolution comparisons ($q < r$, $q > r$, $q = r$).
* `heatmap_generator.py`: Contains the logic for the final-step LDP experiment. It tests a geometric space of $\epsilon$ values over a 50x50 grid of leave/join rates and generates `matplotlib` colormesh heatmaps.
* `population_dynamics.py`: Handles the step-by-step time-series tracking. It records the population size ($N_t$) and absolute error history across multiple trials, generating time-series plots with overlaid $\epsilon$ color gradients.

## Prerequisites

Ensure you have Python 3.8+ installed. You will need the following libraries:

```bash
pip install numpy matplotlib

```

## Usage

Place all three `.py` files in the same directory. To generate the plots, simply execute the main script:

```bash
python main.py

```

### Outputs

The script will automatically create a `plots/` directory in your current working folder. Upon a successful run, you will find the following PDF files:

1. **`agnostic_model_heatmap.pdf`** (or `original_model_heatmap.pdf`): A 2D colormap showing the optimal $\log_{10}(\epsilon^*)$ for every combination of $q$ and $r$.
2. **`pop_history_q_less_than_r.pdf`**: Time-series plot showing exponential/steady population growth.
3. **`pop_history_q_greater_than_r.pdf`**: Time-series plot showing population decay.
4. **`pop_history_q_equals_r.pdf`**: Time-series plot showing equilibrium dynamics.

## Customization

* **Switching Models:** To generate the heatmap for the "original" model instead of the "agnostic" model, open `main.py` and change `chosen_model = "agnostic"` to `chosen_model = "original"`.
* **Simulation Length:** By default, the time horizon ($T$) is set to 200. You can adjust this in the parameter lists within `main.py` or the individual module defaults.
* **Resolution:** If the heatmap takes too long to generate, reduce `grid_size = 50` in `heatmap_generator.py` to a lower number like 20.
