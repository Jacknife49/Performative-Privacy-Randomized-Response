import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

# --------------------------------------------------
# Matplotlib settings (paper friendly)
# --------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

# --------------------------------------------------
# Read files
# --------------------------------------------------
with open("practice.txt") as f:
    practice_text = f.read()

with open("theory.txt") as f:
    theory_text = f.read()

# --------------------------------------------------
# Parse files
# --------------------------------------------------
practice_pattern = re.compile(
    r"q = ([0-9.]+)\s+and r = ([0-9.]+), min error gamma is ([0-9.]+)"
)

theory_pattern = re.compile(
    r"\(q, r\)\s*=\s*\(([0-9.]+),\s*([0-9.]+)\):\s*([0-9.]+)"
)

practice = [
    (float(q), float(r), float(g))
    for q, r, g in practice_pattern.findall(practice_text)
]

theory = [
    (float(q), float(r), float(g))
    for q, r, g in theory_pattern.findall(theory_text)
]

df_pr = pd.DataFrame(practice, columns=["q", "r", "gamma"])
df_th = pd.DataFrame(theory, columns=["q", "r", "gamma"])

# --------------------------------------------------
# Heatmap data
# --------------------------------------------------
pivot_pr = df_pr.pivot(index="r", columns="q", values="gamma")
pivot_th = df_th.pivot(index="r", columns="q", values="gamma")

vmin = min(df_pr.gamma.min(), df_th.gamma.min())
vmax = max(df_pr.gamma.max(), df_th.gamma.max())

# ==================================================
# Figure 1 : Practice heatmap
# ==================================================
fig, ax = plt.subplots(figsize=(3.4, 3.0))

im = ax.imshow(
    pivot_pr.values,
    origin="lower",
    aspect="auto",
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
    extent=[
        pivot_pr.columns.min(),
        pivot_pr.columns.max(),
        pivot_pr.index.min(),
        pivot_pr.index.max(),
    ],
)

ax.set_xlabel("Leave probability ($q$)")
ax.set_ylabel("Join probability ($r$)")
ax.set_title("Practice")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"$\gamma^\star$")

plt.tight_layout()
plt.savefig("practice_heatmap.png", dpi=400)

# ==================================================
# Figure 2 : Theory heatmap
# ==================================================
fig, ax = plt.subplots(figsize=(3.4, 3.0))

im = ax.imshow(
    pivot_th.values,
    origin="lower",
    aspect="auto",
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
    extent=[
        pivot_th.columns.min(),
        pivot_th.columns.max(),
        pivot_th.index.min(),
        pivot_th.index.max(),
    ],
)

ax.set_xlabel("Leave probability ($q$)")
ax.set_ylabel("Join probability ($r$)")
ax.set_title("Theory")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"$\gamma^\star$")

plt.tight_layout()
plt.savefig("theory_heatmap.png", dpi=400)

# ==================================================
# Figure 3 : Line comparison
# ==================================================
fig, ax = plt.subplots(figsize=(3.6, 3.0))

# Plot only representative q values
selected_q = selected_q = [0.01, 0.18, 0.35, 0.52, 0.69]

cmap = plt.cm.viridis
norm = Normalize(
    vmin=min(df_th.q.unique()),
    vmax=max(df_th.q.unique())
)

for q in selected_q:

    theory_curve = (
        df_th[np.isclose(df_th.q, q)]
        .sort_values("r")
    )

    practice_curve = (
        df_pr[np.isclose(df_pr.q, q)]
        .sort_values("r")
    )

    color = cmap(norm(q))

    ax.plot(
        theory_curve.r,
        theory_curve.gamma,
        color=color,
        lw=2,
    )

    ax.plot(
        practice_curve.r,
        practice_curve.gamma,
        color=color,
        ls="--",
        lw=2,
    )

ax.set_yscale("log")

ax.set_xlabel("Join probability ($r$)")
ax.set_ylabel(r"Optimal $\gamma$")

ax.grid(alpha=0.3)

legend = [
    Line2D([0], [0], color="black", lw=2, label="Theory"),
    Line2D([0], [0], color="black", lw=2, ls="--", label="Practice"),
]

ax.legend(
    handles=legend,
    loc="upper right",
    frameon=True,
)

sm = plt.cm.ScalarMappable(
    cmap=cmap,
    norm=norm,
)
sm.set_array([])

cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Leave probability ($q$)")

plt.tight_layout()
plt.savefig("gamma_line_comparison_tau_0.01.png", dpi=400)

plt.show()