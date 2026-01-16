import numpy as np
import matplotlib.pyplot as plt
import os
import re

# =========================
# Plot style (idéntico al tuyo)
# =========================
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 1.1,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
})

# =========================
# Configuración
# =========================
RESULTS_DIR = "results_teresa"
EDO = 3
MAP = "Chebyschev_tower"

SAVE_FIG = True
OUT_FNAME = f"comparison_layers_{MAP}_edo{EDO}.png"

# =========================
# Cargar archivos
# =========================
pattern = re.compile(
    rf"pred_L(\d+)_Q(\d+)_{MAP}_edo{EDO}\.npz"
)

results = []

for fname in os.listdir(RESULTS_DIR):
    match = pattern.match(fname)
    if match:
        L = int(match.group(1))
        Q = int(match.group(2))
        data = np.load(os.path.join(RESULTS_DIR, fname))
        results.append({
            "L": L,
            "Q": Q,
            "x": data["x"],
            "pred": data["pred"],
            "ref": data["ref"]
        })

# Ordenar por número de capas
results = sorted(results, key=lambda r: r["L"])

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(8, 4))

# Solución exacta (una sola vez)
ax.plot(
    results[0]["x"],
    results[0]["ref"],
    color="black",
    linewidth=2.2,
    label="Exact solution"
)

# Predicciones QPINN
for r in results:
    ax.plot(
        r["x"],
        r["pred"],
        linestyle="--",
        linewidth=1.3,
        label=rf"$L={r['L']},\ Q={r['Q']}$"
    )

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$u(x)$")
ax.set_title(
    rf"QPINN solutions vs exact — {MAP}, EDO {EDO}"
)
ax.legend(ncol=2)
ax.grid(True)

if SAVE_FIG:
    fig.savefig(
        os.path.join(RESULTS_DIR, OUT_FNAME),
        bbox_inches="tight",
        dpi=300
    )

plt.show()
plt.close(fig)
