"""Step 11f - Supplementary Figure S1 (confusion matrices, recalibrated probabilities, threshold 0.5) and
Supplementary Figure S9 (calibration by deciles of predicted probability, before and after recalibration).

Input : results/probabilidades_test.csv (step 04) and results/probabilidades_test_recal.csv (step 06)
Output: figures/FiguraS1_matrices_confusion.png, figures/FiguraS9_calibracion_antes_despues.png
"""
import locale

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES, RESULTS

try:
    locale.setlocale(locale.LC_NUMERIC, "es_ES.UTF-8")
except locale.Error:
    pass
plt.rcParams.update({"font.family": "Arial", "axes.formatter.use_locale": True})

original = pd.read_csv(RESULTS / "probabilidades_test.csv")
recal = pd.read_csv(RESULTS / "probabilidades_test_recal.csv")
assert (original["y_test"] == recal["y_test"]).all()
y = recal["y_test"].to_numpy()
MODELOS = ["TabPFN", "XGBoost", "Random Forest", "SVM"]
COLORES = ["#1f6fb2", "#d7425a", "#26a08c", "#f0a030"]


def coma(v, dec=2):
    return f"{v:.{dec}f}".replace(".", ",")


# ---- Figure S1: confusion matrices
fig, ax = plt.subplots(2, 2, figsize=(8, 7.2))
for a, m in zip(ax.ravel(), MODELOS):
    pred = (recal[m].to_numpy() >= 0.5).astype(int)
    cm = np.array([[((y == 0) & (pred == 0)).sum(), ((y == 0) & (pred == 1)).sum()],
                   [((y == 1) & (pred == 0)).sum(), ((y == 1) & (pred == 1)).sum()]])
    a.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
    for i in range(2):
        for j in range(2):
            a.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=15,
                   color="white" if cm[i, j] > cm.max() * 0.55 else "black")
    sens, esp = cm[1, 1] / cm[1].sum(), cm[0, 0] / cm[0].sum()
    a.set_title(f"{m}\nSensibilidad {coma(sens)} · Especificidad {coma(esp)}", fontsize=10)
    a.set_xticks([0, 1], ["Sin RI", "Con RI"])
    a.set_yticks([0, 1], ["Sin RI", "Con RI"])
    a.set_xlabel("Clase predicha")
    a.set_ylabel("Clase observada")
plt.tight_layout()
plt.savefig(FIGURES / "FiguraS1_matrices_confusion.png", dpi=300, bbox_inches="tight")
plt.close()

# ---- Figure S9: calibration by deciles, before and after recalibration
def deciles(p):
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").mean()
    return g["p"], g["y"]


fig, ax = plt.subplots(1, 4, figsize=(15, 4.3), sharey=True)
for a, m, color in zip(ax, MODELOS, COLORES):
    a.plot([0, 1], [0, 1], color="#cccccc", lw=1)
    x, yy = deciles(original[m].to_numpy())
    a.plot(x, yy, "--o", color="gray", ms=4, lw=1.6, label="Sin recalibrar")
    x, yy = deciles(recal[m].to_numpy())
    a.plot(x, yy, "-o", color=color, ms=4, lw=2, label="Recalibrado")
    a.set_title(m)
    a.set_xlabel("Probabilidad media predicha (decil)")
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.legend(frameon=False, loc="upper left", fontsize=9)
    a.spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("Proporción observada de RI")
plt.tight_layout()
plt.savefig(FIGURES / "FiguraS9_calibracion_antes_despues.png", dpi=300, bbox_inches="tight")
plt.close()
print("Figures S1 and S9 saved in", FIGURES)
