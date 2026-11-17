"""Figuras S4 (curvas de codo) y S5 (compromiso parsimonia-desempeño) con las importancias SHAP en probabilidad."""
from pathlib import Path
import locale
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.colors as mcolors
import numpy as np, pandas as pd
from config import ROOT as BASE, DATA, DATA as D, MODELS, RESULTS as R, FIGURES as F, FIGURES as FIG


try:
    locale.setlocale(locale.LC_NUMERIC, "es_ES.UTF-8")   # decimal comma in the figures (optional)
except locale.Error:
    pass
plt.rcParams.update({"font.family": "Arial", "axes.formatter.use_locale": True})
UMBRAL = 0.85
imp = pd.read_csv(R / "comparacion_importancia_shap_prob.csv").set_index("variable")
M = [("rf", "Random Forest"), ("xgb", "XGBoost"), ("tabpfn", "TabPFN")]
curvas, n_vars = {}, {}
for m, nom in M:
    p = imp[f"mean_abs_shap_{m}"].sort_values(ascending=False); a = (p / p.sum()).cumsum().to_numpy()
    curvas[nom] = a; n_vars[nom] = int((a < UMBRAL).sum()) + 1
print(n_vars)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, (m, nom) in zip(axes, M):
    ax.plot(np.arange(1, 44), curvas[nom], marker="o", markersize=3, color="#1f77b4")
    ax.axhline(UMBRAL, color="red", ls="--", lw=1, label="Umbral del 85 %")
    ax.axvline(n_vars[nom], color="gray", ls=":", lw=1.2, label=f"{n_vars[nom]} variables")
    ax.set_title(nom); ax.set_xlabel("Número de variables"); ax.legend(frameon=False, loc="lower right")
axes[0].set_ylabel("Importancia SHAP acumulada (proporción)")
plt.tight_layout(); plt.savefig(F / "FiguraS4_codo_shap.png", dpi=300, bbox_inches="tight"); plt.close()

# --- S5
cv = pd.read_csv(R / "tabla_metricas_cv.csv", header=[0, 1], index_col=0)
filas = {"Random Forest": "RF", "XGBoost": "XGBoost", "TabPFN": "TabPFN"}
nombres = list(filas)
auc = [cv.loc[filas[n], ("AUC-ROC", "Promedio")] for n in nombres]
f1 = [cv.loc[filas[n], ("F1-Score", "Promedio")] for n in nombres]
nv = [n_vars[n] for n in nombres]
# consenso = intersección de los tres paneles de 85 %
sel = {m: set(imp[f"mean_abs_shap_{m}"].sort_values(ascending=False).index[: n_vars[nom]]) for m, nom in M}
n_cons = len(sel["rf"] & sel["xgb"] & sel["tabpfn"]); cons_pct = [round(n_cons / n * 100, 1) for n in nv]
print("consenso", n_cons, cons_pct, auc, f1)
tmin, tmax = 500, 1600; f0, f1m = min(f1), max(f1)
tam = lambda f: tmin + (f - f0) / ((f1m - f0) or 1) * (tmax - tmin)
cmap = plt.get_cmap("RdYlGn"); norm = mcolors.Normalize(vmin=min(cons_pct) - 5, vmax=max(cons_pct) + 5)
off = {"Random Forest": (0, -48, "center", "top"), "XGBoost": (0, 42, "center", "bottom"), "TabPFN": (-95, 15, "left", "bottom")}
fig, ax = plt.subplots(figsize=(11, 7.5))
for nom, n, a, f, cp in zip(nombres, nv, auc, f1, cons_pct):
    ax.scatter(n, a, s=tam(f), color=cmap(norm(cp)), edgecolor="black", linewidth=1.4, zorder=3, alpha=0.92)
    dx, dy, ha, va = off[nom]
    ax.annotate(f"{nom}\n{n} variables · AUC-ROC = {a:.3f}".replace(".", ",") + f"\nConsenso = {cp:.1f} %".replace(".", ","),
                (n, a), textcoords="offset points", xytext=(dx, dy), fontsize=10, ha=ha, va=va,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))
ax.set_xlabel("Número de variables necesarias (85 % de la importancia SHAP acumulada)", fontsize=11.5)
ax.set_ylabel("AUC-ROC (validación cruzada)", fontsize=11.5)
ax.set_xlim(min(nv) - 4, max(nv) + 4); ax.set_ylim(min(auc) - 0.015, max(auc) + 0.015); ax.grid(alpha=0.25, zorder=0)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cb = plt.colorbar(sm, ax=ax, pad=0.015, fraction=0.05); cb.set_label("% de las variables del modelo incluidas en el consenso", fontsize=9.5)
ley = [ax.scatter([], [], s=tam(f), color="gray", edgecolor="black", alpha=0.55,
                  label=("F1 mínimo (" if i == 0 else "F1 máximo (") + f"{f:.3f})".replace(".", ",")) for i, f in enumerate([f0, f1m])]
ax.legend(handles=ley, loc="upper left", fontsize=9, title="Tamaño = F1", title_fontsize=9.5, frameon=True, borderpad=1.4, labelspacing=3.2, handletextpad=3.2)
plt.tight_layout(); plt.savefig(F / "FiguraS5_compromiso_parsimonia.png", dpi=300, bbox_inches="tight"); plt.close()
