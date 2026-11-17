"""Paso 11a: tablas de importancia SHAP y figuras (mapas de calor y gráficos de enjambre)."""
import locale, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, seaborn as sns, shap
from scipy.stats import spearmanr
from config import ROOT as BASE, DATA, DATA as D, MODELS, RESULTS as R, FIGURES as F, FIGURES as FIG



SEED, TARGET, N_SHAP, UMBRAL = 42, "RESISTENCIA_INSULINA", 1828, 0.85
try:
    locale.setlocale(locale.LC_NUMERIC, "es_ES.UTF-8")   # coma decimal en las figuras (opcional)
except locale.Error:
    pass
plt.rcParams.update({"font.family": "Arial", "axes.formatter.use_locale": True})
M = ["rf", "xgb", "tabpfn"]
NOM = {"rf": "Random Forest", "xgb": "XGBoost", "tabpfn": "TabPFN"}

ETIQUETA = {"CINTURA": "Circunferencia de cintura", "IMC": "IMC", "TRIGLICERIDOS": "Triglicéridos", "ALT": "ALT",
            "COLESTEROL_HDL": "Colesterol HDL", "VITAMINA_D": "25-hidroxivitamina D", "PULSO": "Frecuencia de pulso",
            "EDAD": "Edad", "PROTEINA_C_REACTIVA": "Proteína C reactiva (as)", "BRAZO": "Circunferencia de brazo",
            "CADERA": "Circunferencia de cadera", "HIERRO": "Hierro sérico", "AST": "AST",
            "DX_HIPERTENSION": "Dx. hipertensión", "COLESTEROL_LDL": "Colesterol LDL",
            "MAS_100_CIGARRILLOS": "≥ 100 cigarrillos", "INGESTA_PROTEINA": "Ingesta de proteínas"}

train = pd.read_csv(D / "train_modelo.csv")
X_all = train.drop(columns=[TARGET])
X_shap = X_all  # conjunto de entrenamiento completo (n = 1828)

# 43 variables
v43 = {m: np.load(R / f"shap_todo_prob_{m}.npy") for m in ["rf", "xgb", "tabpfn"]}
for m, v in v43.items():
    assert v.shape == X_shap.shape and np.isfinite(v).all(), m
imp = pd.DataFrame({f"mean_abs_shap_{m}": pd.Series(np.abs(v).mean(0), index=X_all.columns) for m, v in v43.items()})
for m in M:
    imp[f"proporcion_{m}"] = imp[f"mean_abs_shap_{m}"] / imp[f"mean_abs_shap_{m}"].sum()
    imp[f"ranking_{m}"] = imp[f"mean_abs_shap_{m}"].rank(ascending=False)
imp["ranking_promedio"] = imp[[f"ranking_{m}" for m in M]].mean(axis=1)
imp = imp.sort_values("ranking_promedio"); imp.index.name = "variable"
imp.to_csv(R / "comparacion_importancia_shap_prob.csv")

sel = {}
for m in M:
    s = imp[f"mean_abs_shap_{m}"].sort_values(ascending=False)
    sel[m] = list(s.index[: int(((s / s.sum()).cumsum() < UMBRAL).sum()) + 1])
VARS = [v for v in sel["rf"] if v in set(sel["xgb"]) & set(sel["tabpfn"])]  # orden = ranking de RF (columnas de los .npy)
print("85 % por modelo:", {m: len(v) for m, v in sel.items()}, "| consenso:", len(VARS), VARS)
for a, b in [("rf", "xgb"), ("rf", "tabpfn"), ("xgb", "tabpfn")]:
    print(f"43 var Spearman {a} vs {b}: {spearmanr(imp[f'ranking_{a}'], imp[f'ranking_{b}'])[0]:.2f}")

# Figure 4
p4 = pd.DataFrame({NOM[m]: imp[f"proporcion_{m}"] for m in M})
p4 = p4.loc[p4.mean(axis=1).sort_values(ascending=False).index[:15]]
p4.index = [ETIQUETA.get(i, i) for i in p4.index]
def heat(prop, archivo, size):
    fig, ax = plt.subplots(figsize=size)
    sns.heatmap(prop, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5,
                cbar_kws={"label": "Proporción de la importancia SHAP total"}, ax=ax)
    for t in ax.texts:
        t.set_text(t.get_text().replace(".", ","))
    ax.set_xlabel(""); ax.set_ylabel(""); plt.tight_layout()
    plt.savefig(F / archivo, dpi=300, bbox_inches="tight"); plt.close(); print("guardada", archivo)
heat(p4, "Figura4_heatmap_shap.png", (6, 7))

# beeswarm plots
ETIQ_C = {**ETIQUETA}
def enjambre(v, X, nombres, archivo, size, maxd, valores_color=None):
    plt.figure()
    shap.summary_plot(v, X if valores_color is None else valores_color, feature_names=nombres, max_display=maxd,
                      show=False, color_bar_label="Valor de la variable", plot_size=size)
    fig = plt.gcf()
    fig.axes[0].set_xlabel("Valor SHAP (contribución a la probabilidad predicha de RI)", fontsize=11)
    cbar = fig.axes[-1]; cbar.set_yticklabels(["Bajo / no", "Alto / sí"]); cbar.set_ylabel("Valor de la variable", fontsize=11)
    plt.tight_layout(); plt.savefig(F / archivo, dpi=300, bbox_inches="tight"); plt.close(); print("guardada", archivo)

if "--solo43" in sys.argv:
    sys.exit(0)

# 10 consensus variables
import json
assert VARS == json.load(open(R / "consenso_todo.json"))["consenso"], "el consenso cambió"
vc = {m: np.load(R / f"shap_todo_consenso_prob_{m}.npy") for m in M}
Xc = X_shap[VARS]
for m, v in vc.items():
    assert v.shape == Xc.shape and np.isfinite(v).all(), (m, v.shape)
tab = pd.DataFrame(index=VARS)
for m in M:
    s = pd.Series(np.abs(vc[m]).mean(0), index=VARS)
    tab[f"mean_abs_shap_{m}"], tab[f"proporcion_{m}"], tab[f"ranking_{m}"] = s, s / s.sum(), s.rank(ascending=False)
tab["ranking_promedio"] = tab[[f"ranking_{m}" for m in M]].mean(axis=1)
tab = tab.sort_values("ranking_promedio"); tab.index.name = "variable"
tab.to_csv(R / "comparacion_importancia_shap_consenso_prob.csv")
print(tab.round(3).to_string())
for a, b in [("rf", "xgb"), ("rf", "tabpfn"), ("xgb", "tabpfn")]:
    r, p = spearmanr(tab[f"ranking_{a}"], tab[f"ranking_{b}"]); print(f"Consenso Spearman {a} vs {b}: rho = {r:.2f}, p = {p:.4g}")
pc = pd.DataFrame({NOM[m]: tab[f"proporcion_{m}"] for m in M}); pc.index = [ETIQUETA[i] for i in pc.index]
heat(pc, "Figura6_heatmap_shap_consenso.png", (6, 4.8))
nom_c = [ETIQUETA[v] for v in VARS]
for m, archivo in [("tabpfn", "Figura7_shap_tabpfn_consenso.png"), ("rf", "FiguraS7_shap_rf_consenso.png"),
                   ("xgb", "FiguraS8_shap_xgb_consenso.png")]:
    enjambre(vc[m], Xc, nom_c, archivo, (9, 5.6), len(VARS))
