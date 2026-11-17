"""Figuras finales del manuscrito: Fig. 2 (ROC con TyG + calibración recalibrada), Fig. 5 (panel de consenso: A mapa de calor, B enjambre TabPFN) y Fig. 6 (curvas de decisión con TyG)."""
import locale
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg, numpy as np, pandas as pd
from scipy.special import logit
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from config import ROOT as BASE, DATA, DATA as D, MODELS, RESULTS as R, FIGURES as F, FIGURES as FIG

try:
    locale.setlocale(locale.LC_NUMERIC, "es_ES.UTF-8")   # decimal comma in the figures (optional)
except locale.Error:
    pass
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.formatter.use_locale": True})

MOD = ["TabPFN", "XGBoost", "Random Forest", "SVM"]
COLOR = {"TabPFN": "#1b6ca8", "XGBoost": "#d1495b", "Random Forest": "#2a9d8f", "SVM": "#e9a03b", "Índice TyG": "#555555"}
rec = pd.read_csv(R / "probabilidades_test_recal.csv"); y = rec["y_test"].to_numpy()
te = pd.read_csv(DATA / "test_imputado.csv"); tyg = np.log(te["TRIGLICERIDOS"] * te["GLUCOSA"] / 2).to_numpy()
ic = pd.read_csv(R / "metricas_test_ic95_recal.csv").set_index("Modelo")
T = pd.read_csv(R / "tyg_comparacion_auc.csv").set_index("Modelo"); K = pd.read_csv(R / "tabla_calibracion_recal.csv")
brier = {r.Modelo: r.Brier for r in K[K.Version == "Recalibrado"].itertuples()}
f3 = lambda v: f"{v:.3f}".replace(".", ",")
lg = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))

def suave(p, yy):
    m = make_pipeline(SplineTransformer(n_knots=4, degree=3, extrapolation="constant"), LogisticRegression(C=1e3, max_iter=2000)).fit(lg(p).reshape(-1, 1), yy)
    g = np.linspace(np.percentile(p, 2), np.percentile(p, 98), 100); return g, m.predict_proba(lg(g).reshape(-1, 1))[:, 1]

# ---- Figura 2
fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
for m in MOD:
    fpr, tpr, _ = roc_curve(y, rec[m]); a = ic.loc[m]
    ax[0].plot(fpr, tpr, c=COLOR[m], lw=1.6, label=f"{m}: {f3(a.AUC)} ({f3(a.AUC_lo)}–{f3(a.AUC_hi)})")
    fr, me = calibration_curve(y, rec[m], n_bins=10, strategy="quantile")
    ax[1].plot(me, fr, marker="o", ms=3.5, c=COLOR[m], lw=1.4, label=f"{m} (Brier {f3(brier[m])})")
fpr, tpr, _ = roc_curve(y, tyg); t = T.loc["Índice TyG"]
ax[0].plot(fpr, tpr, c=COLOR["Índice TyG"], lw=1.8, ls="--", label=f"Índice TyG: {f3(t.AUC)} ({f3(t.IC_inf)}–{f3(t.IC_sup)})")
ax[0].plot([0, 1], [0, 1], c="lightgrey", lw=0.8); ax[1].plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8, label="Calibración perfecta")
ax[0].set(xlabel="1 − Especificidad", ylabel="Sensibilidad"); ax[0].legend(title="AUC-ROC (IC 95 %)", fontsize=8, title_fontsize=8, loc="lower right", frameon=False)
ax[1].set(xlabel="Probabilidad media predicha por decil (recalibrada)", ylabel="Proporción observada de RI", xlim=(0, 1), ylim=(0, 1)); ax[1].legend(fontsize=8, loc="upper left", frameon=False)
for a, l in zip(ax, "AB"): a.text(-0.12, 1.03, l, transform=a.transAxes, fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig(F / "Fig2_roc_calibracion.png", dpi=300, bbox_inches="tight"); plt.close()

# ---- Figura 5 (A mapa de calor + B enjambre del panel de consenso)
A_, B_ = mpimg.imread(F / "Figura6_heatmap_shap_consenso.png"), mpimg.imread(F / "Figura7_shap_tabpfn_consenso.png")
h = 4.6; wa, wb = h * A_.shape[1] / A_.shape[0], h * B_.shape[1] / B_.shape[0]
fig = plt.figure(figsize=(wa + wb + 0.6, h + 0.3))
ax1 = fig.add_axes([0.0, 0.0, wa / (wa + wb + 0.6), 1.0]); ax2 = fig.add_axes([(wa + 0.6) / (wa + wb + 0.6), 0.0, wb / (wa + wb + 0.6), 1.0])
for a, im, l in [(ax1, A_, "A"), (ax2, B_, "B")]:
    a.imshow(im); a.axis("off"); a.text(0.0, 1.0, l, transform=a.transAxes, fontsize=14, fontweight="bold", va="top")
plt.savefig(F / "Fig5_panel_consenso.png", dpi=300, bbox_inches="tight"); plt.close()

# ---- Figura 6 (curvas de decisión con TyG)
d = pd.read_csv(R / "tyg_dca.csv"); prev = y.mean()
fig, a = plt.subplots(figsize=(7.2, 4.6))
for m in MOD: a.plot(d["Umbral"], d[m], c=COLOR[m], lw=1.6, label=m)
a.plot(d["Umbral"], d["Índice TyG"], c=COLOR["Índice TyG"], lw=2.0, ls="--", label="Índice TyG")
a.plot(d["Umbral"], d["Intervenir a todos"], c="grey", ls=":", lw=1.2, label="Intervenir a todos"); a.axhline(0, c="black", lw=0.9, label="No intervenir")
a.set(xlim=(0, 0.8), ylim=(-0.05, prev + 0.03), xlabel="Probabilidad umbral de RI", ylabel="Beneficio neto"); a.legend(frameon=False, fontsize=8.5)
plt.tight_layout(); plt.savefig(F / "Fig6_curvas_decision.png", dpi=300, bbox_inches="tight"); plt.close(); print("figuras guardadas")
