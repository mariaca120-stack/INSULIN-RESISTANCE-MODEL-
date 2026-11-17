"""Métricas del conjunto de prueba con probabilidades recalibradas: mismas columnas que metricas_test_ic95.csv.
AUC-ROC y AUC-PR no cambian (la recalibración es monótona; se copian). Sensibilidad, especificidad, VPP, VPN y F1 con umbral 0,5 y Brier se recalculan
con IC 95 % por bootstrap estratificado por clase (2000 réplicas, semilla 42)."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score
from config import ROOT as BASE, DATA, DATA as D, MODELS, RESULTS as R, FIGURES as F, FIGURES as FIG

rec = pd.read_csv(R / "probabilidades_test_recal.csv"); y = rec["y_test"].to_numpy()
orig = pd.read_csv(R / "metricas_test_ic95.csv").set_index("Modelo")
def met(p, yy):
    pr = p >= 0.5; tp = (pr & (yy == 1)).sum(); fp = (pr & (yy == 0)).sum(); fn = (~pr & (yy == 1)).sum(); tn = (~pr & (yy == 0)).sum()
    se, es, pp, pn = tp / (tp + fn), tn / (tn + fp), tp / (tp + fp), tn / (tn + fn)
    return np.array([se, es, pp, pn, 2 * pp * se / (pp + se), brier_score_loss(yy, p)])
rng = np.random.default_rng(42); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; B = 2000
filas = []
for m in orig.index:
    p = rec[m].to_numpy(); est = met(p, y); bs = []
    for _ in range(B):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]); bs.append(met(p[i], y[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    f = {"Modelo": m}
    for k in ["AUC", "PRAUC"]: f.update({k: orig.loc[m, k], k + "_lo": orig.loc[m, k + "_lo"], k + "_hi": orig.loc[m, k + "_hi"]})
    for j, k in enumerate(["Sens", "Espec", "VPP", "VPN", "F1", "Brier"]): f.update({k: est[j], k + "_lo": lo[j], k + "_hi": hi[j]})
    assert abs(roc_auc_score(y, p) - orig.loc[m, "AUC"]) < 1e-9
    filas.append(f)
out = pd.DataFrame(filas)[list(orig.reset_index().columns)]; out.to_csv(R / "metricas_test_ic95_recal.csv", index=False)
print(out[["Modelo", "Sens", "Espec", "VPP", "VPN", "F1", "Brier"]].round(3).to_string(index=False))
