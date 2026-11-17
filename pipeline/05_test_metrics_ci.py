"""Paso 05: métricas del conjunto de prueba con intervalos de confianza del 95 % (DeLong para el AUC, bootstrap para el resto),
comparación de TabPFN con los demás modelos (DeLong) y beneficio neto."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from config import RESULTS

P = pd.read_csv(RESULTS / "probabilidades_test.csv")
y = P["y_test"].to_numpy()
MODELOS = ["Random Forest", "SVM", "XGBoost", "TabPFN"]
B, SEED = 2000, 42


def midrank(x):
    J = np.argsort(x)
    Z = x[J]
    T = np.zeros(len(x))
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(len(x))
    T2[J] = T
    return T2


def delong_components(yt, preds):
    """AUC y su matriz de covarianza (DeLong rápido; preds tiene forma [n_modelos, n])."""
    yt = np.asarray(yt)
    orden = (-yt).argsort()
    P_ = np.asarray(preds)[:, orden]
    m = int(yt.sum())
    n = len(yt) - m
    tx = np.array([midrank(p[:m]) for p in P_])
    ty = np.array([midrank(p[m:]) for p in P_])
    tz = np.array([midrank(p) for p in P_])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def auc_ic_delong(yt, p):
    aucs, cov = delong_components(yt, np.asarray(p)[None, :])
    se = float(np.sqrt(cov[0, 0]))
    return aucs[0], aucs[0] - 1.959964 * se, min(1.0, aucs[0] + 1.959964 * se)


def metricas(p, yy):
    pred = p >= 0.5
    tp = (pred & (yy == 1)).sum(); fp = (pred & (yy == 0)).sum()
    fn = (~pred & (yy == 1)).sum(); tn = (~pred & (yy == 0)).sum()
    se, es, ppv, npv = tp / (tp + fn), tn / (tn + fp), tp / (tp + fp), tn / (tn + fn)
    return np.array([se, es, ppv, npv, 2 * ppv * se / (ppv + se), average_precision_score(yy, p), brier_score_loss(yy, p)])


NOMBRES = ["Sens", "Espec", "VPP", "VPN", "F1", "PRAUC", "Brier"]
rng = np.random.default_rng(SEED)
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
filas = []
for m in MODELOS:
    p = P[m].to_numpy()
    a, lo, hi = auc_ic_delong(y, p)
    est = metricas(p, y)
    bs = []
    for _ in range(B):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        bs.append(metricas(p[i], y[i]))
    ci = np.percentile(bs, [2.5, 97.5], axis=0)
    fila = {"Modelo": m, "AUC": a, "AUC_lo": lo, "AUC_hi": hi}
    for j, k in enumerate(NOMBRES):
        fila.update({k: est[j], k + "_lo": ci[0][j], k + "_hi": ci[1][j]})
    assert abs(a - roc_auc_score(y, p)) < 1e-9
    filas.append(fila)
columnas = ["Modelo", "AUC", "AUC_lo", "AUC_hi"] + [c for k in ["Sens", "Espec", "VPP", "VPN", "F1", "PRAUC", "Brier"]
                                                    for c in (k, k + "_lo", k + "_hi")]
tabla = pd.DataFrame(filas)[columnas]
tabla.to_csv(RESULTS / "metricas_test_ic95.csv", index=False)
print(tabla[["Modelo", "AUC", "AUC_lo", "AUC_hi", "Sens", "Espec", "F1", "Brier"]].round(3).to_string(index=False))

# prueba de DeLong: TabPFN frente a los otros modelos
filas = []
for otro, etiqueta in [("XGBoost", "TabPFN vs XGBoost"), ("Random Forest", "TabPFN vs Random Forest"), ("SVM", "TabPFN vs SVM")]:
    aucs, cov = delong_components(y, np.vstack([P["TabPFN"].to_numpy(), P[otro].to_numpy()]))
    d = aucs[0] - aucs[1]
    z = d / np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    filas.append({"Comparación": etiqueta, "Diferencia AUC": round(d, 3), "p": round(2 * (1 - stats.norm.cdf(abs(z))), 4)})
pd.DataFrame(filas).to_csv(RESULTS / "delong_tabpfn_vs_otros.csv", index=False)
print(pd.DataFrame(filas).to_string(index=False))

# beneficio neto
prev = y.mean()


def beneficio_neto(p, t):
    pred = p >= t
    return ((pred & (y == 1)).sum() - (pred & (y == 0)).sum() * t / (1 - t)) / len(y)


umbrales = np.round(np.arange(0.05, 0.8001, 0.05), 2)
dca = pd.DataFrame({"Umbral": umbrales,
                    "Intervenir a todos": [round(prev - (1 - prev) * t / (1 - t), 4) for t in umbrales],
                    **{m: [round(beneficio_neto(P[m].to_numpy(), t), 4) for t in umbrales]
                       for m in ["Random Forest", "SVM", "XGBoost", "TabPFN"]}})
dca.to_csv(RESULTS / "dca_completa.csv", index=False)
print(dca.head(4).to_string(index=False))
