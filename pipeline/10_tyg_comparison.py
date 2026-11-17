"""Comparación de los modelos con el índice triglicéridos-glucosa (TyG) en el conjunto de prueba.
TyG = ln[triglicéridos (mg/dL) x glucosa en ayunas (mg/dL) / 2].  La glucosa se toma de train/test_imputado.csv
(alineados fila a fila con train/test_modelo.csv; se verifica). El TyG se usa (a) como puntaje continuo (AUC) y (b) como
única variable de una regresión logística ajustada en entrenamiento, para obtener probabilidades y poder hacer el DCA."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from config import ROOT as BASE, DATA, DATA as D, MODELS, RESULTS as R, FIGURES as F, FIGURES as FIG


SEED, TARGET = 42, "RESISTENCIA_INSULINA"

def carga(nombre):
    imp, mod = pd.read_csv(D / f"{nombre}_imputado.csv"), pd.read_csv(D / f"{nombre}_modelo.csv")
    comunes = [c for c in mod.columns if c in imp.columns]
    assert len(imp) == len(mod) and all(np.allclose(imp[c].astype(float), mod[c].astype(float), equal_nan=True) for c in comunes), nombre
    return imp, mod
tr_i, tr_m = carga("train"); te_i, te_m = carga("test")
tyg = lambda d: np.log(d["TRIGLICERIDOS"] * d["GLUCOSA"] / 2)
x_tr, y_tr, x_te, y_te = tyg(tr_i).to_numpy(), tr_m[TARGET].astype(int).to_numpy(), tyg(te_i).to_numpy(), te_m[TARGET].astype(int).to_numpy()
P = pd.read_csv(R / "probabilidades_test_recal.csv")   # probabilidades recalibradas (recalibracion.py); assert (P["y_test"].to_numpy() == y_te).all()
print(f"TyG train: media {x_tr.mean():.2f} (DE {x_tr.std():.2f}) | test: {x_te.mean():.2f} | n test = {len(y_te)} (RI = {y_te.sum()})")

# ---- DeLong (misma implementación que el cuaderno), con IC del AUC
def midrank(x):
    J = np.argsort(x); Z = x[J]; T = np.zeros(len(x)); i = 0
    while i < len(x):
        j = i
        while j < len(x) and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(len(x)); T2[J] = T; return T2
def delong(yt, preds):
    o = (-yt).argsort(); Pm = np.vstack([p[o] for p in preds]); m = int(yt.sum()); n = len(yt) - m
    tx = np.array([midrank(p[:m]) for p in Pm]); ty = np.array([midrank(p[m:]) for p in Pm]); tz = np.array([midrank(p) for p in Pm])
    aucs = tz[:, :m].sum(1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    return aucs, np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
def comparar(a, b):
    aucs, cov = delong(y_te, [a, b]); d = aucs[0] - aucs[1]; z = d / np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    return d, 2 * (1 - stats.norm.cdf(abs(z)))

# ---- TyG como probabilidad: regresión logística de una variable, ajustada en entrenamiento
lr = LogisticRegression(C=1e6, max_iter=1000).fit(x_tr.reshape(-1, 1), y_tr)
p_tyg = lr.predict_proba(x_te.reshape(-1, 1))[:, 1]
print(f"Logística TyG: OR por unidad de TyG = {np.exp(lr.coef_[0][0]):.2f}; AUC test = {roc_auc_score(y_te, p_tyg):.4f} (monótona: igual al del TyG crudo {roc_auc_score(y_te, x_te):.4f})")

# ---- Tabla de comparación (AUC con IC 95 % por DeLong; DeLong pareado frente al TyG)
modelos = {"TabPFN": P["TabPFN"].to_numpy(), "XGBoost": P["XGBoost"].to_numpy(), "Random Forest": P["Random Forest"].to_numpy(), "SVM": P["SVM"].to_numpy()}
filas = []
aucs, cov = delong(y_te, [x_te]); a, se = aucs[0], np.sqrt(cov[0, 0])
# punto de corte de Youden elegido en ENTRENAMIENTO
fpr, tpr, thr = roc_curve(y_tr, x_tr); corte = thr[np.argmax(tpr - fpr)]
pred = x_te >= corte; sens = (pred & (y_te == 1)).sum() / (y_te == 1).sum(); esp = (~pred & (y_te == 0)).sum() / (y_te == 0).sum()
filas.append({"Modelo": "Índice TyG", "AUC": a, "IC_inf": a - 1.96 * se, "IC_sup": a + 1.96 * se, "Dif_vs_TyG": np.nan, "p_vs_TyG": np.nan,
              "Sens": sens, "Espec": esp})
print(f"Corte de Youden (entrenamiento): TyG >= {corte:.2f}")
for nombre, p in modelos.items():
    aucs, cov = delong(y_te, [p]); a, se = aucs[0], np.sqrt(cov[0, 0]); d, pv = comparar(p, x_te)
    pr = p >= 0.5; s = (pr & (y_te == 1)).sum() / (y_te == 1).sum(); e = (~pr & (y_te == 0)).sum() / (y_te == 0).sum()
    filas.append({"Modelo": nombre, "AUC": a, "IC_inf": a - 1.96 * se, "IC_sup": a + 1.96 * se, "Dif_vs_TyG": d, "p_vs_TyG": pv, "Sens": s, "Espec": e})
T = pd.DataFrame(filas); T.to_csv(R / "tyg_comparacion_auc.csv", index=False); print(T.round(4).to_string(index=False))

# ---- DCA (mismo cálculo que la Figura 3), con el TyG calibrado
umbrales = np.round(np.arange(0.01, 0.81, 0.01), 2); n = len(y_te); prev = y_te.mean()
def nb(p, t): return ((p >= t) & (y_te == 1)).sum() / n - ((p >= t) & (y_te == 0)).sum() / n * t / (1 - t)
dca = pd.DataFrame({"Umbral": umbrales, "Índice TyG": [nb(p_tyg, t) for t in umbrales],
                    **{m: [nb(p, t) for t in umbrales] for m, p in modelos.items()},
                    "Intervenir a todos": [prev - (1 - prev) * t / (1 - t) for t in umbrales]})
dca.to_csv(R / "tyg_dca.csv", index=False)
ref = pd.read_csv(R / "dca_recal_completa.csv"); k = ref["Umbral"].round(2).isin(umbrales)
chk = dca.set_index("Umbral").loc[ref.loc[k, "Umbral"].round(2), "TabPFN"].to_numpy()
print("DCA reproduce el de TabPFN del artículo:", np.allclose(chk, ref.loc[k, "TabPFN"].to_numpy(), atol=1e-4), "(archivo guardado redondeado a 4 decimales)")

# ---- Beneficio neto TabPFN/XGBoost menos TyG, con IC 95 % por bootstrap (2000 réplicas, pareado)
rng = np.random.default_rng(SEED); B = 2000; ts = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
bs = {(m, t): [] for m in ["TabPFN", "XGBoost"] for t in ts}
for _ in range(B):
    i = rng.integers(0, n, n); yb = y_te[i]
    def nbb(p, t): return ((p[i] >= t) & (yb == 1)).sum() / n - ((p[i] >= t) & (yb == 0)).sum() / n * t / (1 - t)
    for t in ts:
        base = nbb(p_tyg, t)
        for m in ["TabPFN", "XGBoost"]: bs[(m, t)].append(nbb(modelos[m], t) - base)
f = []
for t in ts:
    for m in ["TabPFN", "XGBoost"]:
        obs = nb(modelos[m], t) - nb(p_tyg, t); lo, hi = np.percentile(bs[(m, t)], [2.5, 97.5])
        f.append({"Umbral": t, "Modelo": m, "NB_modelo": nb(modelos[m], t), "NB_TyG": nb(p_tyg, t), "Dif": obs, "IC_inf": lo, "IC_sup": hi,
                  "Intervenciones_evitadas_por_100": obs / (t / (1 - t)) * 100})
N = pd.DataFrame(f); N.to_csv(R / "tyg_beneficio_neto.csv", index=False); print(N.round(4).to_string(index=False))

# ---- Modelos de consenso (10 variables) frente al TyG
pc = pd.concat([pd.read_csv(R / "_pred10_skl.csv"), pd.read_csv(R / "_pred10_tabpfn.csv")], axis=1)
f = []
for nombre, col in [("TabPFN", "TabPFN"), ("XGBoost", "XGBoost"), ("Random Forest", "RF"), ("SVM", "SVM")]:
    p = pc[col].to_numpy(); aucs, cov = delong(y_te, [p]); d, pv = comparar(p, x_te)
    f.append({"Modelo": nombre + " (consenso, 10 var)", "AUC": aucs[0], "IC_inf": aucs[0] - 1.96 * np.sqrt(cov[0, 0]),
              "IC_sup": aucs[0] + 1.96 * np.sqrt(cov[0, 0]), "Dif_vs_TyG": d, "p_vs_TyG": pv})
C = pd.DataFrame(f); C.to_csv(R / "tyg_comparacion_auc_consenso.csv", index=False); print(C.round(4).to_string(index=False))
