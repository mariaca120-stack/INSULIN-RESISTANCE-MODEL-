"""Recalibración de los cuatro modelos con predicciones de validación cruzada del ENTRENAMIENTO (sin tocar el conjunto de prueba).
  1) oof:  predicciones fuera de partición (StratifiedKFold 5, semilla 42) de cada modelo en el entrenamiento.
  2) ajuste: regresión logística de una variable sobre logit(p_oof) -> p_cal = expit(a + b*logit(p)); se aplica a las probabilidades de prueba.
La recalibración es monótona: no cambia el AUC-ROC ni el orden de las predicciones.
    python pipeline/06_recalibration.py skl      # RF, SVM, XGBoost (proceso sin torch)
    python pipeline/06_recalibration.py tabpfn   # TabPFN
    python pipeline/06_recalibration.py analisis # calibración antes/después, métricas a 0,5, DCA
"""
import os, sys, time
from pathlib import Path
from config import ROOT as BASE, DATA, DATA as D, MODELS, RESULTS as R, FIGURES as F, FIGURES as FIG
modo = sys.argv[1] if len(sys.argv) > 1 else ""
if modo == "skl":
    import xgboost  # noqa: F401
import joblib, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss


SEED, TARGET = 42, "RESISTENCIA_INSULINA"
train = pd.read_csv(D / "train_modelo.csv"); y_tr = train[TARGET].astype(int).to_numpy(); X_tr = train.drop(columns=[TARGET])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
NOMBRES = {"TabPFN": "TabPFN", "XGBoost": "XGBoost", "Random Forest": "Random Forest", "SVM": "SVM"}

if modo == "skl":
    ms = {"Random Forest": joblib.load(MODELS / "modelo_random_forest.pkl"), "SVM": joblib.load(MODELS / "modelo_svm.pkl"),
          "XGBoost": joblib.load(MODELS / "modelo_xgboost.pkl")}
    oof = {}
    for n, m in ms.items():
        t0 = time.time(); oof[n] = cross_val_predict(clone(m), X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        print(n, f"{time.time()-t0:.0f}s AUC oof = {roc_auc_score(y_tr, oof[n]):.4f} | media oof {oof[n].mean():.3f} vs prev {y_tr.mean():.3f}", flush=True)
    pd.DataFrame(oof).to_csv(R / "_oof_skl.csv", index=False)
elif modo == "tabpfn":
    tok = BASE / ".tabpfn_token"
    if "TABPFN_TOKEN" not in os.environ and tok.exists(): os.environ["TABPFN_TOKEN"] = tok.read_text().strip()
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    ruta = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5).model_path
    m = TabPFNClassifier(model_path=ruta, n_estimators=32, balance_probabilities=True, random_state=SEED, device="mps")
    t0 = time.time(); p = cross_val_predict(clone(m), X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    print(f"TabPFN {time.time()-t0:.0f}s AUC oof = {roc_auc_score(y_tr, p):.4f} | media oof {p.mean():.3f} vs prev {y_tr.mean():.3f}", flush=True)
    pd.DataFrame({"TabPFN": p}).to_csv(R / "_oof_tabpfn.csv", index=False)
elif modo == "analisis":
    P = pd.read_csv(R / "probabilidades_test.csv"); y = P["y_test"].to_numpy(); n = len(y)
    oof = pd.concat([pd.read_csv(R / "_oof_tabpfn.csv"), pd.read_csv(R / "_oof_skl.csv")], axis=1)
    eps = 1e-6; lg = lambda p: logit(np.clip(p, eps, 1 - eps))
    rec, cal = {}, {}
    for m in NOMBRES:
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(lg(oof[m].to_numpy()).reshape(-1, 1), y_tr)
        a, b = lr.intercept_[0], lr.coef_[0][0]; cal[m] = (a, b)
        rec[m] = expit(a + b * lg(P[m].to_numpy()))
        assert abs(roc_auc_score(y, rec[m]) - roc_auc_score(y, P[m])) < 1e-9, m
    pd.DataFrame({"y_test": y, **rec}).to_csv(R / "probabilidades_test_recal.csv", index=False)
    print("Parámetros de recalibración (intercepto, pendiente):", {m: tuple(round(v, 3) for v in cal[m]) for m in cal})

    def citl(p, yy):   # calibración en lo grande: intercepto con pendiente fija en 1 (offset = logit p)
        o = lg(p); f = lambda a: -np.sum(yy * (a + o) - np.log1p(np.exp(a + o))); return minimize_scalar(f, bounds=(-5, 5), method="bounded").x
    def pend(p, yy): return LogisticRegression(C=1e6, max_iter=1000).fit(lg(p).reshape(-1, 1), yy).coef_[0][0]
    rng = np.random.default_rng(SEED); B = 1000; filas = []
    for etiqueta, S in [("Original", {m: P[m].to_numpy() for m in NOMBRES}), ("Recalibrado", rec)]:
        for m in NOMBRES:
            p = S[m]; est = [citl(p, y), pend(p, y), brier_score_loss(y, p)]; bs = []
            for _ in range(B):
                i = rng.integers(0, n, n)
                if y[i].min() == y[i].max(): continue
                bs.append([citl(p[i], y[i]), pend(p[i], y[i]), brier_score_loss(y[i], p[i])])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            filas.append({"Version": etiqueta, "Modelo": m, "Media_predicha": p.mean(), "Prevalencia": y.mean(),
                          "Intercepto": est[0], "Int_lo": lo[0], "Int_hi": hi[0], "Pendiente": est[1], "Pen_lo": lo[1], "Pen_hi": hi[1],
                          "Brier": est[2], "Brier_lo": lo[2], "Brier_hi": hi[2]})
    K = pd.DataFrame(filas); K.to_csv(R / "tabla_calibracion_recal.csv", index=False); print(K.round(3).to_string(index=False))

    # métricas a 0,5 (originales y recalibradas)
    f = []
    for etiqueta, S in [("Original", {m: P[m].to_numpy() for m in NOMBRES}), ("Recalibrado", rec)]:
        for m in NOMBRES:
            pr = S[m] >= 0.5; tp = (pr & (y == 1)).sum(); fp = (pr & (y == 0)).sum(); fn = (~pr & (y == 1)).sum(); tn = (~pr & (y == 0)).sum()
            sens, esp, ppv, npv = tp / (tp + fn), tn / (tn + fp), tp / (tp + fp), tn / (tn + fn)
            f.append({"Version": etiqueta, "Modelo": m, "Sens": sens, "Espec": esp, "VPP": ppv, "VPN": npv, "F1": 2 * ppv * sens / (ppv + sens),
                      "Exactitud": (tp + tn) / n})
    M05 = pd.DataFrame(f); M05.to_csv(R / "tabla_metricas_0_5_recal.csv", index=False); print(M05.round(3).to_string(index=False))

    # DCA
    ths = np.round(np.arange(0.01, 0.81, 0.01), 2); prev = y.mean()
    nb = lambda p, t: ((p >= t) & (y == 1)).sum() / n - ((p >= t) & (y == 0)).sum() / n * t / (1 - t)
    dca = pd.DataFrame({"Umbral": ths, **{m: [nb(rec[m], t) for t in ths] for m in NOMBRES}, "Intervenir a todos": [prev - (1 - prev) * t / (1 - t) for t in ths]})
    dca.to_csv(R / "dca_recal_completa.csv", index=False)
    sel = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]; g = dca.set_index("Umbral").loc[sel]
    ev = {m: (g[m] - g["Intervenir a todos"]) / (np.array(sel) / (1 - np.array(sel))) * 100 for m in NOMBRES}
    out = g.copy(); [out.__setitem__(f"Evitadas_100_{m}", ev[m]) for m in NOMBRES]; out.to_csv(R / "tabla_dca_recal.csv"); print(out.round(3).to_string())
else:
    sys.exit(__doc__)
