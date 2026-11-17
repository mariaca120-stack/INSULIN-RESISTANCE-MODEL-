"""Paso 09: reentrena los modelos con las 10 variables de consenso y los compara con los modelos completos.

    python pipeline/09_consensus_panel.py skl
    python pipeline/09_consensus_panel.py tabpfn
    python pipeline/09_consensus_panel.py unir"""
import json, os, sys, time
from pathlib import Path
from config import ROOT as BASE, DATA, DATA as D, MODELS, RESULTS as R, FIGURES as F, FIGURES as FIG
modo = sys.argv[1] if len(sys.argv) > 1 else ""
if modo == "skl":
    import xgboost  # noqa: F401
import joblib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate


SEED, TARGET, UMBRAL = 42, "RESISTENCIA_INSULINA", 0.85
train = pd.read_csv(D / "train_modelo.csv"); y = train[TARGET].astype(int); X = train.drop(columns=[TARGET])

def consenso():
    """Panel de consenso = intersección de las variables que alcanzan el 85 % del |SHAP| acumulado en RF, XGBoost y TabPFN
    (calculado en el paso 08, modo consenso_arboles, con todo el conjunto de entrenamiento)."""
    return json.load(open(R / "consenso_todo.json"))["consenso"]

VARS = consenso(); print(len(VARS), VARS, flush=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
sc = {"acc": "accuracy", "auc": "roc_auc", "f1": "f1"}

test = pd.read_csv(D / "test_modelo.csv"); Xte, yte = test.drop(columns=[TARGET])[VARS], test[TARGET].astype(int)
PRED = {}

def evalua(nombre, modelo):
    t0 = time.time()
    PRED[nombre] = clone(modelo).fit(X[VARS], y).predict_proba(Xte)[:, 1]
    r = cross_validate(clone(modelo), X[VARS], y, cv=cv, scoring=sc, n_jobs=1, error_score="raise")
    print(nombre, f"{time.time()-t0:.0f}s", {k: round(r[f'test_{k}'].mean(), 4) for k in sc}, flush=True)
    return {"Modelo": nombre, "Conjunto de variables": f"Consenso ({len(VARS)} var)", "AUC-ROC": round(r["test_auc"].mean(), 4),
            "Accuracy": round(r["test_acc"].mean(), 4), "F1-Score": round(r["test_f1"].mean(), 4)}

if modo == "skl":
    ms = {"RF": joblib.load(MODELS / "modelo_random_forest.pkl"), "SVM": joblib.load(MODELS / "modelo_svm.pkl"),
          "XGBoost": joblib.load(MODELS / "modelo_xgboost.pkl")}
    pd.DataFrame([evalua(n, m) for n, m in ms.items()]).to_csv(R / "_cv10_skl.csv", index=False)
    pd.DataFrame(PRED).to_csv(R / "_pred10_skl.csv", index=False)
elif modo == "tabpfn":
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    tok = BASE / ".tabpfn_token"
    if "TABPFN_TOKEN" not in os.environ and tok.exists(): os.environ["TABPFN_TOKEN"] = tok.read_text().strip()
    ruta = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5).model_path
    m = TabPFNClassifier(model_path=ruta, n_estimators=32, balance_probabilities=True, random_state=SEED, device="mps")
    pd.DataFrame([evalua("TabPFN", m)]).to_csv(R / "_cv10_tabpfn.csv", index=False)
    pd.DataFrame(PRED).to_csv(R / "_pred10_tabpfn.csv", index=False)
elif modo == "delong":
    from scipy import stats
    def midrank(x):
        J = np.argsort(x); Z = x[J]; T = np.zeros(len(x)); i = 0
        while i < len(x):
            j = i
            while j < len(x) and Z[j] == Z[i]: j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1; i = j
        T2 = np.empty(len(x)); T2[J] = T; return T2
    def delong(yt, a, b):  # misma implementación que el cuaderno (sección 15.5)
        yt = np.asarray(yt); o = (-yt).argsort(); P = np.vstack([a[o], b[o]]); m = int(yt.sum()); n = len(yt) - m
        tx = np.array([midrank(p[:m]) for p in P]); ty = np.array([midrank(p[m:]) for p in P]); tz = np.array([midrank(p) for p in P])
        aucs = tz[:, :m].sum(1) / (m * n) - (m + 1.0) / (2.0 * n)
        v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
        cov = np.cov(v01) / m + np.cov(v10) / n; l = np.array([1, -1]); d = l @ aucs; z = d / np.sqrt(l @ cov @ l)
        return d, z, 2 * (1 - stats.norm.cdf(abs(z))), aucs[0], aucs[1]
    full = pd.read_csv(R / "probabilidades_test.csv"); assert (full["y_test"].to_numpy() == yte.to_numpy()).all()
    cons = pd.concat([pd.read_csv(R / "_pred10_skl.csv"), pd.read_csv(R / "_pred10_tabpfn.csv")], axis=1)
    mp = {"RF": "Random Forest", "SVM": "SVM", "XGBoost": "XGBoost", "TabPFN": "TabPFN"}
    f = []
    for n, col in mp.items():
        d, z, p, a1, a2 = delong(yte.to_numpy(), full[col].to_numpy(), cons[n].to_numpy())
        f.append({"Modelo": n, "AUC completo": round(a1, 4), "AUC consenso": round(a2, 4), "Diferencia": round(d, 4),
                  "z": round(z, 3), "p-valor": round(p, 4), "Significativo (α=0.05)": "Sí" if p < 0.05 else "No"})
    t = pd.DataFrame(f); t.to_csv(R / "tabla_delong_10var.csv", index=False); print(t.to_string(index=False))
elif modo == "unir":
    cvt = pd.read_csv(R / "tabla_metricas_cv.csv", header=[0, 1], index_col=0)   # paso 04: modelos completos (43 variables)
    old = pd.DataFrame([{"Modelo": m, "Conjunto de variables": "Completo (43 var)", "AUC-ROC": cvt.loc[m, ("AUC-ROC", "Promedio")],
                         "Accuracy": cvt.loc[m, ("Accuracy", "Promedio")], "F1-Score": cvt.loc[m, ("F1-Score", "Promedio")]}
                        for m in ["RF", "SVM", "XGBoost", "TabPFN"]])
    new = pd.concat([pd.read_csv(R / "_cv10_skl.csv"), pd.read_csv(R / "_cv10_tabpfn.csv")])
    t = pd.concat([old, new]); t["_o"] = t["Modelo"].map({"RF": 0, "SVM": 1, "XGBoost": 2, "TabPFN": 3})
    t["_c"] = t["Conjunto de variables"].str.startswith("Consenso"); t = t.sort_values(["_o", "_c"]).drop(columns=["_o", "_c"])
    t.to_csv(R / "tabla_comparacion_consenso_10var.csv", index=False); print(t.to_string(index=False))
else:
    sys.exit(__doc__)
