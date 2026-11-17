"""SHAP en escala de PROBABILIDAD calculado con TODO el conjunto de entrenamiento (n = 1828), para RF, XGBoost y TabPFN.
Salidas NUEVAS (no pisa nada): resultados/shap_todo_prob_{rf,xgb,tabpfn}.npy (1828 x 43), resultados/consenso_todo.json y
resultados/shap_todo_consenso_prob_{rf,xgb,tabpfn}.npy (1828 x variables de consenso, en el orden del ranking de RF).
    rf43 | xgb43 | tabpfn43 | consenso_arboles | tabpfn_consenso      (XGBoost y PyTorch no deben compartir proceso en macOS)"""
import copy, json, os, sys, time
from pathlib import Path
from config import ROOT as BASE, DATA, DATA as D, MODELS, RESULTS as R, FIGURES as F, FIGURES as FIG
modo = sys.argv[1] if len(sys.argv) > 1 else ""
if modo in ("xgb43", "consenso_arboles"):
    import xgboost  # noqa: F401  (antes que cualquier otra cosa; OpenMP en macOS)
import joblib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

SEED, TARGET, UMBRAL, N_FONDO, CHUNK = 42, "RESISTENCIA_INSULINA", 0.85, 200, 200
train = pd.read_csv(D / "train_modelo.csv"); test = pd.read_csv(D / "test_modelo.csv")
X_train, y_train = train.drop(columns=[TARGET]), train[TARGET].astype(int); X_test, y_test = test.drop(columns=[TARGET]), test[TARGET].astype(int)
X_shap = X_train                                   # TODO el conjunto de entrenamiento
print(f"[{modo}] filas a explicar: {len(X_shap)} (entrenamiento completo)", flush=True)

def clase1(v):
    if isinstance(v, list): return v[1]
    return v[:, :, 1] if getattr(v, "ndim", 2) == 3 else v
def importancias():
    return {m: pd.Series(np.abs(np.load(R / f"shap_todo_prob_{m}.npy")).mean(0), index=X_train.columns) for m in ["rf", "xgb", "tabpfn"]}
def consenso():
    sel = {}
    for m, s in importancias().items():
        s = s.sort_values(ascending=False); sel[m] = list(s.index[: int(((s / s.sum()).cumsum() < UMBRAL).sum()) + 1])
    return [v for v in sel["rf"] if v in set(sel["xgb"]) & set(sel["tabpfn"])], {m: len(v) for m, v in sel.items()}, sel
def shap_rf(cols, archivo):
    import shap
    base = joblib.load(MODELS / "modelo_random_forest.pkl"); rf = base if len(cols) == X_train.shape[1] else clone(base).fit(X_train[cols], y_train)
    t0 = time.time(); v = clase1(shap.TreeExplainer(rf).shap_values(X_shap[cols])); p = rf.predict_proba(X_shap[cols])[:, 1]
    np.save(R / archivo, v); print(f"RF ({len(cols)} var): {time.time()-t0:.0f} s | forma {v.shape} | error máx. aditividad {np.abs(v.sum(1)+0.5-p).max():.4f} | AUC test {roc_auc_score(y_test, rf.predict_proba(X_test[cols])[:,1]):.4f}", flush=True)
def shap_xgb(cols, archivo):
    import shap
    base = joblib.load(MODELS / "modelo_xgboost.pkl"); m = base if len(cols) == X_train.shape[1] else clone(base).fit(X_train[cols], y_train)
    m = copy.deepcopy(m).set_params(enable_categorical=False); fondo = X_train.sample(n=N_FONDO, random_state=SEED).index
    t0 = time.time(); e = shap.TreeExplainer(m, data=X_train.loc[fondo, cols], feature_perturbation="interventional", model_output="probability")
    v = clase1(e.shap_values(X_shap[cols], check_additivity=False)); p = m.predict_proba(X_shap[cols])[:, 1]
    err = np.abs(v.sum(1) + np.ravel(e.expected_value)[-1] - p).max(); np.save(R / archivo, v)
    print(f"XGBoost ({len(cols)} var): {time.time()-t0:.0f} s | forma {v.shape} | error máx. aditividad {err:.4f} | AUC test {roc_auc_score(y_test, m.predict_proba(X_test[cols])[:,1]):.4f}", flush=True)
def shap_tabpfn(cols, archivo):
    ruta_token = BASE / ".tabpfn_token"
    if "TABPFN_TOKEN" not in os.environ and ruta_token.exists(): os.environ["TABPFN_TOKEN"] = ruta_token.read_text().strip()
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    from tabpfn_extensions.interpretability.shapiq import get_tabpfn_imputation_explainer
    from tabpfn_extensions.interpretability import shapiq_to_shap_explanation
    import shapiq.explainer.utils as u; u.predict_logits = lambda model, data: model.predict_proba(data)   # valor del juego = probabilidad
    Xtr, Xs = X_train[cols], X_shap[cols]
    ruta = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5).model_path
    modelo = TabPFNClassifier(model_path=ruta, n_estimators=32, balance_probabilities=True, random_state=SEED, device="mps", fit_mode="fit_with_cache").fit(Xtr, y_train)
    explainer = get_tabpfn_imputation_explainer(model=modelo, data=Xtr.values, class_index=1, index="SV", max_order=1, random_state=SEED)
    partes, t0 = [], time.time()
    for i in range(0, len(Xs), CHUNK):
        f = R / f"_chunk_{archivo[:-4]}_{i}.npy"                       # punto de guardado: si se interrumpe, retoma desde aquí
        if f.exists(): partes.append(np.load(f)); print(f"  bloque {i}-{i+CHUNK}: recuperado", flush=True); continue
        v = shapiq_to_shap_explanation(explainer, Xs.values[i:i + CHUNK], budget=256, feature_names=list(cols)).values
        np.save(f, v); partes.append(v); print(f"  bloque {i}-{min(i+CHUNK,len(Xs))} listo | {(time.time()-t0)/60:.0f} min", flush=True)
    v = np.vstack(partes); np.save(R / archivo, v); pp = lambda X: np.concatenate([modelo.predict_proba(X.iloc[j:j+100])[:, 1] for j in range(0, len(X), 100)]); p = pp(Xs); s = v.sum(1); b, _ = np.polyfit(s, p, 1)
    print(f"TabPFN ({len(cols)} var): forma {v.shape} | {(time.time()-t0)/60:.0f} min | rango suma SHAP {s.min():.2f}..{s.max():.2f} | pendiente vs predict_proba = {b:.3f}, r = {np.corrcoef(s,p)[0,1]:.4f} | AUC test {roc_auc_score(y_test, pp(X_test[cols])):.4f}", flush=True)

if modo == "rf43": shap_rf(list(X_train.columns), "shap_todo_prob_rf.npy")
elif modo == "xgb43": shap_xgb(list(X_train.columns), "shap_todo_prob_xgb.npy")
elif modo == "tabpfn43": shap_tabpfn(list(X_train.columns), "shap_todo_prob_tabpfn.npy")
elif modo == "consenso_arboles":
    VARS, tam, sel = consenso(); json.dump({"consenso": VARS, "n85": tam, "paneles": sel}, open(R / "consenso_todo.json", "w"), ensure_ascii=False, indent=1)
    print("Tamaños del 85 %:", tam, "| consenso:", len(VARS), VARS, flush=True)
    shap_rf(VARS, "shap_todo_consenso_prob_rf.npy"); shap_xgb(VARS, "shap_todo_consenso_prob_xgb.npy")
elif modo == "tabpfn_consenso":
    VARS = json.load(open(R / "consenso_todo.json"))["consenso"]; shap_tabpfn(VARS, "shap_todo_consenso_prob_tabpfn.npy")
else: sys.exit(__doc__)
