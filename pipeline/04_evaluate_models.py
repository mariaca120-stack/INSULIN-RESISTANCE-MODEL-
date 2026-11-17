"""Paso 04: ajusta TabPFN y evalúa los cuatro modelos en el conjunto de prueba (métricas, curvas ROC, probabilidades)."""
import os
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score, accuracy_score, precision_score,
                             f1_score, brier_score_loss, confusion_matrix, classification_report, ConfusionMatrixDisplay)
from xgboost import XGBClassifier  # noqa: F401  (import before torch: OpenMP on macOS)
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from config import DATA, MODELS, RESULTS, FIGURES, SEED, TARGET, configurar_token_tabpfn, dispositivo_tabpfn

random.seed(SEED)
np.random.seed(SEED)
FIG_DIR, RESULTS_DIR = FIGURES, RESULTS
N_FOLDS = 5
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
ETIQUETAS_CLASES = ["Sin RI", "Con RI"]
TABPFN_DEVICE = dispositivo_tabpfn()
warnings.filterwarnings("ignore", category=FutureWarning)
display = lambda x, *a, **k: print(x)


def guardar_figura(nombre, dpi=300):
    plt.savefig(FIG_DIR / nombre, dpi=dpi, bbox_inches="tight")


train_modelo = pd.read_csv(DATA / "train_modelo.csv")
test_modelo = pd.read_csv(DATA / "test_modelo.csv")
X_train, y_train = train_modelo.drop(columns=[TARGET]), train_modelo[TARGET].astype(int)
X_test, y_test = test_modelo.drop(columns=[TARGET]), test_modelo[TARGET].astype(int)

rf = joblib.load(MODELS / "modelo_random_forest.pkl")
svm = joblib.load(MODELS / "modelo_svm.pkl")
xgb = joblib.load(MODELS / "modelo_xgboost.pkl")
for nombre, modelo in [("RF", rf), ("SVM", svm), ("XGBoost", xgb)]:
    assert list(modelo.feature_names_in_) == list(X_train.columns), f"{nombre}: las columnas no coinciden con los datos"

configurar_token_tabpfn()
RUTA_MODELO_TABPFN = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5).model_path
TABPFN_PARAMS = {"model_path": RUTA_MODELO_TABPFN, "n_estimators": 32, "balance_probabilities": True,
                 "random_state": SEED, "device": TABPFN_DEVICE}
modelo_tabpfn = TabPFNClassifier(**TABPFN_PARAMS)
modelo_tabpfn.fit(X_train, y_train)


MODELOS = {"Random Forest": rf, "SVM": svm, "XGBoost": xgb, "TabPFN": modelo_tabpfn}


def calcular_metricas(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"AUC-ROC": roc_auc_score(y_true, y_prob), "PR-AUC": average_precision_score(y_true, y_prob),
            "Sensibilidad": tp / (tp + fn), "Especificidad": tn / (tn + fp),
            "Accuracy": accuracy_score(y_true, y_pred), "Precisión": precision_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred), "Brier": brier_score_loss(y_true, y_prob)}


def evaluar_en_test(nombre, modelo, X, y, archivo_figura, dpi=300):
    """Reporte de clasificación, matriz de confusión y curva ROC en test. Devuelve predicciones y métricas."""
    y_pred = modelo.predict(X)
    y_prob = modelo.predict_proba(X)[:, 1]
    metricas = calcular_metricas(y, y_pred, y_prob)

    print(f"\n=== CLASIFICACIÓN ({nombre}) ===")
    print(classification_report(y, y_pred, target_names=ETIQUETAS_CLASES))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ConfusionMatrixDisplay(confusion_matrix(y, y_pred), display_labels=ETIQUETAS_CLASES).plot(ax=axes[0], colorbar=False)
    axes[0].set_title(f"Matriz de Confusion - {nombre}")
    fpr, tpr, _ = roc_curve(y, y_prob)
    axes[1].plot(fpr, tpr, label=f"AUC = {metricas['AUC-ROC']:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set_xlabel("Tasa de Falsos Positivos")
    axes[1].set_ylabel("Tasa de Verdaderos Positivos")
    axes[1].set_title(f"Curva ROC - {nombre}")
    axes[1].legend()
    plt.tight_layout()
    guardar_figura(archivo_figura, dpi=dpi)
    plt.show()
    return {"y_pred": y_pred, "y_prob": y_prob, "metricas": metricas}


ARCHIVOS_RESULTADOS = {"Random Forest": "rf_resultados.png", "SVM": "svm_resultados.png",
                       "XGBoost": "xgb_resultados.png", "TabPFN": "tabpfn_resultados.png"}
resultados_test = {nombre: evaluar_en_test(nombre, modelo, X_test, y_test, ARCHIVOS_RESULTADOS[nombre])
                   for nombre, modelo in MODELOS.items()}

tabla_metricas_test = pd.DataFrame({m: r["metricas"] for m, r in resultados_test.items()}).T.round(4)
tabla_metricas_test.to_csv(RESULTS_DIR / "tabla_metricas_test.csv")
tabla_metricas_test

# guardar las probabilidades de prueba
pd.DataFrame({"y_test": y_test.to_numpy(), **{m: r["y_prob"] for m, r in resultados_test.items()}}).to_csv(RESULTS_DIR / "probabilidades_test.csv", index=False)
tabla_metricas_test

plt.figure(figsize=(8, 6))
for nombre, r in resultados_test.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    plt.plot(fpr, tpr, label=f"{nombre} (AUC = {r['metricas']['AUC-ROC']:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Clasificador Aleatorio (AUC = 0.50)")
plt.title("Curvas ROC comparativas entre modelos")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.legend(loc="lower right")
plt.tight_layout()
guardar_figura("comparacion_roc_modelos.png")
plt.show()

modelos_cv = {"RF": rf, "SVM": svm, "XGBoost": xgb, "TabPFN": modelo_tabpfn}
metricas_cv = {"Accuracy": "accuracy", "Precisión": "precision", "Recall": "recall",
               "F1-Score": "f1", "AUC-ROC": "roc_auc"}

filas = []
for nombre, modelo in modelos_cv.items():
    print(f"Evaluando {nombre}...")
    res = cross_validate(clone(modelo), X_train, y_train, cv=cv, scoring=metricas_cv, n_jobs=1, error_score="raise")
    fila = {("", "Modelo"): nombre}
    for metrica in metricas_cv:
        valores = res[f"test_{metrica}"]
        media, sd = valores.mean(), valores.std(ddof=1)
        fila[(metrica, "Promedio")], fila[(metrica, "Sd")], fila[(metrica, "CV")] = round(media, 4), round(sd, 4), round(sd / media, 4)
    filas.append(fila)

tabla_metricas = pd.DataFrame(filas)
tabla_metricas.columns = pd.MultiIndex.from_tuples(tabla_metricas.columns)
tabla_metricas = tabla_metricas.set_index(("", "Modelo"))
tabla_metricas.index.name = "Modelo"
tabla_metricas.to_csv(RESULTS_DIR / "tabla_metricas_cv.csv")
tabla_metricas
