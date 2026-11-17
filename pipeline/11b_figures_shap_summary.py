"""Step 11b - SHAP summary (beeswarm) plots of TabPFN (Figure 4), random forest (S2) and XGBoost (S3), probability scale,
full training set. Requires step 08 (modes rf43, xgb43, tabpfn43)."""
import locale
import os
import sys
import time
from pathlib import Path

import xgboost  # noqa: F401  (se importa antes que torch; ver nota OpenMP del notebook)
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from config import ROOT as BASE, DATA, DATA as D, MODELS, RESULTS as R, FIGURES as F, FIGURES as FIG


SEED, TARGET, N_SHAP = 42, "RESISTENCIA_INSULINA", 458

try:
    locale.setlocale(locale.LC_NUMERIC, "es_ES.UTF-8")   # decimal comma in the figures (optional)
except locale.Error:
    pass
plt.rcParams.update({"font.family": "Arial", "axes.formatter.use_locale": True})

train = pd.read_csv(D / "train_modelo.csv")
X_train, y_train = train.drop(columns=[TARGET]), train[TARGET].astype(int)
X_shap = X_train  # conjunto de entrenamiento completo (n = 1828)

ETIQUETA = {
    "GENERO": "Sexo femenino", "EDAD": "Edad", "NIVEL_EDUCATIVO": "Nivel educativo", "IMC": "IMC",
    "CINTURA": "Circunferencia de cintura", "BRAZO": "Circunferencia de brazo", "CADERA": "Circunferencia de cadera",
    "PAD": "Presión arterial diastólica", "PAS": "Presión arterial sistólica", "PULSO": "Frecuencia de pulso",
    "COLESTEROL_HDL": "Colesterol HDL", "TRIGLICERIDOS": "Triglicéridos", "COLESTEROL_LDL": "Colesterol LDL",
    "PROTEINA_C_REACTIVA": "Proteína C reactiva (as)", "ALT": "ALT", "AST": "AST", "BUN": "Nitrógeno ureico",
    "CREATININA": "Creatinina", "CPK": "Creatina-fosfocinasa", "HIERRO": "Hierro sérico", "MAGNESIO": "Magnesio",
    "VITAMINA_D": "25-hidroxivitamina D", "DX_HIPERTENSION": "Tensión arterial elevada (≥ 1 ocasión)",
    "DX_HIPERCOLESTEROLEMIA": "Diagnóstico de hipercolesterolemia", "USO_HIPOLIPEMIANTES": "Uso de hipolipemiantes",
    "DX_PREDIABETES": "Diagnóstico de prediabetes", "INGESTA_CALORIAS": "Ingesta de energía",
    "INGESTA_PROTEINA": "Ingesta de proteínas", "INGESTA_CARBOHIDRATOS": "Ingesta de carbohidratos",
    "INGESTA_AZUCAR": "Ingesta de azúcares", "INGESTA_FIBRA": "Ingesta de fibra", "INGESTA_GRASAS": "Ingesta de grasas",
    "TIEMPO_SENTADO": "Tiempo sentado", "MAS_100_CIGARRILLOS": "≥ 100 cigarrillos en la vida",
    "ALCOHOL_VIDA": "Consumo de alcohol alguna vez", "INTENTO_PERDIDA_PESO": "Intento de perder peso",
    "ACTIVIDAD_MODERADA_MIN_SEMANA": "Actividad física moderada",
    "ETNIA_Mexicano-Americano": "Etnia: mexicano-estadounidense", "ETNIA_Otro_Hispano": "Etnia: otro hispano",
    "ETNIA_Blanco_no_Hispano": "Etnia: blanco no hispano", "ETNIA_Negro_no_Hispano": "Etnia: negro no hispano",
    "ETNIA_Asiatico_no_Hispano": "Etnia: asiático no hispano",
    "ETNIA_Otra_raza_Multirracial": "Etnia: otra raza o multirracial",
}
# Variables 1 = sí / 2 = no: solo para colorear se recodifican a 1 = sí / 0 = no (rojo = sí). GENERO: 1 = hombre,
# 2 = mujer -> 1 = mujer / 0 = hombre ("Sexo femenino"). Los valores SHAP no cambian.
BINARIAS_1SI_2NO = ["DX_HIPERTENSION", "DX_HIPERCOLESTEROLEMIA", "USO_HIPOLIPEMIANTES", "DX_PREDIABETES",
                    "MAS_100_CIGARRILLOS", "ALCOHOL_VIDA", "INTENTO_PERDIDA_PESO"]


def valores_para_color(X):
    Xc = X.copy()
    for col in BINARIAS_1SI_2NO:
        Xc[col] = (Xc[col] == 1).astype(float)
    Xc["GENERO"] = (Xc["GENERO"] == 2).astype(float)
    return Xc


def shap_clase_positiva(v):
    if isinstance(v, list):
        return v[1]
    return v[:, :, 1] if v.ndim == 3 else v


def graficar(valores):
    Xc = valores_para_color(X_shap)
    nombres = [ETIQUETA[c] for c in X_shap.columns]
    salidas = {"tabpfn": ("Figura5_shap_tabpfn.png", "probabilidad"), "rf": ("FiguraS2_shap_rf.png", "probabilidad"),
               "xgb": ("FiguraS3_shap_xgb.png", "probabilidad")}
    for m, (archivo, escala) in salidas.items():
        plt.figure()
        shap.summary_plot(valores[m], Xc, feature_names=nombres, max_display=20, show=False,
                          color_bar_label="Valor de la variable", plot_size=(9, 9.5))
        fig = plt.gcf()
        ax = fig.axes[0]
        ax.set_xlabel(f"Valor SHAP (contribución a la predicción de RI, escala {escala})", fontsize=11)
        cbar = fig.axes[-1]
        cbar.set_yticklabels(["Bajo / no", "Alto / sí"])
        cbar.set_ylabel("Valor de la variable", fontsize=11)
        plt.tight_layout()
        plt.savefig(FIG / archivo, dpi=300, bbox_inches="tight")
        plt.close()
        print("guardada", archivo)


valores = {m: np.load(R / f"shap_todo_prob_{m}.npy") for m in ["rf", "xgb", "tabpfn"]}   # step 08
graficar(valores)
