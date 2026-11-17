"""Paso 02: selecciona la cohorte, limpia los datos, define la resistencia a la insulina (HOMA-IR > 2,5),
divide en entrenamiento y prueba (80/20), imputa los datos faltantes con MICE y genera la Tabla 1.
Los valores imputados pueden cambiar un poco entre ejecuciones. Para obtener los números exactos del artículo,
usa los archivos que ya están en data/processed/ y empieza desde el paso 04."""
import os
import time
import random
import warnings
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import scipy.stats as stats
from scipy.stats import spearmanr
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from pyampute.exploration.mcar_statistical_tests import MCARTest
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
import miceforest as mf
import joblib

from config import DATA, DATA_RAW, RESULTS, FIGURES, SEED

display = lambda x, *a, **k: print(x)   # el código original corría en un cuaderno

# seed
random.seed(SEED)
np.random.seed(SEED)

# paths
RUTA_VARIABLES = DATA / "nhanes_variables.csv"      # paso 01
NHANES_DIR     = DATA_RAW                           # archivos .xpt originales (paso 00)
RUTA_DIQ       = NHANES_DIR / "DIQ_L.xpt"           # diabetes questionnaire
RESULTS_DIR    = RESULTS
FIG_DIR        = FIGURES

# cohort
EDAD_MINIMA              = 18
GLUCOSA_MAXIMA           = 126        # mg/dL; se conservan los participantes con glucosa <= 126
UMBRAL_FALTANTES_FILA    = 20         # % máximo de datos faltantes por participante
UMBRAL_FALTANTES_COLUMNA = 20         # % máximo de datos faltantes por variable

# códigos de NHANES para 'se negó a responder' / 'no sabe' (solo variables de cuestionario)
CODIGOS_FALTANTES_POR_VARIABLE = {
    **{v: [7, 9] for v in ["NIVEL_EDUCATIVO", "DX_HIPERTENSION", "DX_HTA", "USO_ANTIHIPERTENSIVOS",
                           "DX_HIPERCOLESTEROLEMIA", "USO_HIPOLIPEMIANTES", "DX_PREDIABETES",
                           "MAS_100_CIGARRILLOS", "FRECUENCIA_CONSUMO", "ALCOHOL_VIDA", "INTENTO_PERDIDA_PESO"]},
    "ALCOHOL_12_MESES": [77, 99],      # ALQ121: escala 0–10, 7 y 9 son categorías válidas
    "NUMERO_BEBIDAS":   [777, 999],    # ALQ130: número de bebidas, 7 y 9 son valores válidos
    **{v: [7777, 9999] for v in ["FRECUENCIA_ACTIVIDAD_MODERADA", "DURACION_ACTIVIDAD_MODERADA",
                                 "FRECUENCIA_ACTIVIDAD_VIGOROSA", "DURACION_ACTIVIDAD_VIGOROSA",
                                 "TIEMPO_SENTADO"]},
}
# read_sas lee los ceros como 5.4e-79: se restauran a 0
UMBRAL_CERO_SAS = 1e-10

# outcome
TARGET        = "RESISTENCIA_INSULINA"
HOMA_DIVISOR  = 405
HOMA_IR_CORTE = 2.5                   # RI = 1 si HOMA-IR > 2.5

# partición, validación cruzada y búsqueda
TEST_SIZE       = 0.2
N_FOLDS         = 5
N_ITER_BUSQUEDA = 50
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# MICE imputation
MICE_N_DATASETS    = 5
MICE_ITERACIONES   = 5
MICE_DATASET_USADO = 0

# descriptive statistics
ALFA = 0.05
N_BOOTSTRAP_NORMALIDAD = 2000

# multicollinearity
UMBRAL_VIF = 10
VARIABLES_PROTEGIDAS_VIF = ["IMC"]

# SHAP
N_SHAP = 458                          # mismo tamaño que el test (costo computacional de TabPFN)
BUDGET_SHAP_TABPFN = 256
UMBRAL_IMPORTANCIA_ACUMULADA = 0.85
N_TOP_HEATMAP = 15

# curva de decisións
UMBRALES_DCA = np.round(np.arange(0.01, 1.00, 0.01), 2)
UMBRALES_DCA_TABLA = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

ETIQUETAS_CLASES = ["Sin RI", "Con RI"]

# display options
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 250)
warnings.filterwarnings("ignore", category=FutureWarning)


def guardar_figura(nombre, dpi=300):
    """Guarda la figura actual en figures/."""
    plt.savefig(FIG_DIR / nombre, dpi=dpi, bbox_inches="tight")

FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RUTA_VARIABLES)
diabetes = pd.read_sas(RUTA_DIQ, format="xport", encoding="utf-8")[["SEQN", "DIQ010", "DIQ050", "DIQ070"]]

print(f"nhanes_variables.csv: {df.shape[0]} participantes × {df.shape[1]} columnas")
print(f"DIQ_L.xpt:            {diabetes.shape[0]} participantes")
assert df["SEQN"].is_unique, "SEQN duplicado en la base consolidada"

flujo_cohorte = [{"Paso": "Participantes en nhanes_variables.csv", "Excluidos": 0, "n restante": len(df)}]


def aplicar_criterio(datos, conservar, descripcion):
    """Conserva las filas donde `conservar` es True y registra cuántas se excluyeron."""
    n_antes = len(datos)
    datos = datos[conservar].copy()
    flujo_cohorte.append({"Paso": descripcion, "Excluidos": n_antes - len(datos), "n restante": len(datos)})
    return datos


# inclusión: insulina y glucosa en ayunas
muestra = aplicar_criterio(df, df[["INSULINA", "GLUCOSA"]].notna().all(axis=1),
                           "Sin insulina o glucosa en ayuno")

# exclusion: clinical criteria
muestra = muestra.merge(diabetes, on="SEQN", how="left")
muestra = aplicar_criterio(muestra, muestra["EDAD"] >= EDAD_MINIMA, "Menores de 18 años")
muestra = aplicar_criterio(muestra, muestra["DIQ010"] != 1, "Diagnóstico previo de diabetes (DIQ010 = 1)")
muestra = aplicar_criterio(muestra, muestra["DIQ050"] != 1, "Uso de insulina (DIQ050 = 1)")
muestra = aplicar_criterio(muestra, muestra["DIQ070"] != 1, "Uso de hipoglucemiantes (DIQ070 = 1)")
muestra = aplicar_criterio(muestra, muestra["GLUCOSA"] <= GLUCOSA_MAXIMA, "Glucosa en ayuno > 126 mg/dL")
muestra = muestra.drop(columns=["DIQ010", "DIQ050", "DIQ070"])
pd.DataFrame(flujo_cohorte)

conteo_codigos = {}
for col, codigos in CODIGOS_FALTANTES_POR_VARIABLE.items():
    if col in muestra.columns:
        es_codigo = muestra[col].isin(codigos)
        conteo_codigos[col] = int(es_codigo.sum())
        muestra[col] = muestra[col].mask(es_codigo)

print("Códigos de no respuesta convertidos en NaN:")
print(pd.Series(conteo_codigos)[lambda s: s > 0].to_string())

# se conservan los códigos 7/9/7777/9999
otras_numericas = [c for c in muestra.select_dtypes(include=[np.number]).columns
                   if c not in CODIGOS_FALTANTES_POR_VARIABLE and c not in ["SEQN", "ETNIA"]]
conservados = muestra[otras_numericas].isin([7, 9, 7777, 9999]).sum()
print("\nMediciones reales iguales a 7/9/7777/9999 que se conservan:")
print(conservados[conservados > 0].to_string())

columnas_numericas = muestra.select_dtypes(include=[np.number]).columns
es_cero_sas = (muestra[columnas_numericas].abs() < UMBRAL_CERO_SAS) & (muestra[columnas_numericas] != 0)

print("Ceros SAS restaurados a 0, por variable:")
print(es_cero_sas.sum()[lambda s: s > 0].to_string())
muestra[columnas_numericas] = muestra[columnas_numericas].mask(es_cero_sas, 0.0)

# las ingestas de 0 kcal se informan, no se modifican
n_cero_kcal = (muestra["INGESTA_CALORIAS"] == 0).sum()
if n_cero_kcal:
    print(f"\n⚠ {n_cero_kcal} participantes con INGESTA_CALORIAS = 0 (recordatorio posiblemente no válido).")

pct_faltantes_fila = muestra.isnull().mean(axis=1) * 100
muestra = aplicar_criterio(muestra, pct_faltantes_fila <= UMBRAL_FALTANTES_FILA,
                           f"> {UMBRAL_FALTANTES_FILA}% de datos faltantes")

flujo_cohorte = pd.DataFrame(flujo_cohorte)
flujo_cohorte

resumen_faltantes = pd.DataFrame({
    "n_faltantes": muestra.isna().sum(),
    "porcentaje_faltantes": (muestra.isna().mean() * 100).round(2),
}).sort_values("porcentaje_faltantes", ascending=False)
print(f"Variables con al menos un faltante: {(resumen_faltantes['n_faltantes'] > 0).sum()} de {muestra.shape[1]}")
display(resumen_faltantes[resumen_faltantes["n_faltantes"] > 0])
resumen_faltantes.rename_axis('variable').to_csv(RESULTS_DIR / 'resumen_faltantes.csv')

faltantes_por_columna = muestra.isnull().mean() * 100
columnas_excluidas_faltantes = faltantes_por_columna[faltantes_por_columna > UMBRAL_FALTANTES_COLUMNA].index.tolist()
columnas_excluidas = list(dict.fromkeys(columnas_excluidas_faltantes + ["FRECUENCIA_ACTIVIDAD_VIGOROSA"]))

dataset_final = muestra.drop(columns=columnas_excluidas)
print(f"Variables excluidas por > {UMBRAL_FALTANTES_COLUMNA}% de faltantes ({len(columnas_excluidas_faltantes)}): "
      f"{columnas_excluidas_faltantes}")
print(f"dataset_final: {dataset_final.shape[0]} filas × {dataset_final.shape[1]} columnas")

frecuencia = dataset_final["FRECUENCIA_ACTIVIDAD_MODERADA"]
duracion = dataset_final["DURACION_ACTIVIDAD_MODERADA"]
unidad = dataset_final["UNIDAD_ACTIVIDAD_MODERADA"]
print(unidad.value_counts(dropna=False).to_string(), "\n")

FACTOR_A_SEMANA = {"D": 7, "W": 1, "M": 1 / 4.345, "Y": 1 / 52}
dataset_final["ACTIVIDAD_MODERADA_MIN_SEMANA"] = np.where(
    frecuencia == 0, 0.0, frecuencia * duracion * unidad.map(FACTOR_A_SEMANA)
)
dataset_final = dataset_final.drop(columns=["FRECUENCIA_ACTIVIDAD_MODERADA", "UNIDAD_ACTIVIDAD_MODERADA",
                                            "DURACION_ACTIVIDAD_MODERADA"])

print(f"Participantes con 0 min/semana: {(dataset_final['ACTIVIDAD_MODERADA_MIN_SEMANA'] == 0).sum()}")
print(dataset_final["ACTIVIDAD_MODERADA_MIN_SEMANA"].describe().round(2).to_string())

dataset_final.to_csv(DATA / "dataset_final.csv", index=False)

variables_varianza_cero = dataset_final.columns[dataset_final.nunique(dropna=True) <= 1].tolist()
df_para_mcar = dataset_final.drop(columns=variables_varianza_cero + ["SEQN"])

p_mcar = MCARTest(method="little")(df_para_mcar)
print(f"Variables en el test: {df_para_mcar.shape[1]} (excluidas: {variables_varianza_cero + ['SEQN']})")
print(f"Test de Little: p = {p_mcar:.4g} → "
      + ("se rechaza MCAR" if p_mcar < ALFA else "no se rechaza MCAR"))

dataset_final["HOMA_IR"] = (dataset_final["GLUCOSA"] * dataset_final["INSULINA"]) / HOMA_DIVISOR
assert dataset_final["HOMA_IR"].notna().all(), "Hay participantes sin HOMA-IR: el outcome quedaría mal definido"
dataset_final[TARGET] = (dataset_final["HOMA_IR"] > HOMA_IR_CORTE).astype(int)

print(dataset_final[["GLUCOSA", "INSULINA", "HOMA_IR"]].describe().round(2).T.to_string(), "\n")
distribucion = dataset_final[TARGET].value_counts().sort_index()
print(pd.DataFrame({"n": distribucion, "%": (distribucion / len(dataset_final) * 100).round(1)}).to_string())

# domain: (categorical, quantitative)
VARIABLES_POR_DOMINIO = {
    "Demográficas":             (["GENERO", "ETNIA", "NIVEL_EDUCATIVO"], ["EDAD"]),
    "Antropométricas":          ([], ["PESO", "ALTURA", "IMC", "CINTURA", "BRAZO", "CADERA"]),
    "Clínicas":                 (["DX_HIPERTENSION", "DX_HIPERCOLESTEROLEMIA", "USO_HIPOLIPEMIANTES", "DX_PREDIABETES"],
                                 ["PAD", "PAS", "PULSO"]),
    "Componentes del desenlace": ([], ["GLUCOSA", "INSULINA", "HOMA_IR"]),
    "Bioquímicas":              ([], ["COLESTEROL_HDL", "TRIGLICERIDOS", "COLESTEROL_LDL", "COLESTEROL_TOTAL",
                                      "PROTEINA_C_REACTIVA", "ALT", "AST", "BUN", "CREATININA", "CPK",
                                      "HIERRO", "MAGNESIO", "VITAMINA_D"]),
    "Nutricionales":            ([], ["INGESTA_CALORIAS", "INGESTA_PROTEINA", "INGESTA_CARBOHIDRATOS",
                                      "INGESTA_AZUCAR", "INGESTA_FIBRA", "INGESTA_GRASAS"]),
    "Estilo de vida":           (["MAS_100_CIGARRILLOS", "ALCOHOL_VIDA", "INTENTO_PERDIDA_PESO"],
                                 ["TIEMPO_SENTADO", "ACTIVIDAD_MODERADA_MIN_SEMANA"]),
}
VARS_CATEGORICAS   = [v for categoricas, _ in VARIABLES_POR_DOMINIO.values() for v in categoricas]
VARS_CUANTITATIVAS = [v for _, cuantitativas in VARIABLES_POR_DOMINIO.values() for v in cuantitativas]

# glucosa, insulina y HOMA-IR definen el desenlace: solo se describen, nunca son predictores
COLUMNAS_DESENLACE = ["GLUCOSA", "INSULINA", "HOMA_IR"]
VARS_CUANTITATIVAS_PREDICTORAS = [v for v in VARS_CUANTITATIVAS if v not in COLUMNAS_DESENLACE]

ETIQUETAS = {
    "GENERO": {1: "Hombre", 2: "Mujer"},
    "ETNIA": {1: "Mexicano-Americano", 2: "Otro Hispano", 3: "Blanco no Hispano",
              4: "Negro no Hispano", 6: "Asiatico no Hispano", 7: "Otra raza / Multirracial"},
    "NIVEL_EDUCATIVO": {1: "Menos de 9° grado", 2: "9°-11° grado (incompleto)", 3: "Bachillerato / GED",
                        4: "Universidad incompleta / AA", 5: "Universidad completa o superior"},
    **{v: {1: "Si", 2: "No"} for v in ["DX_HIPERTENSION", "DX_HIPERCOLESTEROLEMIA", "USO_HIPOLIPEMIANTES",
                                       "DX_PREDIABETES", "MAS_100_CIGARRILLOS", "ALCOHOL_VIDA",
                                       "INTENTO_PERDIDA_PESO"]},
}

# all study variables must be present
faltan = set(VARS_CATEGORICAS + VARS_CUANTITATIVAS) - set(dataset_final.columns)
assert not faltan, f"Variables del estudio excluidas por faltantes: {faltan}"

# variables que pasan el filtro de faltantes pero no están en el conjunto de la tesis: se excluyen
no_previstas = sorted(set(dataset_final.columns) - set(VARS_CATEGORICAS + VARS_CUANTITATIVAS) - {"SEQN", TARGET})
if no_previstas:
    print(f"⚠ Variables no previstas que superan el filtro de faltantes (se excluyen): {no_previstas}")
    dataset_final = dataset_final.drop(columns=no_previstas)

pd.DataFrame([(dominio, "Categórica", v) for dominio, (cat, _) in VARIABLES_POR_DOMINIO.items() for v in cat] +
             [(dominio, "Cuantitativa", v) for dominio, (_, cuant) in VARIABLES_POR_DOMINIO.items() for v in cuant],
             columns=["Dominio", "Tipo", "Variable"]).sort_values(["Dominio", "Tipo"], kind="stable")

train_df, test_df = train_test_split(
    dataset_final, test_size=TEST_SIZE, stratify=dataset_final[TARGET], random_state=SEED
)

print(f"Train: {len(train_df)} ({len(train_df)/len(dataset_final):.1%})  |  Test: {len(test_df)} ({len(test_df)/len(dataset_final):.1%})")
pd.DataFrame({"Train": train_df[TARGET].value_counts(normalize=True).round(3),
              "Test": test_df[TARGET].value_counts(normalize=True).round(3)})

COLUMNAS_FUERA_DE_IMPUTACION = ["SEQN", TARGET] + COLUMNAS_DESENLACE
assert train_df[COLUMNAS_DESENLACE + [TARGET]].notna().all().all()

train_predictores = train_df.drop(columns=COLUMNAS_FUERA_DE_IMPUTACION).reset_index(drop=True)
test_predictores = test_df.drop(columns=COLUMNAS_FUERA_DE_IMPUTACION).reset_index(drop=True)

# modelo de imputación ajustado solo con los predictores del entrenamiento
kernel = mf.ImputationKernel(train_predictores, num_datasets=MICE_N_DATASETS, random_state=SEED)
kernel.mice(iterations=MICE_ITERACIONES)
test_kernel = kernel.impute_new_data(new_data=test_predictores, random_state=SEED)


def reincorporar_desenlace(imputado, original):
    """Agrega de nuevo el desenlace y sus componentes (sin imputar), en el orden original de columnas."""
    completo = imputado.copy()
    for col in COLUMNAS_DESENLACE + [TARGET]:
        completo[col] = original[col].to_numpy()
    return completo[[c for c in original.columns if c != "SEQN"]]


train_imputado = reincorporar_desenlace(kernel.complete_data(dataset=MICE_DATASET_USADO), train_df)
test_imputado = reincorporar_desenlace(test_kernel.complete_data(dataset=MICE_DATASET_USADO), test_df)
print(f"Train imputado: {train_imputado.shape} | NaN restantes: {train_imputado.isna().sum().sum()}")
print(f"Test imputado:  {test_imputado.shape} | NaN restantes: {test_imputado.isna().sum().sum()}")

# NaN restantes en prueba: se completan con la mediana (continuas) o la moda (binarias) del entrenamiento
for col in test_imputado.columns[test_imputado.isna().any()]:
    n_afectados = test_imputado[col].isna().sum()
    if pd.api.types.is_numeric_dtype(train_imputado[col]) and train_imputado[col].nunique() > 2:
        valor_relleno, metodo = train_imputado[col].median(), "mediana"
    else:
        valor_relleno, metodo = train_imputado[col].mode()[0], "moda"
    test_imputado[col] = test_imputado[col].fillna(valor_relleno)
    print(f"{col}: {n_afectados} valor(es) rellenado(s) con {valor_relleno} ({metodo} de train)")

assert test_imputado.isna().sum().sum() == 0

train_imputado.to_csv(DATA / "train_imputado.csv", index=False)
test_imputado.to_csv(DATA / "test_imputado.csv", index=False)

muestra_analitica = pd.concat([train_imputado, test_imputado], axis=0, ignore_index=True)
print(f"Muestra analítica: {muestra_analitica.shape[0]} participantes")
print(muestra_analitica[TARGET].value_counts().sort_index().to_string())

def normalidad_lilliefors(serie, n_bootstrap=N_BOOTSTRAP_NORMALIDAD, seed=SEED):
    """Estadístico D de Lilliefors y valor p (con IC del 95 % por bootstrap de D si n_bootstrap > 0)."""
    datos = serie.dropna().values
    d_stat, p_valor = lilliefors(datos, dist="norm")
    resultado = {"D": d_stat, "p_valor": p_valor, "es_normal": p_valor > ALFA}
    if n_bootstrap:
        rng = np.random.default_rng(seed)
        boot = [lilliefors(rng.choice(datos, size=len(datos), replace=True), dist="norm")[0]
                for _ in range(n_bootstrap)]
        resultado["IC_95_inf"], resultado["IC_95_sup"] = np.percentile(boot, [2.5, 97.5])
    return resultado


def comparar_categorica(tabla):
    """Chi-cuadrado (prueba exacta de Fisher en tablas 2x2 con frecuencia esperada < 5)."""
    _, p_chi2, _, esperadas = stats.chi2_contingency(tabla)
    if tabla.shape == (2, 2) and (esperadas < 5).any():
        return stats.fisher_exact(tabla)[1], "Fisher exacto"
    return p_chi2, "Chi-cuadrado"


def comparar_cuantitativa(grupo_a, grupo_b):
    """Prueba t (Student o Welch) si ambos grupos son normales; si no, Mann-Whitney U."""
    grupo_a, grupo_b = grupo_a.dropna(), grupo_b.dropna()
    es_normal = (normalidad_lilliefors(grupo_a, n_bootstrap=0)["es_normal"]
                 and normalidad_lilliefors(grupo_b, n_bootstrap=0)["es_normal"])
    if es_normal:
        varianzas_iguales = stats.levene(grupo_a, grupo_b)[1] > ALFA
        p = stats.ttest_ind(grupo_a, grupo_b, equal_var=varianzas_iguales)[1]
        return p, ("t-Student" if varianzas_iguales else "t-Welch"), True
    return stats.mannwhitneyu(grupo_a, grupo_b, alternative="two-sided")[1], "Mann-Whitney", False


def resumir_cuantitativa(serie, es_normal):
    """Mean ± SD if normal, otherwise median [IQR]."""
    serie = serie.dropna()
    if es_normal:
        return f"{serie.mean():.2f} ± {serie.std():.2f}"
    q1, mediana, q3 = serie.quantile([0.25, 0.5, 0.75])
    return f"{mediana:.2f} [{q1:.2f}–{q3:.2f}]"


def resumir_categoria(serie, codigo):
    """n (%) de la categoría `codigo` en el grupo."""
    n, total = (serie == codigo).sum(), len(serie)
    return f"{n} ({n / total * 100:.1f}%)" if total else "0 (0.0%)"


def grupos_por_desenlace(df, var):
    return df.loc[df[TARGET] == 0, var], df.loc[df[TARGET] == 1, var]

def graficar_normalidad(df, var):
    grupo0, grupo1 = (g.dropna() for g in grupos_por_desenlace(df, var))
    r0, r1 = normalidad_lilliefors(grupo0, n_bootstrap=0), normalidad_lilliefors(grupo1, n_bootstrap=0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Evaluación visual de normalidad: {var}", fontsize=13)
    axes[0, 0].hist(grupo0, bins=30, color="steelblue", edgecolor="white")
    axes[0, 0].set_title("Sin RI - Histograma")
    axes[0, 1].hist(grupo1, bins=30, color="indianred", edgecolor="white")
    axes[0, 1].set_title("Con RI - Histograma")
    for ax, grupo, r, nombre in [(axes[1, 0], grupo0, r0, "Sin RI"), (axes[1, 1], grupo1, r1, "Con RI")]:
        stats.probplot(grupo, dist="norm", plot=ax)
        p_str = "<.001" if r["p_valor"] < 0.001 else f"{r['p_valor']:.4f}"
        ax.set_title(f"{nombre} - QQ-plot\nD={r['D']:.3f}, p={p_str}")
    plt.tight_layout()
    guardar_figura(f"qq_{var}.png", dpi=200)
    plt.show()
    plt.close()


for var in VARS_CUANTITATIVAS:
    graficar_normalidad(train_imputado, var)

# normality table (Lilliefors)
filas_normalidad = []
for var in VARS_CUANTITATIVAS:
    for conjunto, df_ref in [("Overall", muestra_analitica), ("Train", train_imputado), ("Test", test_imputado)]:
        for grupo, serie in zip(["Sin_RI", "Con_RI"], grupos_por_desenlace(df_ref, var)):
            if serie.dropna().shape[0] < 4:
                continue
            r = normalidad_lilliefors(serie)
            filas_normalidad.append({
                "Variable": var, "Conjunto": conjunto, "Grupo": grupo, "n": serie.dropna().shape[0],
                "D": round(r["D"], 4), "IC_95_inf": round(r["IC_95_inf"], 4), "IC_95_sup": round(r["IC_95_sup"], 4),
                "p_valor": "<.001" if r["p_valor"] < 0.001 else round(r["p_valor"], 4),
                "Normal": "Sí" if r["es_normal"] else "No",
            })

tabla_normalidad = pd.DataFrame(filas_normalidad)
tabla_normalidad.to_csv(RESULTS_DIR / "normalidad_lilliefors.csv", index=False)
tabla_normalidad

def construir_tabla_1(df, prefijo):
    """Resumen por grupo (Sin RI / Con RI) y valor p de la asociación con el desenlace."""
    filas = []
    for var in VARS_CATEGORICAS:
        p, prueba = comparar_categorica(pd.crosstab(df[var], df[TARGET]))
        g0, g1 = grupos_por_desenlace(df, var)
        for i, (codigo, etiqueta) in enumerate(ETIQUETAS[var].items()):
            filas.append({"Variable": var if i == 0 else "", "Categoria": etiqueta,
                          f"{prefijo}_Sin_RI": resumir_categoria(g0, codigo),
                          f"{prefijo}_Con_RI": resumir_categoria(g1, codigo),
                          f"p_{prefijo.lower()}": round(p, 4) if i == 0 else "",
                          f"Test_usado_{prefijo.lower()}": prueba if i == 0 else ""})
    for var in VARS_CUANTITATIVAS:
        g0, g1 = grupos_por_desenlace(df, var)
        p, prueba, es_normal = comparar_cuantitativa(g0, g1)
        filas.append({"Variable": var, "Categoria": "-",
                      f"{prefijo}_Sin_RI": resumir_cuantitativa(g0, es_normal),
                      f"{prefijo}_Con_RI": resumir_cuantitativa(g1, es_normal),
                      f"p_{prefijo.lower()}": round(p, 4), f"Test_usado_{prefijo.lower()}": prueba})
    return pd.DataFrame(filas)


tabla1_overall = construir_tabla_1(muestra_analitica, "Overall")
tabla1_overall.to_csv(RESULTS_DIR / "tabla1_overall.csv", index=False)

# Table 1 (overall, train, test)
tabla1_extendida = pd.concat(
    [tabla1_overall,
     construir_tabla_1(train_imputado, "Train").drop(columns=["Variable", "Categoria"]),
     construir_tabla_1(test_imputado, "Test").drop(columns=["Variable", "Categoria"])],
    axis=1,
)
tabla1_overall

conjunto = pd.Series(np.r_[np.zeros(len(train_imputado)), np.ones(len(test_imputado))], name="CONJUNTO")

filas_traintest = []
for var in VARS_CATEGORICAS:
    p, prueba = comparar_categorica(pd.crosstab(muestra_analitica[var], conjunto))
    for i, (codigo, etiqueta) in enumerate(ETIQUETAS[var].items()):
        filas_traintest.append({"Variable": var if i == 0 else "", "Categoria": etiqueta,
                                "Train": resumir_categoria(train_imputado[var], codigo),
                                "Test": resumir_categoria(test_imputado[var], codigo),
                                "p": round(p, 4) if i == 0 else "", "Test_usado": prueba if i == 0 else ""})
for var in VARS_CUANTITATIVAS:
    p, prueba, es_normal = comparar_cuantitativa(train_imputado[var], test_imputado[var])
    filas_traintest.append({"Variable": var, "Categoria": "-",
                            "Train": resumir_cuantitativa(train_imputado[var], es_normal),
                            "Test": resumir_cuantitativa(test_imputado[var], es_normal),
                            "p": round(p, 4), "Test_usado": prueba})

tabla_traintest = pd.DataFrame(filas_traintest)
tabla_traintest.to_csv(RESULTS_DIR / "equivalencia_train_test.csv", index=False)
tabla_traintest

def calcular_vif(df, variables):
    X = add_constant(df[variables])
    vif = pd.DataFrame({"Variable": X.columns,
                        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
    return vif[vif["Variable"] != "const"].sort_values("VIF", ascending=False).reset_index(drop=True)


def eliminar_multicolinealidad(df, variables, umbral, protegidas):
    """Elimina iterativamente la variable (no protegida) con mayor VIF hasta que todas queden <= umbral."""
    variables, eliminadas = list(variables), []
    while True:
        vif = calcular_vif(df, variables)
        candidatas = vif[~vif["Variable"].isin(protegidas)]
        if candidatas.empty or candidatas.iloc[0]["VIF"] <= umbral:
            return variables, eliminadas, vif
        var, valor = candidatas.iloc[0]["Variable"], candidatas.iloc[0]["VIF"]
        eliminadas.append({"Variable": var, "VIF": round(valor, 2)})
        variables.remove(var)


variables_vif, eliminadas_vif, vif_final = eliminar_multicolinealidad(
    train_imputado, VARS_CUANTITATIVAS_PREDICTORAS, UMBRAL_VIF, VARIABLES_PROTEGIDAS_VIF
)
COLUMNAS_ELIMINADAS_VIF = [e["Variable"] for e in eliminadas_vif]
print(f"Variables cuantitativas evaluadas: {len(VARS_CUANTITATIVAS_PREDICTORAS)}")
print(f"Eliminadas por VIF > {UMBRAL_VIF}: {eliminadas_vif}")
# la altura se excluye (está implícita en el IMC)
COLUMNAS_ELIMINADAS_VIF = COLUMNAS_ELIMINADAS_VIF + ["ALTURA"]
if set(COLUMNAS_ELIMINADAS_VIF) != {"COLESTEROL_TOTAL", "PESO", "ALTURA"}:
    print("⚠ Difiere de la versión original, que excluía COLESTEROL_TOTAL, PESO y ALTURA.")
vif_final

COLUMNAS_NO_PREDICTORAS = COLUMNAS_DESENLACE + COLUMNAS_ELIMINADAS_VIF + [TARGET]
CATEGORIAS_ETNIA = {float(codigo): "ETNIA_" + etiqueta.replace(" / ", "_").replace(" ", "_")
                    for codigo, etiqueta in ETIQUETAS["ETNIA"].items()}


def preparar_predictores(df):
    """Quita outcome y variables eliminadas por VIF; codifica ETNIA con una columna indicadora por categoría."""
    X = df.drop(columns=COLUMNAS_NO_PREDICTORAS)
    etnia = X.pop("ETNIA")
    assert etnia.isin(CATEGORIAS_ETNIA).all(), "Código de ETNIA no previsto"
    for codigo, nombre in CATEGORIAS_ETNIA.items():
        X[nombre] = (etnia == codigo).astype(float)
    return X


X_train, X_test = preparar_predictores(train_imputado), preparar_predictores(test_imputado)
y_train = train_imputado[TARGET].astype(int)
y_test = test_imputado[TARGET].astype(int)

pd.concat([X_train, y_train], axis=1).to_csv(DATA / "train_modelo.csv", index=False)
pd.concat([X_test, y_test], axis=1).to_csv(DATA / "test_modelo.csv", index=False)

print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"Predictores ({X_train.shape[1]}): {list(X_train.columns)}")
