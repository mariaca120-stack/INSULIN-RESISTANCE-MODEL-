"""Paso 11g: histogramas y gráficos Q-Q de las variables cuantitativas."""
import locale
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.diagnostic import lilliefors
from config import ROOT as BASE, DATA, DATA as D, MODELS, RESULTS as R, FIGURES as F, FIGURES as FIG



FIG = F / 'qq'
FIG.mkdir(parents=True, exist_ok=True)
try:
    locale.setlocale(locale.LC_NUMERIC, "es_ES.UTF-8")   # coma decimal en las figuras (opcional)
except locale.Error:
    pass
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.formatter.use_locale": True,
                     "axes.spines.top": False, "axes.spines.right": False})

TARGET = "RESISTENCIA_INSULINA"
datos = pd.concat([pd.read_csv(D / "train_imputado.csv"), pd.read_csv(D / "test_imputado.csv")], ignore_index=True)

VARIABLES = [  # mismo orden y unidades que la Tabla 1
    ("EDAD", "Edad (años)"), ("PESO", "Peso (kg)"), ("ALTURA", "Talla (cm)"), ("IMC", "IMC (kg/m²)"),
    ("CINTURA", "Circunferencia de cintura (cm)"), ("BRAZO", "Circunferencia de brazo (cm)"),
    ("CADERA", "Circunferencia de cadera (cm)"), ("PAS", "Presión arterial sistólica (mmHg)"),
    ("PAD", "Presión arterial diastólica (mmHg)"), ("PULSO", "Frecuencia de pulso (lpm)"),
    ("GLUCOSA", "Glucosa en ayunas (mg/dL)"), ("INSULINA", "Insulina en ayunas (µU/mL)"), ("HOMA_IR", "HOMA-IR"),
    ("COLESTEROL_HDL", "Colesterol HDL (mg/dL)"), ("TRIGLICERIDOS", "Triglicéridos (mg/dL)"),
    ("COLESTEROL_LDL", "Colesterol LDL (mg/dL)"), ("COLESTEROL_TOTAL", "Colesterol total (mg/dL)"),
    ("PROTEINA_C_REACTIVA", "PCR-as (mg/L)"), ("ALT", "ALT (U/L)"), ("AST", "AST (U/L)"),
    ("BUN", "Nitrógeno ureico (mmol/L)"), ("CREATININA", "Creatinina (mg/dL)"), ("CPK", "Creatina-fosfocinasa (U/L)"),
    ("HIERRO", "Hierro sérico (µg/dL)"), ("MAGNESIO", "Magnesio (mg/dL)"), ("VITAMINA_D", "25-hidroxivitamina D (nmol/L)"),
    ("INGESTA_CALORIAS", "Energía (kcal/día)"), ("INGESTA_PROTEINA", "Proteínas (g/día)"),
    ("INGESTA_CARBOHIDRATOS", "Carbohidratos (g/día)"), ("INGESTA_AZUCAR", "Azúcares totales (g/día)"),
    ("INGESTA_FIBRA", "Fibra (g/día)"), ("INGESTA_GRASAS", "Grasas totales (g/día)"),
    ("TIEMPO_SENTADO", "Tiempo sentado (min/día)"), ("ACTIVIDAD_MODERADA_MIN_SEMANA", "Actividad física moderada (min/semana)"),
]
GRUPOS = [(0, "Sin RI", "#4682b4"), (1, "Con RI", "#cd5c5c")]
POR_PAGINA = 4


def p_texto(p):
    return "p < 0,001" if p < 0.001 else f"p = {p:.3f}".replace(".", ",")


resumen = []
for pagina, inicio in enumerate(range(0, len(VARIABLES), POR_PAGINA), start=1):
    bloque = VARIABLES[inicio:inicio + POR_PAGINA]
    fig, axes = plt.subplots(len(bloque), 4, figsize=(10, 2.35 * len(bloque)), squeeze=False)
    for fila, (var, etiqueta) in enumerate(bloque):
        for g, (codigo, nombre, color) in enumerate(GRUPOS):
            serie = datos.loc[datos[TARGET] == codigo, var].dropna().to_numpy()
            d_stat, p = lilliefors(serie, dist="norm")
            resumen.append((var, nombre, d_stat, p))
            ax_h, ax_q = axes[fila, 2 * g], axes[fila, 2 * g + 1]
            ax_h.hist(serie, bins=30, color=color, edgecolor="white", linewidth=0.3)
            ax_h.set_title(f"{nombre} (n = {len(serie)})", fontsize=8)
            ax_h.set_xlabel(etiqueta, fontsize=7.5)
            ax_h.set_ylabel("Frecuencia", fontsize=7.5)
            (osm, osr), (pendiente, intercepto, _) = stats.probplot(serie, dist="norm")
            ax_q.scatter(osm, osr, s=3, color=color, alpha=0.6, linewidths=0)
            ax_q.plot(osm, pendiente * osm + intercepto, color="black", lw=0.8)
            d_txt = f"{d_stat:.3f}".replace(".", ",")
            ax_q.set_title(f"Q-Q {nombre}: D = {d_txt}; {p_texto(p)}", fontsize=8)
            ax_q.set_xlabel("Cuantiles teóricos", fontsize=7.5)
            ax_q.set_ylabel("Cuantiles observados", fontsize=7.5)
        axes[fila, 0].annotate(etiqueta, xy=(-0.42, 0.5), xycoords="axes fraction", rotation=90, ha="center",
                               va="center", fontsize=8.5, fontweight="bold")
    plt.tight_layout(h_pad=1.2, w_pad=0.8)
    plt.savefig(FIG / f"FiguraS7_qq_{pagina:02d}.png", dpi=250, bbox_inches="tight")
    plt.close()
    print(f"página {pagina}: {[v for v, _ in bloque]}")

# comparación con la Tabla S3
ref = pd.read_csv(R / "normalidad_lilliefors.csv")
ref = ref[ref.Conjunto == "Overall"].set_index(["Variable", "Grupo"])["D"]
dif = max(abs(dd - ref[(v, "Sin_RI" if g == "Sin RI" else "Con_RI")]) for v, g, dd, _ in resumen)
print(f"diferencia máxima del estadístico D frente a la Tabla S3: {dif:.4f}")
