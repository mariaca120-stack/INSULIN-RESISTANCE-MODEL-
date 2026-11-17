# Clasificación de la resistencia a la insulina con aprendizaje automático (NHANES 2021–2023)

Este repositorio contiene el código de un estudio transversal que desarrolla y compara modelos de aprendizaje
automático para clasificar la resistencia a la insulina (RI, definida como HOMA-IR > 2,5) en adultos de Estados
Unidos, sin usar la insulina ni la glucosa en ayunas como predictores.

## Resumen del estudio

- **Datos:** NHANES 2021–2023 (públicos). 2285 adultos sin diabetes diagnosticada (prevalencia de RI 41,7 %),
  divididos en entrenamiento (n = 1828) y prueba (n = 457) antes de cualquier preprocesamiento.
- **Modelos:** random forest, SVM, XGBoost y TabPFN (v2.5), comparados por discriminación (AUC-ROC), calibración
  y análisis de curvas de decisión. Las probabilidades se recalibraron con un intercepto y una pendiente logísticos
  ajustados con predicciones de validación cruzada del entrenamiento.
- **Interpretabilidad:** valores SHAP (escala de probabilidad) calculados con todo el conjunto de entrenamiento.
  El panel de consenso de 10 variables es la intersección de las variables que alcanzan el 85 % de la importancia
  SHAP acumulada en los tres modelos.
- **Comparador:** índice triglicéridos-glucosa (TyG).
- **Reporte:** TRIPOD+AI.

## Resultados principales

En el conjunto de prueba, TabPFN tuvo el mayor AUC-ROC (0,861; IC 95 % 0,827–0,894), seguido de XGBoost (0,850),
random forest (0,841) y SVM (0,841), todos por encima del índice TyG (0,762). El panel de 10 variables redujo
levemente la discriminación (AUC de TabPFN 0,849).

## Cómo ejecutarlo

Todo se ejecuta desde este repositorio: no hace falta Google Drive ni descargas manuales.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

./run_all.sh                  # reproduce el artículo con los datos y modelos de este repositorio (pasos 04-11)
./run_all.sh --retrain        # además repite la búsqueda de hiperparámetros (paso 03) y sobrescribe models/
./run_all.sh --from-scratch   # además descarga NHANES de los CDC y reconstruye los datos (pasos 00-02, luego 03)
```

Cada paso también se puede ejecutar por separado, por ejemplo `python pipeline/05_test_metrics_ci.py`.
Todas las rutas están definidas en `pipeline/config.py` (relativas a la carpeta del repositorio).

| Paso | Script | Qué hace | Resultado principal |
|---|---|---|---|
| 00 | `pipeline/00_download_nhanes.py` | Descarga los módulos `.xpt` de NHANES 2021–2023 desde los CDC | `data/raw/` |
| 01 | `pipeline/01_build_dataset.py` | Une los módulos por `SEQN` y renombra las variables | `data/processed/nhanes_variables.csv` |
| 02 | `pipeline/02_prepare_data.py` | Cohorte, limpieza, desenlace HOMA-IR, partición 80/20, imputación MICE, Tabla 1, VIF | `data/processed/train_modelo.csv`, `test_modelo.csv` |
| 03 | `pipeline/03_train_models.py` | *(opcional)* Random forest, SVM y XGBoost: búsqueda aleatoria con validación cruzada de 5 particiones | `models/*.pkl` |
| 04 | `pipeline/04_evaluate_models.py` | Ajusta TabPFN; métricas de prueba, ROC, validación cruzada | `results/probabilidades_test.csv`, `tabla_metricas_*.csv` |
| 05 | `pipeline/05_test_metrics_ci.py` | IC del 95 % (DeLong, bootstrap), comparación de DeLong, beneficio neto | `results/metricas_test_ic95.csv` |
| 06 | `pipeline/06_recalibration.py` | Recalibración logística con predicciones de validación cruzada del entrenamiento | `results/probabilidades_test_recal.csv` |
| 07 | `pipeline/07_recalibrated_metrics_ci.py` | Métricas recalibradas con IC por bootstrap | `results/metricas_test_ic95_recal.csv` |
| 08 | `pipeline/08_shap_full_training_set.py` | SHAP (escala de probabilidad) con las 1828 filas de entrenamiento; panel de consenso | `results/shap_todo_*.npy`, `consenso_todo.json` |
| 09 | `pipeline/09_consensus_panel.py` | Modelos reentrenados con el panel de consenso; prueba de DeLong | `results/tabla_comparacion_consenso_10var.csv` |
| 10 | `pipeline/10_tyg_comparison.py` | Comparación con el índice TyG (AUC, beneficio neto) | `results/tyg_*.csv` |
| 11 | `pipeline/11*_figures_*.py` | Figuras | `figures/` |

El paso 08 es el más lento (unas 5,5 h para TabPFN en un portátil con GPU; se puede reanudar). Los pasos 04-11
toman unas 8 h en total.

## Estructura del repositorio

- `pipeline/` – scripts de análisis numerados y `config.py`.
- `data/processed/` – conjuntos de datos del análisis derivados de los archivos públicos de NHANES (los de
  entrenamiento y prueba usados en el artículo).
- `models/` – modelos entrenados de random forest, SVM y XGBoost usados en el artículo.
- `results/` – tablas y arreglos SHAP producidos por el análisis.
- `notebooks/` – los cuadernos originales (Colab y locales) a partir de los cuales se armó el pipeline. Se
  conservan como referencia; la versión para ejecutar son los scripts de `pipeline/`.

## Notas de reproducibilidad

- **Réplica exacta.** `./run_all.sh` parte de los datos y modelos de este repositorio, que son los usados en el
  artículo (pasos 04-11).
- **Reentrenamiento (paso 03).** La búsqueda aleatoria de hiperparámetros no garantiza elegir los mismos valores
  en otro computador o versión de las librerías (varios candidatos tienen un AUC de validación cruzada casi
  idéntico). Al repetirla con el entrenamiento provisto se eligieron otros hiperparámetros para random forest y
  XGBoost, con un AUC de prueba algo distinto (random forest 0,846 y XGBoost 0,857, en lugar de 0,841 y 0,850; SVM
  sin cambios). Usa `--retrain` solo si quieres repetir la búsqueda; los modelos de `models/` son los del artículo.
- **Reconstrucción desde los datos originales.** `./run_all.sh --from-scratch` reproduce exactamente la cohorte
  (3483 participantes con insulina y glucosa; 2285 analizados; partición 1828 / 457; mismos valores observados).
  La imputación MICE (miceforest/LightGBM) no es idéntica bit a bit entre plataformas: cerca del 18 % de las
  filas de entrenamiento tienen al menos un valor imputado que difiere levemente. Junto con el paso 03, esto hace
  que los números no coincidan exactamente con los del artículo (deberían ser muy cercanos).
- **Presión arterial y pulso:** se usa la primera de las tres lecturas oscilométricas (`BPXOSY1`, `BPXODI1`,
  `BPXOPLS1`).
- **Altura:** se excluye porque está implícita en el IMC; el colesterol total y el peso se excluyen por VIF > 10.
- **TabPFN** (v2.5) necesita un token gratuito (https://ux.priorlabs.ai). Defínelo en la variable de entorno
  `TABPFN_TOKEN` o guárdalo en un archivo llamado `.tabpfn_token` en la raíz del repositorio (está en
  `.gitignore`; nunca lo subas). El modelo TabPFN no se guarda en disco (el archivo superaría los 6 GB); se
  reajusta con una semilla fija en los pasos que lo necesitan.
- **macOS.** XGBoost y PyTorch deben compartir un mismo OpenMP; `run_all.sh` define `DYLD_LIBRARY_PATH` para esto.
  Los pasos de SHAP, recalibración y consenso también se dividen en procesos separados (uno por modo).
- Las figuras usan etiquetas en español y coma decimal (están hechas para la tesis en español).

## Datos

Los datos de NHANES son públicos (CDC/NCHS). El paso 00 descarga los archivos originales; los conjuntos de
entrenamiento y prueba usados en el artículo están en `data/processed/`. Los códigos y definiciones de las
variables están en la Tabla Suplementaria S1 del artículo.

## Licencia

Ver `LICENSE`.
