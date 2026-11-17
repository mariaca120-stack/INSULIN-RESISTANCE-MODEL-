#!/usr/bin/env bash
# Ejecuta todo el análisis, paso a paso.
#   ./run_all.sh                 pasos 04-11, con los datos y modelos que ya están en el repositorio (reproduce el artículo)
#   ./run_all.sh --from-scratch  además los pasos 00-02: descarga NHANES de los CDC y reconstruye los datos
#   ./run_all.sh --retrain       además el paso 03: repite la búsqueda de hiperparámetros y sobrescribe models/
#                                (la búsqueda puede elegir hiperparámetros ligeramente distintos pero equivalentes)
# Tiempo total aproximado: 8 h (el paso 08, SHAP de TabPFN con todo el entrenamiento, toma ~5,5 h en un portátil con GPU).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

# macOS: XGBoost y PyTorch deben compartir el mismo OpenMP (si no, el proceso falla o se cuelga)
if [[ "$(uname)" == "Darwin" ]]; then
  TORCH_LIB="$($PY -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
  export DYLD_LIBRARY_PATH="$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

paso() { echo; echo "=== $* ($(date '+%H:%M:%S')) ==="; $PY "$@"; }

FROM_SCRATCH=0; RETRAIN=0
for arg in "$@"; do
  case "$arg" in
    --from-scratch) FROM_SCRATCH=1; RETRAIN=1 ;;   # datos nuevos requieren modelos nuevos
    --retrain) RETRAIN=1 ;;
    *) echo "Opción desconocida: $arg"; exit 1 ;;
  esac
done

if [[ $FROM_SCRATCH == 1 ]]; then
  paso pipeline/00_download_nhanes.py
  paso pipeline/01_build_dataset.py
  paso pipeline/02_prepare_data.py
fi
if [[ $RETRAIN == 1 ]]; then
  paso pipeline/03_train_models.py
fi

paso pipeline/04_evaluate_models.py
paso pipeline/05_test_metrics_ci.py

# Recalibración (cada modo en su propio proceso: en macOS XGBoost y PyTorch no deben compartir proceso)
paso pipeline/06_recalibration.py skl
paso pipeline/06_recalibration.py tabpfn
paso pipeline/06_recalibration.py analisis
paso pipeline/07_recalibrated_metrics_ci.py

# SHAP con todo el entrenamiento y panel de consenso
for modo in rf43 xgb43 tabpfn43 consenso_arboles tabpfn_consenso; do
  paso pipeline/08_shap_full_training_set.py "$modo"
done
for modo in skl tabpfn delong unir; do
  paso pipeline/09_consensus_panel.py "$modo"
done

paso pipeline/10_tyg_comparison.py

# Figuras
paso pipeline/11a_figures_shap_tables.py
paso pipeline/11b_figures_shap_summary.py
paso pipeline/11c_figures_supplementary.py
paso pipeline/11e_figures_main.py
paso pipeline/11f_figures_confusion_calibration.py
paso pipeline/11g_figures_normality.py
echo; echo "Listo. Tablas en results/, figuras en figures/."
