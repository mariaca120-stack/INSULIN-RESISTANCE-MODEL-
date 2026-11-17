#!/usr/bin/env bash
# Runs the whole analysis, step by step.
#   ./run_all.sh                 steps 04-11, using the datasets and models already in the repository (reproduces the paper)
#   ./run_all.sh --from-scratch  also steps 00-02: downloads NHANES from the CDC and rebuilds the datasets
#   ./run_all.sh --retrain       also step 03: repeats the hyperparameter search and overwrites models/
#                                (the search can select slightly different, equivalent hyperparameters)
# Expected total time: ~8 h (step 08, TabPFN SHAP on the full training set, takes ~5.5 h on a laptop GPU).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

# macOS: XGBoost and PyTorch must share the same OpenMP runtime (otherwise the process crashes or hangs)
if [[ "$(uname)" == "Darwin" ]]; then
  TORCH_LIB="$($PY -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
  export DYLD_LIBRARY_PATH="$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

paso() { echo; echo "=== $* ($(date '+%H:%M:%S')) ==="; $PY "$@"; }

FROM_SCRATCH=0; RETRAIN=0
for arg in "$@"; do
  case "$arg" in
    --from-scratch) FROM_SCRATCH=1; RETRAIN=1 ;;   # new datasets require new models
    --retrain) RETRAIN=1 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
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

# Recalibration (each mode in its own process: XGBoost and PyTorch must not share a process on macOS)
paso pipeline/06_recalibration.py skl
paso pipeline/06_recalibration.py tabpfn
paso pipeline/06_recalibration.py analisis
paso pipeline/07_recalibrated_metrics_ci.py

# SHAP on the full training set and consensus panel
for modo in rf43 xgb43 tabpfn43 consenso_arboles tabpfn_consenso; do
  paso pipeline/08_shap_full_training_set.py "$modo"
done
for modo in skl tabpfn delong unir; do
  paso pipeline/09_consensus_panel.py "$modo"
done

paso pipeline/10_tyg_comparison.py

# Figures
paso pipeline/11a_figures_shap_tables.py
paso pipeline/11b_figures_shap_summary.py
paso pipeline/11c_figures_supplementary.py
paso pipeline/11e_figures_main.py
paso pipeline/11f_figures_confusion_calibration.py
paso pipeline/11g_figures_normality.py
echo; echo "Done. Tables in results/, figures in figures/."
