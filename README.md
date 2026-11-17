# Insulin resistance classification with machine learning (NHANES 2021–2023)

This repository contains the code for a cross-sectional study that develops and compares machine learning
models to classify insulin resistance (IR, defined as HOMA-IR > 2.5) in US adults, without using fasting
insulin or glucose as predictors.

## Study summary

- **Data:** NHANES 2021–2023 (public). 2,285 adults without diagnosed diabetes (IR prevalence 41.7%), split into
  training (n = 1,828) and test (n = 457) sets before any preprocessing.
- **Models:** random forest, SVM, XGBoost and TabPFN (v2.5), compared by discrimination (AUC-ROC), calibration and
  decision-curve analysis. Probabilities were recalibrated with a logistic intercept/slope fitted on out-of-fold
  training predictions.
- **Interpretability:** SHAP values (probability scale) computed on the full training set. A consensus panel of
  10 variables was defined as the intersection of the variables reaching 85% of cumulative SHAP importance in the
  three models.
- **Comparator:** the triglyceride-glucose (TyG) index.
- **Reporting:** TRIPOD+AI.

## Main results

On the test set, TabPFN had the highest AUC-ROC (0.861, 95% CI 0.827–0.894), followed by XGBoost (0.850), random
forest (0.841) and SVM (0.841), all higher than the TyG index (0.762). The 10-variable panel reduced discrimination
slightly (TabPFN AUC 0.849).

## How to run

Everything runs from this repository: no Google Drive or manual downloads are needed.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

./run_all.sh                  # reproduces the paper from the datasets and models in this repository (steps 04-11)
./run_all.sh --retrain        # also repeats the hyperparameter search (step 03) and overwrites models/
./run_all.sh --from-scratch   # also downloads NHANES from the CDC and rebuilds the datasets (steps 00-02, then 03)
```

Each step can also be run on its own, e.g. `python pipeline/05_test_metrics_ci.py`. All paths are defined in
`pipeline/config.py` (relative to the repository root).

| Step | Script | What it does | Main output |
|---|---|---|---|
| 00 | `pipeline/00_download_nhanes.py` | Downloads the NHANES 2021–2023 `.xpt` modules from the CDC | `data/raw/` |
| 01 | `pipeline/01_build_dataset.py` | Merges the modules by `SEQN` and renames variables | `data/processed/nhanes_variables.csv` |
| 02 | `pipeline/02_prepare_data.py` | Cohort, cleaning, HOMA-IR outcome, 80/20 split, MICE imputation, Table 1, VIF | `data/processed/train_modelo.csv`, `test_modelo.csv` |
| 03 | `pipeline/03_train_models.py` | *(optional)* Random forest, SVM and XGBoost: randomized search with 5-fold CV | `models/*.pkl` |
| 04 | `pipeline/04_evaluate_models.py` | Fits TabPFN; test metrics, ROC, cross-validation | `results/probabilidades_test.csv`, `tabla_metricas_*.csv` |
| 05 | `pipeline/05_test_metrics_ci.py` | 95% CIs (DeLong, bootstrap), DeLong comparison, net benefit | `results/metricas_test_ic95.csv` |
| 06 | `pipeline/06_recalibration.py` | Logistic recalibration with out-of-fold training predictions | `results/probabilidades_test_recal.csv` |
| 07 | `pipeline/07_recalibrated_metrics_ci.py` | Recalibrated metrics with bootstrap CIs | `results/metricas_test_ic95_recal.csv` |
| 08 | `pipeline/08_shap_full_training_set.py` | SHAP (probability scale) on all 1,828 training rows; consensus panel | `results/shap_todo_*.npy`, `consenso_todo.json` |
| 09 | `pipeline/09_consensus_panel.py` | Models retrained with the consensus panel; DeLong test | `results/tabla_comparacion_consenso_10var.csv` |
| 10 | `pipeline/10_tyg_comparison.py` | Comparison with the TyG index (AUC, net benefit) | `results/tyg_*.csv` |
| 11 | `pipeline/11*_figures_*.py` | Figures | `figures/` |

Step 08 is the slowest (about 5.5 h for TabPFN on a laptop GPU, resumable). Steps 04-11 take about 8 h in total.

## Repository structure

- `pipeline/` – numbered analysis scripts and `config.py`.
- `data/processed/` – analysis datasets derived from the public NHANES files (training/test sets used in the paper).
- `models/` – trained random forest, SVM and XGBoost models used in the paper.
- `results/` – tables and SHAP arrays produced by the analysis.
- `notebooks/` – the original exploratory notebooks (Colab and local) from which the pipeline was derived. They
  are kept for reference; the scripts in `pipeline/` are the version to run.

## Reproducibility notes

- **Exact replication.** `./run_all.sh` starts from the datasets and models provided in this repository, which are
  the ones used in the paper (steps 04-11).
- **Retraining (step 03).** The randomized hyperparameter search is not guaranteed to select the same
  hyperparameters on another machine or package build (several candidates have almost identical cross-validated
  AUC). Repeating it on the provided training set selected different random forest and XGBoost settings, with
  slightly different test AUC (random forest 0.846, XGBoost 0.857 instead of 0.841 and 0.850; SVM unchanged).
  Use `--retrain` only if you want to repeat the search; the models in `models/` are the ones behind the paper.
- **Rebuilding from the raw data.** `./run_all.sh --from-scratch` reproduces the cohort exactly (3,483 participants
  with insulin and glucose; 2,285 analysed; 1,828 / 457 split; same observed values). MICE imputation
  (miceforest/LightGBM) is not bit-for-bit reproducible across platforms: about 18% of training rows have at least
  one imputed value that differs slightly. Together with step 03, this means the numbers will not match the paper
  exactly (they should be very close).
- **Blood pressure and pulse** use the first of the three oscillometric readings (`BPXOSY1`, `BPXODI1`, `BPXOPLS1`).
- **Height** is excluded because it is implicit in BMI; total cholesterol and weight are excluded by VIF > 10.
- **TabPFN** (v2.5) needs a free token (https://ux.priorlabs.ai). Set the environment variable `TABPFN_TOKEN`, or put
  it in a file named `.tabpfn_token` at the repository root (it is git-ignored; never commit it). The TabPFN model
  is not saved to disk (the file would exceed 6 GB); it is refitted with a fixed seed in the steps that need it.
- **macOS.** XGBoost and PyTorch must share one OpenMP runtime; `run_all.sh` sets `DYLD_LIBRARY_PATH` for this.
  The SHAP, recalibration and consensus steps are also split into separate processes (one per mode).
- The figures use Spanish labels and decimal commas (they are made for the Spanish-language thesis).

## Data

NHANES data are public (CDC/NCHS). The raw files are downloaded by step 00; the processed training and test sets
used in the paper are in `data/processed/`. Variable codes and definitions are in Supplementary Table S1 of the
paper.

## License

See `LICENSE`.
