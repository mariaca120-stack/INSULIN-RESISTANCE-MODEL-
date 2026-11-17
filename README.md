## Study summary

- **Data:** NHANES 2021–2023 (public). 2,285 adults without diagnosed diabetes
  (IR prevalence 41.7%), split into training (n = 1,828) and test (n = 457) sets
  before any preprocessing.
- **Models:** random forest, SVM, XGBoost and TabPFN (v2.5), compared by
  discrimination (AUC-ROC), calibration and decision-curve analysis. Probabilities
  were recalibrated with a logistic intercept/slope fitted on out-of-fold training
  predictions.
- **Interpretability:** SHAP values (probability scale) computed on the full
  training set. A consensus panel of 10 variables was defined as the intersection of
  the variables reaching 85% of cumulative SHAP importance in the three models.
- **Comparator:** the triglyceride-glucose (TyG) index.
- **Reporting:** TRIPOD+AI.

## Main results

On the test set, TabPFN had the highest AUC-ROC (0.861, 95% CI 0.827–0.894),
followed by XGBoost (0.850), random forest (0.841) and SVM (0.841), all higher than
the TyG index (0.762). The 10-variable panel reduced discrimination slightly
(TabPFN AUC 0.849).

## Data

NHANES data are public and are not included here. See the NHANES website for the
2021–2023 cycle files listed in the variable dictionary (Supplementary Table S1).

## Requirements

Python 3 with the packages and versions listed in `requirements.txt`.
