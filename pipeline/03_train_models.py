"""Paso 03: entrena random forest, SVM y XGBoost
import json
import random
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config import DATA, MODELS, RESULTS, SEED, TARGET

random.seed(SEED)
np.random.seed(SEED)
N_FOLDS, N_ITER_BUSQUEDA = 5, 50
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
warnings.filterwarnings("ignore", category=FutureWarning)

train = pd.read_csv(DATA / "train_modelo.csv")
X_train, y_train = train.drop(columns=[TARGET]), train[TARGET].astype(int)

conteo = y_train.value_counts().sort_index()
print(pd.DataFrame({"n": conteo, "%": (conteo / len(y_train) * 100).round(1)}).to_string(), "\n")

# ponderación de clases en lugar de SMOTE
clases = np.unique(y_train)
class_weights = dict(zip(clases.tolist(),
                         compute_class_weight(class_weight="balanced", classes=clases, y=y_train).tolist()))
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()   # equivalente para XGBoost

print(f"class_weight (RF, SVM): {class_weights}")
print(f"scale_pos_weight (XGBoost): {scale_pos_weight:.4f}")

def buscar_hiperparametros(nombre, estimador, espacio, X, y):
    busqueda = RandomizedSearchCV(estimator=estimador, param_distributions=espacio, n_iter=N_ITER_BUSQUEDA,
                                  scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=-1, verbose=1)
    busqueda.fit(X, y)
    print(f"{nombre} – mejores hiperparámetros: {busqueda.best_params_}")
    print(f"{nombre} – AUC-ROC medio en CV: {busqueda.best_score_:.4f}")
    return busqueda

busqueda_rf = buscar_hiperparametros(
    "Random Forest",
    RandomForestClassifier(class_weight=class_weights, random_state=SEED, n_jobs=-1),
    {"n_estimators": [200, 300, 500, 800], "max_depth": [None, 5, 10, 15, 20],
     "min_samples_leaf": [1, 2, 5, 10], "min_samples_split": [2, 5, 10], "max_features": ["sqrt", "log2"]},
    X_train, y_train,
)
rf = busqueda_rf.best_estimator_

busqueda_svm = buscar_hiperparametros(
    "SVM",
    Pipeline([("scaler", StandardScaler()),
              ("svc", SVC(class_weight=class_weights, probability=True, random_state=SEED))]),
    {"svc__C": [0.01, 0.1, 1, 10, 100], "svc__kernel": ["rbf", "linear", "poly"],
     "svc__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1], "svc__degree": [2, 3, 4]},   # degree solo aplica a "poly"
    X_train, y_train,
)
svm = busqueda_svm.best_estimator_
print(f"Kernel seleccionado: {busqueda_svm.best_params_['svc__kernel']}")

busqueda_xgb = buscar_hiperparametros(
    "XGBoost",
    XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=SEED, n_jobs=-1),
    {"n_estimators": [200, 300, 500, 800], "max_depth": [3, 4, 5, 6, 8],
     "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2], "subsample": [0.6, 0.8, 1.0],
     "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5, 10], "gamma": [0, 0.1, 0.3, 0.5]},
    X_train, y_train,
)
xgb = busqueda_xgb.best_estimator_

# guardar los modelos
for nombre_archivo, objeto in [("modelo_random_forest.pkl", rf), ("modelo_svm.pkl", svm), ("modelo_xgboost.pkl", xgb)]:
    joblib.dump(objeto, MODELS / nombre_archivo)
hiperparametros = {
    "RF": busqueda_rf.best_params_,
    "SVM": {k.replace("svc__", ""): v for k, v in busqueda_svm.best_params_.items()},
    "XGB": {**busqueda_xgb.best_params_, "scale_pos_weight": float(scale_pos_weight)},
}
json.dump(hiperparametros, open(RESULTS / "hiperparametros.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("Modelos guardados en", MODELS)
