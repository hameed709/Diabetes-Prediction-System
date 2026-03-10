# src/train_models.py
"""
Training pipeline for Diabetes Risk Screening (AegisDiab™)

✔ Feature engineering INSIDE pipeline (Age_bin, BMI_category, Glucose_category)
✔ Recall-first threshold optimization
✔ Class imbalance handled
✔ Calibrated probabilities
✔ Single portable pipeline saved as best_model.pkl
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from feature_engineering import FeatureEngineer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    accuracy_score,
    precision_score,
    f1_score
)

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DATA_PATH = Path("data") / "processed" / "improved_pima_diabetes_clean.csv"
TARGET_COL = "Outcome"

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
MIN_RECALL = 0.70


# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------
def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")
    return df

def compute_best_threshold(pipe, X_val, y_val, min_recall=MIN_RECALL):
    probs = pipe.predict_proba(X_val)[:, 1]

    for t in np.linspace(0.05, 0.5, 50):
        preds = (probs >= t).astype(int)
        if recall_score(y_val, preds) >= min_recall:
            return float(t)

    return 0.15

# ------------------------------------------------------------
# MODEL PIPELINES
# ------------------------------------------------------------
def build_pipelines():
    base_steps = [
        ("features", FeatureEngineer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]

    pipes = {
        "logistic": Pipeline(base_steps + [
            ("clf", LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                class_weight={0: 1, 1: 3}
            ))
        ]),
        "decision_tree": Pipeline(base_steps + [
            ("clf", DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                class_weight={0: 1, 1: 3}
            ))
        ]),
        "random_forest": Pipeline(base_steps + [
            ("clf", RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                class_weight={0: 1, 1: 3}
            ))
        ]),
        "gradient_boost": Pipeline(base_steps + [
            ("clf", GradientBoostingClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE
            ))
        ]),
        "mlp": Pipeline(base_steps + [
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=RANDOM_STATE
            ))
        ])
    }

    if HAS_XGB:
        pipes["xgboost"] = Pipeline(base_steps + [
            ("clf", XGBClassifier(
                n_estimators=300,
                eval_metric="logloss",
                random_state=RANDOM_STATE
            ))
        ])

    return pipes

# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------
def main():
    df = load_data(DATA_PATH)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE,
        stratify=y_trainval, random_state=RANDOM_STATE
    )

    pipelines = build_pipelines()

    best_name = None
    best_recall = -1
    best_pipe = None
    best_thr = None

    for name, pipe in pipelines.items():
        print(f"\nTraining {name} ...")
        pipe.fit(X_tr, y_tr)

        probs = pipe.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, probs)

        thr = compute_best_threshold(pipe, X_val, y_val)
        preds = (probs >= thr).astype(int)
        recall = recall_score(y_val, preds)

        print(f"{name:14} AUC={auc:.4f}  Recall_val={recall:.4f}  thr={thr:.2f}")

        if recall > best_recall:
            best_recall = recall
            best_name = name
            best_pipe = pipe
            best_thr = thr

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------
    print(f"\nRefitting & calibrating best model ({best_name}) ...")

    X_cal = pd.concat([X_tr, X_val])
    y_cal = pd.concat([y_tr, y_val])

    base_pipe = best_pipe
    clf = base_pipe.named_steps["clf"]

    calibrated_clf = CalibratedClassifierCV(
        clf,
        method="sigmoid",
        cv=5
    )

    final_pipe = Pipeline(
        base_pipe.steps[:-1] + [("clf", calibrated_clf)]
    )

    final_pipe.fit(X_cal, y_cal)

    # --------------------------------------------------------
    # SAVE ARTIFACTS
    # --------------------------------------------------------
    joblib.dump(final_pipe, MODELS_DIR / "best_model.pkl")
    with open(MODELS_DIR / "threshold.json", "w") as f:
        json.dump({"threshold": best_thr, "model": best_name}, f)

    print("\nSaved models/best_model.pkl")
    print(f"Saved threshold.json (threshold={best_thr})")

    # --------------------------------------------------------
    # FINAL EVALUATION
    # --------------------------------------------------------
    probs_test = final_pipe.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= best_thr).astype(int)

    print("\n=== Final Holdout Evaluation ===")
    print("AUC:       ", roc_auc_score(y_test, probs_test))
    print("Accuracy:  ", accuracy_score(y_test, preds_test))
    print("Precision: ", precision_score(y_test, preds_test))
    print("Recall:    ", recall_score(y_test, preds_test))
    print("F1:        ", f1_score(y_test, preds_test))

if __name__ == "__main__":
    main()
