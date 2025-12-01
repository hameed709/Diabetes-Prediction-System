# src/evaluate.py
"""
Robust evaluation script.

Behavior:
- Loads data/processed/X_test.csv and y_test.csv
- Uses numeric columns only (drops string/categorical columns)
- Imputes numeric NaNs using models/train_imputer.pkl if available, otherwise median imputation
- Evaluates all saved models in models/ and saves metrics and plots to reports/figures/
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer

PROC_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
REPORT_DIR = os.path.join("reports", "figures")
os.makedirs(REPORT_DIR, exist_ok=True)

def load_test():
    X_test = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PROC_DIR, "y_test.csv"))
    # normalize y
    if "Outcome" in y_test.columns:
        y_test = y_test["Outcome"]
    else:
        y_test = y_test.iloc[:,0]
    return X_test, y_test

def prepare_test_numeric(X_test_df):
    """
    Return numeric-only DataFrame (columns kept in original order),
    and the column list used.
    """
    numeric_cols = X_test_df.select_dtypes(include=[np.number]).columns.tolist()
    X_test_num = X_test_df[numeric_cols].copy()
    return X_test_num, numeric_cols

def impute_numeric(X_df, imputer_path=None):
    """
    Impute numeric DataFrame:
    - If imputer_path exists, load and use it (expects sklearn SimpleImputer-like object)
    - Otherwise, do median imputation and return the imputer used.
    Returns: (X_imputed_df, imputer_obj)
    """
    numeric_cols = X_df.columns.tolist()
    if imputer_path and os.path.exists(imputer_path):
        try:
            imputer = joblib.load(imputer_path)
            arr = imputer.transform(X_df)
            X_imp = pd.DataFrame(arr, columns=numeric_cols, index=X_df.index)
            return X_imp, imputer
        except Exception as e:
            print("Failed loading imputer from", imputer_path, " — falling back to median imputer. Error:", e)

    # Fallback median imputer
    imputer = SimpleImputer(strategy="median")
    arr = imputer.fit_transform(X_df)
    X_imp = pd.DataFrame(arr, columns=numeric_cols, index=X_df.index)
    return X_imp, imputer

def evaluate_model(model_path, X_test_arr, y_test, name):
    """
    Evaluate a single model and save plots.
    X_test_arr: numpy array (n_samples, n_features)
    """
    model = joblib.load(model_path)

    # Predict
    try:
        y_pred = model.predict(X_test_arr)
    except Exception as e:
        raise RuntimeError(f"Failed to predict with model {name}: {e}")

    # Probabilities / scores
    try:
        y_prob = model.predict_proba(X_test_arr)[:,1]
    except Exception:
        try:
            score = model.decision_function(X_test_arr)
            # scale to [0,1]
            y_prob = (score - score.min())/(score.max()-score.min()+1e-8)
        except Exception:
            # fallback to predicted labels as 0/1
            y_prob = (y_pred).astype(float)

    # Metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc_auc = float("nan")

    metrics = {
        "model": name,
        "roc_auc": roc_auc,
        "precision": report.get("1", {}).get("precision", np.nan),
        "recall": report.get("1", {}).get("recall", np.nan),
        "f1": report.get("1", {}).get("f1-score", np.nan),
        "accuracy": float((y_pred == y_test).mean())
    }

    # Plot ROC
    try:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
        plt.plot([0,1],[0,1], linestyle="--", color="grey")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC - {name}")
        plt.legend()
        plt.savefig(os.path.join(REPORT_DIR, f"roc_{name}.png"))
        plt.close()
    except Exception as e:
        print("ROC plot failed for", name, e)

    # Confusion matrix heatmap
    try:
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.savefig(os.path.join(REPORT_DIR, f"cm_{name}.png"))
        plt.close()
    except Exception as e:
        print("Confusion matrix plot failed for", name, e)

    return metrics

def main():
    X_test_df, y_test = load_test()
    print("Loaded X_test shape:", X_test_df.shape, "y_test shape:", y_test.shape)

    # Keep numeric-only columns
    X_test_num_df, numeric_cols = prepare_test_numeric(X_test_df)
    print("Numeric test columns used for evaluation:", numeric_cols)

    # Impute numeric NaNs using training imputer if available
    imputer_path = os.path.join(MODELS_DIR, "train_imputer.pkl")
    X_test_imp_df, imputer_obj = impute_numeric(X_test_num_df, imputer_path=imputer_path)

    X_test_arr = X_test_imp_df.values
    # y_test as numpy
    y_test_arr = y_test.values

    # Find model files in models/
    candidates = {
        "logistic": os.path.join(MODELS_DIR, "logistic.pkl"),
        "random_forest": os.path.join(MODELS_DIR, "random_forest.pkl"),
        "xgb": os.path.join(MODELS_DIR, "xgb.pkl"),
        "svm": os.path.join(MODELS_DIR, "svm.pkl"),
        "mlp": os.path.join(MODELS_DIR, "mlp.pkl"),
    }

    metrics = []
    for name, path in candidates.items():
        if not os.path.exists(path):
            print("Model not found, skipping:", path)
            continue
        print("Evaluating", name)
        try:
            m = evaluate_model(path, X_test_arr, y_test_arr, name)
            metrics.append(m)
        except Exception as e:
            print("Evaluation failed for", name, ":", e)

    if metrics:
        dfm = pd.DataFrame(metrics)
        dfm.to_csv(os.path.join(REPORT_DIR, "model_metrics.csv"), index=False)
        print("Saved model metrics to", os.path.join(REPORT_DIR, "model_metrics.csv"))
    else:
        print("No metrics collected — no models evaluated successfully.")

if __name__ == "__main__":
    main()
