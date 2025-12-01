# src/train_models.py
"""
Robust training script for Diabetes project.

Improvements:
- Select numeric columns explicitly.
- Drop numeric columns that are all-NaN in the TRAIN set (can't impute them).
- Ensure the same numeric column set is used for test (add missing cols as 0).
- Impute remaining numeric NaNs with median and persist the imputer.
- Train multiple models and save them.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

PROC_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def load_processed():
    X_train = pd.read_csv(os.path.join(PROC_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROC_DIR, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(PROC_DIR, "y_test.csv"))

    # Normalize y series
    if "Outcome" in y_train.columns:
        y_train = y_train["Outcome"]
    else:
        y_train = y_train.iloc[:,0]

    if "Outcome" in y_test.columns:
        y_test = y_test["Outcome"]
    else:
        y_test = y_test.iloc[:,0]

    return X_train, X_test, y_train, y_test

def prepare_numeric(X_train_df, X_test_df):
    # Choose numeric columns explicitly
    numeric_cols = [c for c in X_train_df.columns if pd.api.types.is_numeric_dtype(X_train_df[c])]
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in X_train.csv. Check processed data.")

    Xtr_num = X_train_df[numeric_cols].copy()
    Xte_num = X_test_df[numeric_cols].copy()  # if test lacks a numeric col, this will KeyError; we'll handle below

    # Identify numeric cols that are entirely NaN in training set and drop them
    cols_all_nan = [c for c in numeric_cols if Xtr_num[c].isna().all()]
    if cols_all_nan:
        print("Dropping numeric columns that are all-NaN in training data:", cols_all_nan)
        Xtr_num = Xtr_num.drop(columns=cols_all_nan)
        # Also drop from numeric_cols list
        numeric_cols = [c for c in numeric_cols if c not in cols_all_nan]

    # Ensure test has the same numeric columns. If test lacks columns, create them filled with NaN.
    for c in numeric_cols:
        if c not in Xte_num.columns:
            Xte_num[c] = np.nan

    # Reorder test columns to match train
    Xte_num = Xte_num[numeric_cols]

    # Impute numeric NaNs with median
    imputer = SimpleImputer(strategy="median")
    Xtr_imp_array = imputer.fit_transform(Xtr_num)
    Xte_imp_array = imputer.transform(Xte_num)

    # Convert back to DataFrame
    Xtr_imp = pd.DataFrame(Xtr_imp_array, columns=numeric_cols, index=Xtr_num.index)
    Xte_imp = pd.DataFrame(Xte_imp_array, columns=numeric_cols, index=Xte_num.index)

    # Save imputer
    joblib.dump(imputer, os.path.join(MODELS_DIR, "train_imputer.pkl"))

    return Xtr_imp.values, Xte_imp.values, numeric_cols

def train_and_save():
    X_train_df, X_test_df, y_train, y_test = load_processed()
    print("Loaded processed datasets.")
    print("X_train shape:", X_train_df.shape, "X_test shape:", X_test_df.shape)
    print("y_train distribution:\n", y_train.value_counts())

    X_train, X_test, numeric_cols = prepare_numeric(X_train_df, X_test_df)
    print("Training with numeric columns:", numeric_cols)
    # Safety: ensure no NaNs remain (after imputation)
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        print("Warning: NaNs still present after imputation. Converting NaNs to 0 as last resort.")
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)

    results = []

    # Logistic Regression
    print("Training LogisticRegression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    joblib.dump(lr, os.path.join(MODELS_DIR, "logistic.pkl"))
    results.append(("Logistic", accuracy_score(y_test, lr.predict(X_test))))

    # Random Forest
    print("Training RandomForest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))
    results.append(("RandomForest", accuracy_score(y_test, rf.predict(X_test))))

    # XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, os.path.join(MODELS_DIR, "xgb.pkl"))
    results.append(("XGBoost", accuracy_score(y_test, xgb.predict(X_test))))

    # SVM
    print("Training SVM...")
    svc = SVC(probability=True)
    svc.fit(X_train, y_train)
    joblib.dump(svc, os.path.join(MODELS_DIR, "svm.pkl"))
    results.append(("SVM", accuracy_score(y_test, svc.predict(X_test))))

    # MLP
    print("Training MLP...")
    mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    joblib.dump(mlp, os.path.join(MODELS_DIR, "mlp.pkl"))
    results.append(("MLP", accuracy_score(y_test, mlp.predict(X_test))))

    # Save summary
    df_summary = pd.DataFrame(results, columns=["model", "accuracy"])
    df_summary.to_csv(os.path.join(MODELS_DIR, "models_summary.csv"), index=False)
    print("Training complete. Models saved in models/.")

if __name__ == "__main__":
    train_and_save()
