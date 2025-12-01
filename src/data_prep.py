# src/data_prep.py
"""
Data preparation for Diabetes Prediction project.

Behavior:
- Prefer cleaned dataset at data/processed/improved_pima_diabetes.csv if present.
- Otherwise use raw CSV at data/raw/pima_diabetes.csv.
- Impute, scale, optional SMOTE, and save processed CSVs and scaler/imputer.
- Robust stratified train/test splitting that avoids sklearn error on tiny datasets.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Paths
IMPROVED_PATH = os.path.join("data", "processed", "improved_pima_diabetes.csv")
RAW_PATH = os.path.join("data", "raw", "pima_diabetes.csv")
PROC_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"

# Columns in Pima dataset that sometimes have zeros (treated as missing)
COLS_WITH_ZERO = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def load_data():
    """Load improved dataset if present, otherwise fallback to raw."""
    if os.path.exists(IMPROVED_PATH):
        print(f"Loading improved dataset: {IMPROVED_PATH}")
        df = pd.read_csv(IMPROVED_PATH)
    elif os.path.exists(RAW_PATH):
        print(f"Improved dataset not found. Loading raw dataset: {RAW_PATH}")
        df = pd.read_csv(RAW_PATH)
    else:
        raise FileNotFoundError(f"Neither {IMPROVED_PATH} nor {RAW_PATH} were found. Place dataset in data/raw or run preprocessing.")
    return df

def clean_and_engineer(df):
    """Replace impossible zeros with NaN for specific columns and impute medians.
       Returns cleaned df and fitted imputer.
    """
    df = df.copy()
    existing_cols_with_zero = [c for c in COLS_WITH_ZERO if c in df.columns]
    # Replace 0 with NaN where 0 is not a valid clinical value
    for c in existing_cols_with_zero:
        df[c] = df[c].replace(0, np.nan)

    # Fit median imputer on those columns and transform
    if existing_cols_with_zero:
        imputer = SimpleImputer(strategy="median")
        df[existing_cols_with_zero] = imputer.fit_transform(df[existing_cols_with_zero])
    else:
        imputer = None

    # Example feature engineering: keep existing engineered features if present
    # (If improved dataset already has Age_bin/BMI_category, leave them)
    # No aggressive feature creation here to keep pipeline simple.

    return df, imputer

def robust_train_test_split(X, y, test_fraction=0.2, random_state=42):
    """
    Perform a stratified split but ensure the test set has at least one sample per class.
    If the dataset is very small, adjust test_size to be an integer >= n_classes.
    Returns X_train, X_test, y_train, y_test.
    """
    n_samples = len(y)
    classes = np.unique(y)
    n_classes = len(classes)

    # compute integer test size (number of samples)
    test_size_int = max(int(np.ceil(test_fraction * n_samples)), n_classes)

    # avoid test_size >= n_samples
    if test_size_int >= n_samples:
        # make a reasonable split — put at least 1 sample in train and test
        if n_samples >= 2:
            test_size_int = max(1, n_samples // 2)
        else:
            raise ValueError("Dataset too small to split into train and test.")

    # sklearn accepts float or int for test_size; using int forces exact number of test samples
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_int, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def split_and_scale(df):
    """
    Split dataframe into train/test, scale numeric features, and return
    raw and scaled arrays/Series along with the fitted scaler.
    """
    # Ensure Outcome column exists
    if "Outcome" not in df.columns:
        raise KeyError("Dataset must contain 'Outcome' column (0/1).")

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # Keep column names for saving later
    feature_names = X.columns.tolist()

    # Robust stratified split
    X_train, X_test, y_train, y_test = robust_train_test_split(X, y, test_fraction=0.2, random_state=42)

    # Fit scaler on training set (numeric columns only)
    scaler = StandardScaler()
    # Ensure scaler is applied only to numeric columns; keep non-numeric columns intact
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_train_num = X_train[numeric_cols]
    X_test_num = X_test[numeric_cols]

    X_train_num_scaled = pd.DataFrame(scaler.fit_transform(X_train_num), columns=numeric_cols, index=X_train.index)
    X_test_num_scaled = pd.DataFrame(scaler.transform(X_test_num), columns=numeric_cols, index=X_test.index)

    # Reconstruct X_train_scaled and X_test_scaled combining numeric and any non-numeric columns
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = X_train_num_scaled
    X_test_scaled[numeric_cols] = X_test_num_scaled

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def resample_smote(X_scaled, y_train):
    """
    Apply SMOTE to the scaled numeric feature matrix.
    We will only use numeric columns for SMOTE.
    """
    numeric_cols = X_scaled.select_dtypes(include=[np.number]).columns.tolist()
    X_num = X_scaled[numeric_cols]
    sm = SMOTE(random_state=42)
    X_res_array, y_res = sm.fit_resample(X_num.values, y_train.values)
    # Convert back to DataFrame with same numeric column names
    X_res = pd.DataFrame(X_res_array, columns=numeric_cols)
    return X_res, y_res, numeric_cols

def save_processed(X_train_res, X_test_scaled, y_train_res, y_test, scaler, imputer, numeric_cols, original_feature_order):
    """Save processed datasets and fitted artifacts."""
    os.makedirs(PROC_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save processed CSVs with column names. If X_train_res has only numeric_cols (after SMOTE),
    # try to re-attach any missing non-numeric columns with NaNs.
    def ensure_full_df(X_df, full_cols):
        for c in full_cols:
            if c not in X_df.columns:
                X_df[c] = np.nan
        return X_df[full_cols]

    # X_train_res may only contain numeric_cols (post-SMOTE). Ensure same column order as original features.
    X_train_to_save = ensure_full_df(X_train_res if isinstance(X_train_res, pd.DataFrame) else pd.DataFrame(X_train_res, columns=numeric_cols),
                                     original_feature_order)
    X_test_to_save = ensure_full_df(X_test_scaled, original_feature_order)

    X_train_to_save.to_csv(os.path.join(PROC_DIR, "X_train.csv"), index=False)
    X_test_to_save.to_csv(os.path.join(PROC_DIR, "X_test.csv"), index=False)
    pd.Series(y_train_res).to_csv(os.path.join(PROC_DIR, "y_train.csv"), index=False, header=["Outcome"])
    pd.Series(y_test).to_csv(os.path.join(PROC_DIR, "y_test.csv"), index=False, header=["Outcome"])

    # Save scaler and imputer
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    if imputer is not None:
        joblib.dump(imputer, os.path.join(MODELS_DIR, "imputer.pkl"))

    print(f"Saved processed files to {PROC_DIR} and artifacts to {MODELS_DIR}.")

def main():
    # Create directories if missing
    os.makedirs(PROC_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load data
    df = load_data()
    print("Dataset loaded. Shape:", df.shape)
    print("Outcome distribution:\n", df["Outcome"].value_counts())

    # Clean / engineer
    df_clean, imputer = clean_and_engineer(df)
    print("Imputation complete. Any remaining NaNs per column:\n", df_clean.isna().sum())

    # Split & scale
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(df_clean)
    print("Split complete. Train/test shapes:", X_train.shape, X_test.shape)

    # Apply SMOTE on scaled numeric features
    try:
        X_res, y_res, numeric_cols = resample_smote(X_train_scaled, y_train)
        # After resampling we have only numeric columns; keep as DataFrame
        print("SMOTE applied. Resampled shapes:", X_res.shape, y_res.shape)
    except Exception as e:
        print("SMOTE failed or skipped:", e)
        # Fallback: use original train
        X_res = X_train_scaled.copy()
        y_res = y_train.copy()
        numeric_cols = X_res.select_dtypes(include=[np.number]).columns.tolist()

    # Save processed files (ensure feature order preserved)
    original_feature_order = X_train.columns.tolist()
    save_processed(X_res, X_test_scaled, y_res, y_test, scaler, imputer, numeric_cols, original_feature_order)

if __name__ == "__main__":
    main()
