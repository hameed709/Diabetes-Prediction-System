"""
Create an improved, cleaned Pima diabetes CSV.

Place the original dataset at:
  diabetes-prediction/data/raw/pima_diabetes.csv

Run:
  python scripts/create_improved_dataset.py

Outputs:
  diabetes-prediction/data/processed/improved_pima_diabetes.csv
"""

import os
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_PATH = os.path.join(ROOT, "data", "raw", "pima_diabetes.csv")
OUT_DIR = os.path.join(ROOT, "data", "processed")
OUT_PATH = os.path.join(OUT_DIR, "improved_pima_diabetes.csv")

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Put the original dataset at: {RAW_PATH}")

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(RAW_PATH)

# Columns where zero is treated as missing
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for c in cols_with_zero:
    if c in df.columns:
        df[c] = df[c].replace(0, np.nan)

# Impute missing values with median (robust for skew)
for c in cols_with_zero:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())

# Feature engineering
# Age bins
if "Age" in df.columns:
    df["Age_bin"] = pd.cut(df["Age"], bins=[0,30,40,50,60,200],
                           labels=["<=30","31-40","41-50","51-60",">60"]).astype(str)

# BMI categories
if "BMI" in df.columns:
    df["BMI_category"] = pd.cut(df["BMI"], bins=[0,18.5,24.9,29.9,100],
                                labels=["Underweight","Normal","Overweight","Obese"]).astype(str)

# Glucose categories (simple grouping)
if "Glucose" in df.columns:
    df["Glucose_category"] = pd.cut(df["Glucose"], bins=[0,99,125,10000],
                                    labels=["Normal","Prediabetes","Diabetes_range"]).astype(str)

# Add normalized (z-score) numeric columns for convenience
num_cols = [c for c in ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"] if c in df.columns]
if len(num_cols) > 0:
    df_z = df[num_cols].apply(lambda x: (x - x.mean())/(x.std()+1e-9))
    df_z.columns = [c + "_z" for c in df_z.columns]
    df = pd.concat([df, df_z], axis=1)

# Reorder columns (prettier)
preferred = [c for c in ["Pregnancies","Glucose","Glucose_category","BloodPressure","SkinThickness","Insulin","BMI","BMI_category","DiabetesPedigreeFunction","Age","Age_bin","Outcome"] if c in df.columns]
others = [c for c in df.columns if c not in preferred]
df = df[preferred + others]

# Save
df.to_csv(OUT_PATH, index=False)
print(f"Created cleaned dataset: {OUT_PATH}")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
