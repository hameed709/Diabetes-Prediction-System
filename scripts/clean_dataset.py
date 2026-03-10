import pandas as pd

INPUT_PATH = "data/processed/improved_pima_diabetes.csv"
OUTPUT_PATH = "data/processed/improved_pima_diabetes_clean.csv"

RAW_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

df = pd.read_csv(INPUT_PATH)

# Keep ONLY raw features + target
df_clean = df[RAW_FEATURES]

df_clean.to_csv(OUTPUT_PATH, index=False)

print("✅ Clean dataset written to:", OUTPUT_PATH)
print("Columns:", df_clean.columns.tolist())
