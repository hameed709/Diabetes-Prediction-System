# src/app_streamlit.py
"""
Diabetes Risk Screening App (Pipeline-Correct)

✔ Uses ONLY trained pipeline (models/best_model.pkl)
✔ No manual scaling / imputation
✔ Recall-first screening logic
✔ Risk bands instead of hard diagnosis
✔ Evaluation uses SAME threshold logic
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from feature_engineering import FeatureEngineer


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Diabetes Risk Screening", layout="wide")

MODELS_DIR = "models"
PROC_DIR = os.path.join("data", "processed")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
THRESH_PATH = os.path.join(MODELS_DIR, "threshold.json")

BASE_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

DEMO_CASES = {
    "Select demo case": None,
    "Healthy adult": {
        "Pregnancies": 0, "Glucose": 90, "BloodPressure": 72,
        "SkinThickness": 20, "Insulin": 60, "BMI": 22.5,
        "DiabetesPedigreeFunction": 0.2, "Age": 25
    },
    "Prediabetic profile": {
        "Pregnancies": 2, "Glucose": 125, "BloodPressure": 78,
        "SkinThickness": 25, "Insulin": 110, "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.6, "Age": 45
    },
    "High-risk diabetic": {
        "Pregnancies": 4, "Glucose": 180, "BloodPressure": 88,
        "SkinThickness": 30, "Insulin": 190, "BMI": 34.0,
        "DiabetesPedigreeFunction": 1.1, "Age": 55
    },
    "Severe uncontrolled": {
        "Pregnancies": 6, "Glucose": 220, "BloodPressure": 95,
        "SkinThickness": 35, "Insulin": 250, "BMI": 38.0,
        "DiabetesPedigreeFunction": 1.4, "Age": 62
    }
}

# ------------------------------------------------------------
# SAFE LOADERS
# ------------------------------------------------------------
@st.cache_resource
def load_pipeline(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_resource
def load_threshold(path, default=0.05):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return float(json.load(f)["threshold"])
    except Exception:
        return default

# ------------------------------------------------------------
# RISK & RECOMMENDATIONS
# ------------------------------------------------------------
def risk_band(prob):
    if prob < 0.10:
        return "Low Risk", "green"
    elif prob < 0.30:
        return "Moderate Risk", "orange"
    else:
        return "High Risk", "red"

DIET_RECIPES = {
    "low_risk": [
        "Vegetable oats porridge",
        "Idli with sambar (less oil)",
        "Grilled vegetable sandwich (whole wheat)",
        "Fruit bowl (apple, papaya, berries)",
        "Plain curd with flaxseeds"
    ],
    "moderate_risk": [
        "Ragi dosa with vegetable curry",
        "Vegetable khichdi (less rice, more dal)",
        "Brown rice with mixed vegetable sambar",
        "Boiled chickpea salad",
        "Sprouts stir-fry"
    ],
    "high_risk": [
        "Millet roti with vegetable sabzi",
        "Steamed vegetables with paneer/tofu",
        "Vegetable soup (no cream)",
        "Lentil curry with cauliflower rice",
        "Grilled fish or boiled eggs with vegetables"
    ],
    "very_high_glucose": [
        "Vegetable clear soup",
        "Steamed broccoli, carrot, beans",
        "Paneer/tofu bhurji (no oil)",
        "Green salad with lemon dressing"
    ],
    "weight_loss": [
        "Cucumber & tomato salad",
        "Vegetable stir-fry (minimal oil)",
        "Moong dal chilla",
        "Boiled eggs / tofu slices"
    ]
}


def get_recommendations(prob, row):
    """
    Personalized suggestions based on:
    - Model probability
    - Key clinical features
    """

    suggestions = []
    diet = []

    glucose = float(row["Glucose"])
    bmi = float(row["BMI"])
    age = int(row["Age"])
    bp = float(row["BloodPressure"])

    # --- Risk band (model-driven) ---
    if prob < 0.10:
        risk = "Low"
        risk_key = "low_risk"
        suggestions.append("Overall diabetes risk is currently low.")
        suggestions.append("Continue healthy lifestyle and routine screening.")
    elif prob < 0.30:
        risk = "Moderate"
        risk_key = "moderate_risk"
        suggestions.append("Moderate diabetes risk detected.")
        suggestions.append("Lifestyle modification is strongly recommended.")
    else:
        risk = "High"
        risk_key = "high_risk"
        suggestions.append("High diabetes risk detected.")
        suggestions.append("Consult a healthcare professional promptly.")


    # --- Glucose-based personalization ---
    if glucose >= 180:
        suggestions.append("Very high glucose levels detected — immediate medical review advised.")
        diet.extend(DIET_RECIPES["very_high_glucose"][:2])

    elif glucose >= 140:
        suggestions.append("Elevated glucose — risk of prediabetes/diabetes.")
        diet.append("Prefer low-glycemic foods (millets, oats, legumes).")
    elif glucose < 100:
        suggestions.append("Glucose level is within normal range.")
        diet.append("Maintain balanced meals with complex carbohydrates.")

    # --- BMI-based personalization ---
    if bmi >= 30:
        diet.extend(DIET_RECIPES["weight_loss"][:2])

    elif bmi >= 25:
        suggestions.append("Overweight — increase physical activity.")
        diet.append("Reduce refined carbs; prefer fiber-rich foods.")
    else:
        suggestions.append("Healthy BMI range maintained.")

    # --- Blood pressure ---
    if bp >= 140:
        suggestions.append("High blood pressure detected — monitor regularly.")
        diet.append("Reduce salt intake; avoid processed foods.")

    # --- Age factor ---
    if age >= 50:
        suggestions.append("Age-related risk — regular glucose testing every 6 months recommended.")

    # --- Universal advice ---
    suggestions.append("Aim for at least 150 minutes of moderate exercise per week.")
    suggestions.append("Ensure adequate sleep (7–8 hours) and stress management.")

    # --- Add meal suggestions based on risk ---
    if risk_key in DIET_RECIPES:
        diet.extend(DIET_RECIPES[risk_key][:3])

    return risk, suggestions, diet

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("🩺Diabetes Risk Screening")
st.write("High-sensitivity screening tool. **Not a diagnostic system.**")
st.write("---")

# ------------------------------------------------------------
# Initialize session state defaults (IMPORTANT)
# ------------------------------------------------------------
for k, v in {
    "Pregnancies": 0,
    "Glucose": 120.0,
    "BloodPressure": 70.0,
    "SkinThickness": 20.0,
    "Insulin": 80.0,
    "BMI": 30.0,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 30,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


pipeline = load_pipeline(MODEL_PATH)
threshold = load_threshold(THRESH_PATH)

if pipeline is None:
    st.error("Model not found. Train the model first.")
    st.stop()

st.info(f"Model operating in **high-sensitivity screening mode** (threshold = {threshold:.2f})")

st.subheader("Quick Demo Cases")

demo_choice = st.selectbox("Load a demo profile", list(DEMO_CASES.keys()))

if DEMO_CASES[demo_choice] is not None:
    for k, v in DEMO_CASES[demo_choice].items():
        st.session_state[k] = v




# ------------------------------------------------------------
# INPUT FORM
# ------------------------------------------------------------
with st.form("patient_form"):
    cols = st.columns(4)

    cols[0].number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        key="Pregnancies"
    )

    cols[1].number_input(
        "Age",
        min_value=1,
        max_value=120,
        key="Age"
    )

    cols[0].number_input(
        "Glucose (mg/dL)",
        min_value=0.0,
        max_value=300.0,
        key="Glucose"
    )

    cols[1].number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        key="BloodPressure"
    )

    cols[2].number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        key="SkinThickness"
    )

    cols[2].number_input(
        "Insulin",
        min_value=0.0,
        max_value=900.0,
        key="Insulin"
    )

    cols[3].number_input(
        "BMI",
        min_value=0.0,
        max_value=60.0,
        key="BMI"
    )

    cols[3].number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        key="DiabetesPedigreeFunction"
    )

    submitted = st.form_submit_button("Predict Risk")


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------
if submitted:
    X_input = pd.DataFrame(
    [{k: st.session_state[k] for k in BASE_FEATURES}],
    columns=BASE_FEATURES
    )

    prob = float(pipeline.predict_proba(X_input)[0][1])
    label, color = risk_band(prob)

    st.write("---")
    st.subheader("🔍 Prediction Result")

    st.metric("Predicted Diabetes Risk Probability", f"{prob:.3f}")
    st.markdown(
        f"### Risk Level: <span style='color:{color}'>{label}</span>",
        unsafe_allow_html=True,
    )

    # Probability bar
    fig, ax = plt.subplots(figsize=(6, 0.6))
    ax.barh([0], [prob], color=color)
    ax.set_xlim(0, 1)
    ax.axis("off")
    ax.text(prob / 2, 0, f"{prob*100:.1f}%", va="center", ha="center", fontweight="bold")
    st.pyplot(fig)

    risk_level, tips, diet = get_recommendations(prob, X_input.iloc[0])

    st.subheader("🧭 Recommendations")
    for t in tips:
        st.write(f"- {t}")

    st.subheader("🥗 Diet Guidance")
    for d in diet:
        st.write(f"- {d}")

# ------------------------------------------------------------
# OPTIONAL EVALUATION (TEST SET)
# ------------------------------------------------------------
x_test_path = os.path.join(PROC_DIR, "X_test.csv")
y_test_path = os.path.join(PROC_DIR, "y_test.csv")

if os.path.exists(x_test_path) and os.path.exists(y_test_path):
    st.write("---")
    st.subheader("📊 Model Evaluation (Holdout Test Set)")

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.legend()
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    st.pyplot(fig)

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)
