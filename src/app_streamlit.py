# src/app_streamlit.py
"""
AegisDiab™ — Upgraded Streamlit app
Features:
- Auto-selects best model (prefers models/best_model.pkl)
- Uses train_imputer.pkl and scaler.pkl for exact preprocessing
- Loads threshold from models/threshold.json (default 0.4)
- SHAP explainability (if shap installed)
- Personalized suggestions & diet plan
Run:
    streamlit run src/app_streamlit.py
"""

import os, json, joblib
import pandas as pd
import numpy as np
import streamlit as st

# UI
st.set_page_config(page_title="AegisDiab — Risk & Prescriptions", layout="centered")
st.markdown("<h1 style='text-align:center'>AegisDiab™</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:gray'>Diabetes Risk Assessment · Explainability · Personalized Diet & Suggestions</h4>", unsafe_allow_html=True)
st.write("---")

# Paths & defaults
MODELS_DIR = "models"
PROC_DIR = os.path.join("data", "processed")
METRICS_CSV = os.path.join("reports", "figures", "model_metrics.csv")
SCALE_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
IMPUTER_PATH = os.path.join(MODELS_DIR, "train_imputer.pkl")
THRESH_PATH = os.path.join(MODELS_DIR, "threshold.json")
DEFAULT_THRESHOLD = 0.40

# Helper: load numeric feature order used during training (from X_train.csv)
def get_training_numeric_order():
    xtrain_path = os.path.join(PROC_DIR, "X_train.csv")
    if os.path.exists(xtrain_path):
        df = pd.read_csv(xtrain_path)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return num_cols
    # fallback to expected Pima columns order
    fallback = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
    return fallback

training_numeric_order = get_training_numeric_order()

# Helper: list available models
def list_available_models():
    # Exclude non-model files
    excluded = {"scaler.pkl", "imputer.pkl", "train_imputer.pkl", "best_model.pkl"}
    models = []
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.endswith(".pkl") and f not in excluded:
                models.append(f)
    return sorted(models)

# Helper: determine default model (best)
def get_default_model_name():
    # 1) explicit best model file
    if os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl")):
        return "best_model.pkl" # We might want to map this back to the actual filename if possible, but for now let's just use the logic to find the best name from metrics
    
    # 2) choose via metrics CSV (roc_auc)
    if os.path.exists(METRICS_CSV):
        try:
            dfm = pd.read_csv(METRICS_CSV)
            if "roc_auc" in dfm.columns and dfm["roc_auc"].notna().any():
                best = dfm.sort_values("roc_auc", ascending=False).iloc[0]["model"]
            else:
                best = dfm.sort_values("accuracy", ascending=False).iloc[0]["model"]
            
            map_file = {
                "logistic": "logistic.pkl",
                "random_forest": "random_forest.pkl",
                "xgb": "xgb.pkl",
                "svm": "svm.pkl",
                "mlp": "mlp.pkl"
            }.get(best, None)
            
            if map_file and os.path.exists(os.path.join(MODELS_DIR, map_file)):
                return map_file
        except Exception:
            pass
            
    # 3) fallback
    for fallback in ["xgb.pkl", "random_forest.pkl", "logistic.pkl"]:
        if os.path.exists(os.path.join(MODELS_DIR, fallback)):
            return fallback
    return None

# Load artifacts
available_models = list_available_models()
default_model = get_default_model_name()

scaler = None
imputer = None
threshold = DEFAULT_THRESHOLD

if os.path.exists(SCALE_PATH):
    try:
        scaler = joblib.load(SCALE_PATH)
    except Exception as e:
        st.warning(f"Failed to load scaler: {e}")

if os.path.exists(IMPUTER_PATH):
    try:
        imputer = joblib.load(IMPUTER_PATH)
    except Exception as e:
        st.warning(f"Failed to load imputer: {e}")

if os.path.exists(THRESH_PATH):
    try:
        with open(THRESH_PATH, "r") as f:
            threshold = json.load(f).get("threshold", DEFAULT_THRESHOLD)
    except Exception:
        threshold = DEFAULT_THRESHOLD

# Sidebar
with st.sidebar:
    st.header("Project status")
    
    # Model Selector
    if available_models:
        # Try to set index to default model
        try:
            default_idx = available_models.index(default_model) if default_model in available_models else 0
        except:
            default_idx = 0
            
        selected_model_file = st.selectbox("Select Model", available_models, index=default_idx)
        
        # Load the selected model
        model_path = os.path.join(MODELS_DIR, selected_model_file)
        try:
            model = joblib.load(model_path)
            model_file = selected_model_file
            st.success(f"Loaded: {model_file}")
        except Exception as e:
            st.error(f"Failed to load {selected_model_file}: {e}")
            model = None
            model_file = None
    else:
        st.error("No models found in models/ directory.")
        model = None
        model_file = None
    st.write(f"Using threshold = {threshold:.2f}")
    st.write("---")
    st.markdown("**Preset examples**")
    if st.button("Load sample: Low risk"):
        st.session_state.update({"Pregnancies":0,"Glucose":95.0,"BloodPressure":70.0,"SkinThickness":18.0,"Insulin":80.0,"BMI":24.0,"DPF":0.2,"Age":29})
    if st.button("Load sample: High risk"):
        st.session_state.update({"Pregnancies":3,"Glucose":180.0,"BloodPressure":88.0,"SkinThickness":32.0,"Insulin":200.0,"BMI":34.0,"DPF":1.2,"Age":55})
    st.write("---")
    st.markdown("Notes:\n- Ensure models/scaler.pkl & models/train_imputer.pkl exist.\n- To retrain or change threshold, run pipeline scripts.")
    # --- Demo cases (paste inside the sidebar block) ---
st.markdown("### Quick demo cases")
demo_cases = {
    "Select a demo case": None,
    "Low risk (healthy adult)": {
        "Pregnancies":0,"Glucose":90.0,"BloodPressure":72.0,"SkinThickness":22.0,
        "Insulin":60.0,"BMI":23.0,"DiabetesPedigreeFunction":0.2,"Age":25
    },
    "Moderate risk (prediabetic)": {
        "Pregnancies":2,"Glucose":120.0,"BloodPressure":75.0,"SkinThickness":20.0,
        "Insulin":90.0,"BMI":29.0,"DiabetesPedigreeFunction":0.5,"Age":45
    },
    "High risk (diabetic)": {
        "Pregnancies":4,"Glucose":170.0,"BloodPressure":85.0,"SkinThickness":28.0,
        "Insulin":190.0,"BMI":33.5,"DiabetesPedigreeFunction":1.0,"Age":50
    },
    "Severe case (uncontrolled)": {
        "Pregnancies":6,"Glucose":220.0,"BloodPressure":90.0,"SkinThickness":35.0,
        "Insulin":250.0,"BMI":38.0,"DiabetesPedigreeFunction":1.4,"Age":60
    },
    "Obese but normal glucose (metabolic risk)": {
        "Pregnancies":3,"Glucose":95.0,"BloodPressure":80.0,"SkinThickness":25.0,
        "Insulin":100.0,"BMI":34.0,"DiabetesPedigreeFunction":0.7,"Age":42
    }
}

chosen_demo = st.selectbox("Load demo case", list(demo_cases.keys()))
if chosen_demo and demo_cases.get(chosen_demo):
    # populate session_state so the form uses these values
    vals = demo_cases[chosen_demo]
    st.session_state.update({
        "Pregnancies": vals["Pregnancies"],
        "Glucose": vals["Glucose"],
        "BloodPressure": vals["BloodPressure"],
        "SkinThickness": vals["SkinThickness"],
        "Insulin": vals["Insulin"],
        "BMI": vals["BMI"],
        "DPF": vals["DiabetesPedigreeFunction"],
        "Age": vals["Age"]
    })


# Input form
# ---- Input form (replace existing form block) ----
st.subheader("Enter patient details")

# Ensure session_state has defaults so widgets initialize correctly
defaults = {
    "Pregnancies": 0,
    "Glucose": 120.0,
    "BloodPressure": 70.0,
    "SkinThickness": 20.0,
    "Insulin": 79.0,
    "BMI": 32.0,
    "DPF": 0.5,
    "Age": 33
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

with st.form("patient_form"):
    c1, c2, c3, c4 = st.columns(4)
    # Note: keys MUST match those used by the demo selector earlier
    c1.number_input("Pregnancies", min_value=0, max_value=20, key="Pregnancies", value=st.session_state["Pregnancies"])
    c2.number_input("Age", min_value=1, max_value=120, key="Age", value=st.session_state["Age"])
    c1.number_input("Glucose (mg/dL)", min_value=0.0, key="Glucose", value=st.session_state["Glucose"], format="%.1f")
    c2.number_input("Blood Pressure (mm Hg)", min_value=0.0, key="BloodPressure", value=st.session_state["BloodPressure"], format="%.1f")
    c3.number_input("Skin Thickness (mm)", min_value=0.0, key="SkinThickness", value=st.session_state["SkinThickness"], format="%.1f")
    c3.number_input("Insulin (mu U/ml)", min_value=0.0, key="Insulin", value=st.session_state["Insulin"], format="%.1f")
    c4.number_input("BMI", min_value=0.0, key="BMI", value=st.session_state["BMI"], format="%.1f")
    c4.number_input("Diabetes Pedigree Function", min_value=0.0, key="DPF", value=st.session_state["DPF"], format="%.3f")

    submitted = st.form_submit_button("Predict & Prescribe")

# After form submit (or if values changed via demo), read values from session_state
pregnancies = st.session_state["Pregnancies"]
glucose = st.session_state["Glucose"]
bp = st.session_state["BloodPressure"]
skin = st.session_state["SkinThickness"]
insulin = st.session_state["Insulin"]
bmi = st.session_state["BMI"]
dpf = st.session_state["DPF"]
age = st.session_state["Age"]

# Build input DataFrame from these authoritative values
input_raw = pd.DataFrame([[
    pregnancies, glucose, bp, skin, insulin, bmi, dpf, age
]], columns=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"])


def prepare_input_for_model(input_df):
    """
    Build an input vector that exactly matches the features the scaler/model expect.
    Strategy:
    1. Prefer scaler.feature_names_in_ if present (exact names/order).
    2. Otherwise use scaler.mean_.shape[0] to determine expected count and derive that many
       columns from training_numeric_order while dropping obvious categorical engineered cols.
    3. Ensure X_imp_df has the used column names for SHAP/plots, but ALWAYS pass a numpy array
       of the correct shape to scaler.transform and model.predict_proba.
    Returns: X_scaled (numpy), X_imp_df (DataFrame)
    """
    # 1) Copy input and ensure base raw features exist
    X_in = input_df.copy()
    base_cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
    for c in base_cols:
        if c not in X_in.columns:
            X_in[c] = np.nan

    # 2) Decide which columns the scaler/model expect
    expected_cols = None
    expected_n = None

    # If scaler has explicit feature names recorded, use them (most reliable)
    if 'scaler' in globals() and scaler is not None:
        if hasattr(scaler, "feature_names_in_"):
            expected_cols = list(getattr(scaler, "feature_names_in_"))
            expected_n = len(expected_cols)
        else:
            try:
                expected_n = int(getattr(scaler, "mean_").shape[0])
            except Exception:
                expected_n = None

    # If no scaler info, try model.n_features_in_
    if expected_cols is None and expected_n is None and 'model' in globals() and model is not None:
        try:
            expected_n = int(getattr(model, "n_features_in_"))
        except Exception:
            expected_n = None

    # Fallback: use training_numeric_order
    numeric_order = training_numeric_order[:]  # copy

    # If we don't yet have expected_cols but do have expected_n: derive columns by removing known engineered categorical columns
    if expected_cols is None and expected_n is not None:
        # Remove clearly non-numeric engineered columns if present
        discard = {"Glucose_category", "BMI_category", "Age_bin"}
        filtered = [c for c in numeric_order if c not in discard]
        # If filtered still longer than expected_n, truncate; if shorter, pad with *_z zeros or paders
        if len(filtered) >= expected_n:
            expected_cols = filtered[:expected_n]
        else:
            # take filtered then fill with remaining numeric_order entries (including *_z) and pad if still short
            extra = [c for c in numeric_order if c not in filtered]
            combined = filtered + extra
            if len(combined) >= expected_n:
                expected_cols = combined[:expected_n]
            else:
                # pad with filler names
                pad_count = expected_n - len(combined)
                expected_cols = combined + [f"__pad_feature_{i}" for i in range(pad_count)]

    # If still no expected_n/cols, default to the numeric_order trimmed to 16 (safe)
    if expected_cols is None and expected_n is None:
        expected_cols = [c for c in numeric_order if not c.endswith("_category") and not c.endswith("_bin")]
        # if too long: pick first 16, else pad
        if len(expected_cols) >= 16:
            expected_cols = expected_cols[:16]
        else:
            extra = [c for c in numeric_order if c not in expected_cols]
            expected_cols = expected_cols + extra
            if len(expected_cols) < 16:
                expected_cols += [f"__pad_feature_{i}" for i in range(16 - len(expected_cols))]

    # Ensure expected_n is set
    expected_n = len(expected_cols)

    # 3) Build X_num DataFrame with those expected_cols in exact order.
    X_num = pd.DataFrame(index=X_in.index)
    for col in expected_cols:
        if col in X_in.columns:
            # if raw exists, cast to float where possible
            try:
                X_num[col] = X_in[col].astype(float)
            except Exception:
                X_num[col] = pd.to_numeric(X_in[col], errors="coerce")
        elif col.endswith("_z"):
            # standardized column -> mean = 0
            X_num[col] = 0.0
        else:
            # missing engineered or pad column -> zeros
            X_num[col] = 0.0

    # 4) Impute using train imputer (prefer) or median
    if imputer is not None:
        try:
            arr_imp = imputer.transform(X_num)
            X_imp_df = pd.DataFrame(arr_imp, columns=X_num.columns, index=X_num.index)
        except Exception:
            X_imp_df = X_num.fillna(X_num.median())
    else:
        X_imp_df = X_num.fillna(X_num.median())

    # 5) ALWAYS pass numpy array to scaler to avoid feature-name checks
    X_imp_np = X_imp_df.values  # shape (1, expected_n)

    if 'scaler' in globals() and scaler is not None:
        try:
            X_scaled = scaler.transform(X_imp_np)
        except Exception:
            # adapt shape strictly
            if X_imp_np.shape[1] > expected_n:
                X_temp = X_imp_np[:, :expected_n]
            elif X_imp_np.shape[1] < expected_n:
                pad = np.zeros((X_imp_np.shape[0], expected_n - X_imp_np.shape[1]))
                X_temp = np.hstack([X_imp_np, pad])
            else:
                X_temp = X_imp_np
            try:
                X_scaled = scaler.transform(X_temp)
            except Exception as e:
                st.warning(f"Scaler transform ultimately failed; using unscaled values. Error: {e}")
                X_scaled = X_temp
    else:
        X_scaled = X_imp_np

    # Return scaled numpy array and the imputed DataFrame for explainability display
    return X_scaled, X_imp_df


def get_recommendations(prob, input_df):
    suggestions = []
    diet = []
    prob = float(prob)
    # Risk buckets
    if prob < 0.40:
        risk = "Low"
        suggestions.append("Maintain your current healthy lifestyle and routine checks.")
        diet.extend([
            "Balanced meals (vegetables, lean protein, whole grains).",
            "Limit sugary snacks and beverages."
        ])
    elif 0.40 <= prob <= 0.70:
        risk = "Moderate"
        suggestions.append("Adopt lifestyle changes & monitor glucose weekly.")
        suggestions.append("Reduce refined carbs and added sugar.")
        diet.extend([
            "Increase fiber: oats, legumes, leafy vegetables.",
            "Replace sugary drinks with water; prefer complex carbs."
        ])
    else:
        risk = "High"
        suggestions.append("High risk — consult a healthcare professional promptly.")
        suggestions.append("Adopt a low-carb, high-fiber diet and daily activity.")
        diet.extend([
            "Avoid sweets, sugar-sweetened beverages, white rice, and white bread.",
            "Prefer vegetables, lean protein, pulses, and controlled portions."
        ])

    # Personalization
    bmi_val = float(input_df["BMI"].values[0])
    glucose_val = float(input_df["Glucose"].values[0])
    age_val = int(input_df["Age"].values[0])

    if bmi_val >= 30:
        suggestions.append("BMI in obese range — target 5-10% weight loss over 6 months.")
        diet.append("Portion control & calorie-aware meals; track daily intake.")
    elif bmi_val >= 25:
        suggestions.append("Overweight — increase physical activity; reduce simple carbs.")
        diet.append("Add more vegetables; minimize fried foods.")

    if glucose_val > 140:
        suggestions.append("Fasting glucose appears elevated — consult for further testing.")
        diet.append("Prefer low-GI foods: barley, lentils, legumes.")

    if age_val >= 50:
        suggestions.append("Regular checkups recommended (every 6-12 months).")

    suggestions.append("Aim for 150 minutes of moderate exercise weekly, good sleep (7-8h), and stress management.")
    return risk, suggestions, diet

# Prediction & UI output
if submitted:
    st.markdown("---")
    st.subheader("Prediction & Personalized Plan")

    if model is None:
        st.error("No trained model available. Run src/train_models.py and ensure models are in models/ directory.")
    else:
        X_scaled, X_imp_df = prepare_input_for_model(input_raw)

        # predict probability
                # predict probability (robust)
        prob = None
        # First, prefer predict_proba
        try:
            prob = float(model.predict_proba(X_scaled)[0][1])
        except Exception as e_proba:
            # If predict_proba not available, try decision_function
            try:
                score = model.decision_function(X_scaled)
                # scale to [0,1]
                score = np.asarray(score)
                prob = float((score - score.min())/(score.max() - score.min() + 1e-9))
            except Exception as e_dec:
                # If neither are available, fall back to predict() and treat 1 as prob=1, 0 as 0
                try:
                    pred_lab = model.predict(X_scaled)[0]
                    prob = float(pred_lab)
                    st.info("Model does not support probabilities; showing label-derived probability (0 or 1).")
                except Exception as e_all:
                    st.error(f"Prediction failed with errors: predict_proba error: {e_proba}; decision_function error: {e_dec}; predict error: {e_all}")
                    st.stop()

        # At this point prob is a float in [0,1]
        pred_label = "High risk" if prob >= threshold else "Low risk"
        st.metric("Predicted probability of diabetes", f"{prob:.3f}")
        st.write(f"**Risk label (threshold={threshold:.2f}):** {pred_label}")
        
        # --- Risk meter (paste immediately after showing the prob/label) ---
        import matplotlib.pyplot as plt

        # map prob (0-1) to color: green (low) -> yellow (mid) -> red (high)
        def prob_to_color(p):
            if p < 0.40:
                return "#2ecc71"  # green
            elif p < 0.70:
                return "#f1c40f"  # yellow
            else:
                return "#e74c3c"  # red

        fig, ax = plt.subplots(figsize=(6, 0.6))
        ax.barh([0], [prob], color=prob_to_color(prob))
        ax.set_xlim(0, 1)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%','25%','50%','75%','100%'])
        ax.set_yticks([])
        ax.set_frame_on(False)
        # annotate percent
        ax.text(prob + 0.02 if prob < 0.9 else prob - 0.07, 0, f"{prob*100:.1f}%", va='center', fontweight='bold', color='black' if prob<0.9 else 'white')
        st.pyplot(fig)
        # small legend
        st.write(f"**Risk meter:** {prob*100:.1f}% — color indicates risk level (green→yellow→red)")


        # Personalized suggestions
        risk_level, suggestions, diet_plan = get_recommendations(prob, input_raw)
        st.markdown(f"### 🩺 Risk Level: **{risk_level}**")
        st.markdown("### 🧭 Personalized Suggestions")
        for s in suggestions:
            st.write(f"- {s}")

        st.markdown("### 🥗 Recommended Diet Plan (examples)")
        for d in diet_plan:
            st.write(f"- {d}")

        # SHAP explainability
                # ---- Explainability: robust SHAP handling ----
                # ---- Explainability: robust SHAP handling ----
        # ---- Explainability: robust SHAP handling (replace existing block) ----
        st.markdown("---")
        st.subheader("Explainability — feature contributions")
        try:
            import shap
            # Build an explainer; TreeExplainer preferred for tree models
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                explainer = shap.Explainer(model)

            shap_out = explainer(X_imp_df)

            # Normalize shap_out into a 2D ndarray of shape (n_samples, n_features)
            if hasattr(shap_out, "values"):
                vals = shap_out.values
            else:
                vals = np.asarray(shap_out)

            # Handle list-like outputs (common for multi-output / classification)
            if isinstance(vals, list):
                # Usually vals = [neg_array, pos_array]; prefer pos_array if present
                if len(vals) == 2:
                    arr = np.asarray(vals[1])
                else:
                    # collapse by summing across entries (fallback)
                    arr = np.sum([np.asarray(v) for v in vals], axis=0)
            else:
                arr = np.asarray(vals)

            # Some explainers return shape (n_classes, n_samples, n_features) -> try to bring to (n_samples, n_features)
            if arr.ndim == 3:
                # if shape[0] == n_classes, take class 1 if exists, else take last
                if arr.shape[0] >= 2:
                    arr = arr[1]  # pick class-1 contributions
                else:
                    arr = np.squeeze(arr, axis=0)

            # Ensure 2D shape
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            # Align arr features with X_imp_df columns: if mismatch, truncate or pad zeros
            n_feats = X_imp_df.shape[1]
            if arr.shape[1] > n_feats:
                arr = arr[:, :n_feats]
            elif arr.shape[1] < n_feats:
                pad = np.zeros((arr.shape[0], n_feats - arr.shape[1]))
                arr = np.hstack([arr, pad])

            # Compute mean absolute SHAP per feature and plot bar
            mean_abs = np.mean(np.abs(arr), axis=0)
            feat_names = X_imp_df.columns.tolist()
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(feat_names))))
            ax.barh(feat_names, mean_abs)
            ax.set_xlabel("mean |SHAP value|")
            ax.set_title("Feature contribution (mean |SHAP|)")
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.info("SHAP not available or failed to render robustly. Install 'shap' for fuller explainability. Error: " + str(e))

        # ---- Feature importances for tree models (safe) ----
        try:
            if hasattr(model, "feature_importances_"):
                fi = np.array(model.feature_importances_)
                feat_names = X_imp_df.columns.tolist()
                if fi.shape[0] == len(feat_names):
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(feat_names))))
                    ax.barh(feat_names, fi)
                    ax.set_xlabel("Importance")
                    ax.set_title("Model feature importances")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.write("Model feature_importances_ length differs from input features; skipping plot.")
        except Exception:
            pass


        # Download recommendation
        def format_reco(prob, label, suggestions, diet):
            lines = []
            lines.append("AegisDiab™ — Personalized Recommendation")
            lines.append(f"Predicted probability: {prob:.3f}")
            lines.append(f"Risk level: {label}")
            lines.append("\nSuggestions:")
            for s in suggestions:
                lines.append(f"- {s}")
            lines.append("\nDiet (examples):")
            for d in diet:
                lines.append(f"- {d}")
            return "\n".join(lines)

        reco_text = format_reco(prob, risk_level, suggestions, diet_plan)
        st.download_button("Download recommendation (txt)", reco_text, file_name="aegisdiab_recommendation.txt")
