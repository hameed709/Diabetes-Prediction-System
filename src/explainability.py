# explainability.py
import joblib
import pandas as pd
import shap
import os
import matplotlib.pyplot as plt

MODELS_DIR = "models"
PROC_DIR = os.path.join("data", "processed")
REPORT_DIR = os.path.join("reports", "figures")
os.makedirs(REPORT_DIR, exist_ok=True)

def load_test():
    X_test = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))
    return X_test

def explain_xgb():
    xgb = joblib.load(os.path.join(MODELS_DIR, "xgb.pkl"))
    X_test = load_test()
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)
    # summary plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(REPORT_DIR, "shap_summary_xgb.png"), bbox_inches='tight')
    plt.close()
    print("SHAP explanation saved for XGBoost.")

if __name__ == "__main__":
    explain_xgb()
