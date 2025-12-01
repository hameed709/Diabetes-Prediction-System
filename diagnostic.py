# diagnostic.py
import joblib, pandas as pd, numpy as np, traceback, os, json

def main():
    try:
        print("Project root:", os.getcwd())
        # 1. X_train numeric cols
        xtrain_path = os.path.join("data","processed","X_train.csv")
        if os.path.exists(xtrain_path):
            Xtr = pd.read_csv(xtrain_path)
            num_cols = Xtr.select_dtypes(include=[np.number]).columns.tolist()
            print("X_train numeric columns (count={}):".format(len(num_cols)))
            print(num_cols)
        else:
            print("Missing:", xtrain_path)

        # 2. scaler info
        scaler_path = os.path.join("models","scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            try:
                mean_shape = getattr(scaler,"mean_").shape[0]
                print("Scaler mean_ length (features expected):", mean_shape)
            except Exception as e:
                print("Scaler loaded but no mean_ attribute or error:", e)
        else:
            print("Scaler not found at", scaler_path)

        # 3. pick a model file
        candidate_files = ["models/best_model.pkl","models/random_forest.pkl","models/xgb.pkl","models/logistic.pkl"]
        model = None
        model_file = None
        for f in candidate_files:
            if os.path.exists(f):
                try:
                    model = joblib.load(f)
                    model_file = f
                    break
                except Exception as e:
                    print("Failed to load", f, ":", e)
        if model is None:
            # try listing any .pkl in models/
            for f in os.listdir("models") if os.path.exists("models") else []:
                if f.endswith(".pkl"):
                    try:
                        model = joblib.load(os.path.join("models", f))
                        model_file = os.path.join("models", f)
                        break
                    except:
                        continue

        if model is not None:
            print("Loaded model:", model_file)
            print("Model type:", type(model))
            print("Model n_features_in_:", getattr(model, "n_features_in_", "N/A"))
            # show if supports predict_proba
            print("Has predict_proba:", hasattr(model, "predict_proba"))
        else:
            print("No model loaded from models/")

        # 4. Now build a test row using your Low Risk example
        # Values: Pregnancies=0, Glucose=90, BloodPressure=72, SkinThickness=22, Insulin=60, BMI=23, DPF=0.2, Age=25
        test_vals = {
            "Pregnancies": 0,
            "Glucose": 90.0,
            "BloodPressure": 72.0,
            "SkinThickness": 22.0,
            "Insulin": 60.0,
            "BMI": 23.0,
            "DiabetesPedigreeFunction": 0.2,
            "Age": 25
        }

        # get numeric cols we will use (use Xtr numeric cols if available)
        if os.path.exists(xtrain_path):
            numeric_order = Xtr.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_order = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

        print("Numeric order to build test input (len={}):".format(len(numeric_order)))
        print(numeric_order)

        # Build DataFrame
        X_test = pd.DataFrame([ [ test_vals.get(c, 0.0) if not c.endswith("_z") else 0.0 for c in numeric_order ] ], columns=numeric_order)

        # Use imputer if available
        imputer_path = os.path.join("models","train_imputer.pkl")
        if os.path.exists(imputer_path):
            imputer = joblib.load(imputer_path)
            try:
                X_imp = pd.DataFrame(imputer.transform(X_test), columns=numeric_order)
                print("Applied train_imputer. Any NaNs after imputer:", X_imp.isna().sum().sum())
            except Exception as e:
                print("Imputer transform failed:", e)
                X_imp = X_test.fillna(X_test.median())
        else:
            print("train_imputer.pkl not found; using median fill")
            X_imp = X_test.fillna(X_test.median())

        # Scale with scaler (use numpy to avoid name checks)
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
                X_scaled = scaler.transform(X_imp.values)  # pass numpy array
                print("Scaled test input shape:", X_scaled.shape)
            except Exception as e:
                print("Scaler transform failed:", e)
                X_scaled = X_imp.values
        else:
            X_scaled = X_imp.values

        # Run model prediction if model loaded
        if model is not None:
            try:
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X_scaled)[0][1]
                    print("Model predict_proba -> probability:", prob)
                else:
                    pred = model.predict(X_scaled)[0]
                    print("Model predict -> label:", pred)
            except Exception as e:
                print("Model prediction failed:")
                traceback.print_exc()
        else:
            print("No model to predict with.")

    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()
