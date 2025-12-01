# Diabetes Prediction System (Final Year Project)

## Setup
1. Clone repo and change directory:
   ```
   git clone <repo-url>
   cd diabetes-prediction
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Put dataset CSV as `data/raw/pima_diabetes.csv` (Pima Indians Diabetes CSV with columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome)

## Steps
1. Preprocess:
   ```
   python src/data_prep.py
   ```
2. Train models:
   ```
   python src/train_models.py
   ```
3. Evaluate:
   ```
   python src/evaluate.py
   ```
4. Explain:
   ```
   python src/explainability.py
   ```
5. Run app:
   ```
   streamlit run src/app_streamlit.py
   ```

## Docker
```
docker build -t diabetes-app .
docker run -p 8501:8501 diabetes-app
```

## Deliverables
- `models/` contains persisted models
- `reports/figures` contains figures and model metrics
- `reports/final_report.md` — fill with methodology and results
