# src/feature_engineering.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # --- Age bin ---
        X["Age_bin"] = pd.cut(
            X["Age"],
            bins=[0, 30, 45, 60, 120],
            labels=[0, 1, 2, 3],
            include_lowest=True
        )
        X["Age_bin"] = X["Age_bin"].cat.add_categories([-1]).fillna(-1).astype(int)

        # --- BMI category ---
        X["BMI_category"] = pd.cut(
            X["BMI"],
            bins=[0, 18.5, 25, 30, 100],
            labels=[0, 1, 2, 3],
            include_lowest=True
        )
        X["BMI_category"] = X["BMI_category"].cat.add_categories([-1]).fillna(-1).astype(int)

        # --- Glucose category ---
        X["Glucose_category"] = pd.cut(
            X["Glucose"],
            bins=[0, 100, 140, 200, 500],
            labels=[0, 1, 2, 3],
            include_lowest=True
        )
        X["Glucose_category"] = X["Glucose_category"].cat.add_categories([-1]).fillna(-1).astype(int)

        return X
