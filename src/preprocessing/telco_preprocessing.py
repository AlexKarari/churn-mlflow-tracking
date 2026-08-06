"""
telco_preprocessing.py
=======================
Shared preprocessing pipeline for the Telco Customer Churn dataset.

Why this file exists
---------------------
Logistic Regression, Decision trees, Random Forest, and Gradient Boosting were each built in separate project folders and preprocessed
the data slightly differently (e.g. Logistic Regression README reports
30 encoded features; Decision Tree README reports 26). That's fine in
isolation, but it breaks a fair MLflow comparison — if each model saw different
features, differences in accuracy partly reflect differences in preprocessing,
not the algorithm.

This module is the single source of truth going forward: one cleaning +
encoding + split, reused by every model in src/tracking/mlflow_experiments.py.
It follows the Day 3 (Logistic Regression) approach — one-hot encoding with
drop_first=True — since that's the most complete/documented pipeline of the
four, and it works fine for tree-based models too (trees just don't need the
StandardScaler part, but it doesn't hurt them either).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET_COL = "Churn"
ID_COL = "customerID"


def load_and_prepare(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Load the Telco churn CSV and produce a train/test split ready for any
    of the four models (scratch or sklearn).

    Steps
    -----
    1. Drop customerID (identifier, not a feature)
    2. Coerce TotalCharges to numeric (11 rows are blank strings for
       brand-new customers with tenure=0 -> filled with 0.0)
    3. Encode target: Churn Yes/No -> 1/0
    4. One-hot encode all categorical columns (drop_first=True to avoid
       multicollinearity)
    5. Train/test split (stratified on target to preserve the ~26.5% churn rate)
    6. Fit StandardScaler on numeric columns using TRAIN only, apply to both

    Returns
    -------
    X_train, X_test : np.ndarray
        Scaled, encoded feature matrices.
    y_train, y_test : np.ndarray
        Binary target arrays.
    feature_names : list[str]
        Column names in the same order as the feature matrix columns.
    """
    df = pd.read_csv(csv_path)

    df = df.drop(columns=[ID_COL])

    # TotalCharges has blank strings, not NaN, for tenure=0 customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)

    categorical_cols = [
        c for c in df.columns
        if c not in NUMERIC_COLS + [TARGET_COL] and df[c].dtype == object
    ]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # SeniorCitizen is already 0/1 int; everything else non-numeric is now encoded
    y = df[TARGET_COL].values
    X_df = df.drop(columns=[TARGET_COL])
    feature_names = X_df.columns.tolist()

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_df = X_train_df.copy()
    X_test_df = X_test_df.copy()
    X_train_df[NUMERIC_COLS] = scaler.fit_transform(X_train_df[NUMERIC_COLS])
    X_test_df[NUMERIC_COLS] = scaler.transform(X_test_df[NUMERIC_COLS])

    X_train = X_train_df.values.astype(float)
    X_test = X_test_df.values.astype(float)

    return X_train, X_test, y_train, y_test, feature_names
