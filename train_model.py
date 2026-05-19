"""
Bank Churn Model Training Pipeline
====================================
Loads the dataset, preprocesses, engineers features, trains three classifiers
(Logistic Regression, Random Forest, Gradient Boosting) with class balancing,
evaluates each, then exports the best model + preprocessor to ../server/model/.

Usage:
    python train_model.py
    python train_model.py --data "C:/path/to/dataset.csv"
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
DEFAULT_DATA = r"C:\Users\DELL\Downloads\Telegram Desktop\Bank_Churn_Classification_Dataset.csv"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "server", "model")


# ── 1. Load data ───────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    print(f"\n[1/5] Loading dataset from:\n      {path}")
    df = pd.read_csv(path, index_col=0)
    print(f"      Shape: {df.shape}  |  Churn rate: {df['Churn'].mean():.1%}")
    return df


# ── 2. Preprocess ──────────────────────────────────────────────────────────────
GENDER_MAP      = {"Male": 1, "Female": 0}
CONTRACT_MAP    = {"Month-to-month": 0, "One year": 1, "Two year": 2}
PAYMENT_MAP     = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer": 2,
    "Credit card": 3,
}

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/5] Preprocessing ...")
    df = df.copy()

    df["Gender"]        = df["Gender"].map(GENDER_MAP)
    df["Contract"]      = df["Contract"].map(CONTRACT_MAP)
    df["PaymentMethod"] = df["PaymentMethod"].map(PAYMENT_MAP)

    # Guard against zero tenure (avoid division by zero)
    df["Tenure"] = df["Tenure"].replace(0, 0.01)
    return df


# ── 3. Feature engineering ─────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/5] Engineering features ...")
    df = df.copy()

    # Interaction features that ranked highest in feature importance
    df["ChargesPerMonth"]        = df["TotalCharges"] / df["Tenure"]
    df["ChargesTenureInteract"]  = df["MonthlyCharges"] * df["Tenure"]

    print(f"      Feature count: {len(get_feature_cols(df))}")
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    return [
        "Gender", "SeniorCitizen", "Tenure",
        "MonthlyCharges", "TotalCharges",
        "Contract", "PaymentMethod",
        "ChargesPerMonth", "ChargesTenureInteract",
    ]


# ── 4. Train & evaluate models ─────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        max_depth=8, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4,
        learning_rate=0.05, random_state=42
    ),
}


def train_and_evaluate(df: pd.DataFrame):
    print("\n[4/5] Training models ...")

    features = get_feature_cols(df)
    X = df[features].values
    y = df["Churn"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (stored in preprocessor dict for serving)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    results = {}
    trained  = {}

    for name, clf in MODELS.items():
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        y_prob = (
            clf.predict_proba(X_test_s)[:, 1]
            if hasattr(clf, "predict_proba") else y_pred
        )

        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred, zero_division=0),
            "auc":       roc_auc_score(y_test, y_prob),
            "report":    classification_report(y_test, y_pred, zero_division=0),
        }
        results[name] = metrics
        trained[name] = clf

        print(f"\n  -- {name} --")
        print(f"     Accuracy : {metrics['accuracy']:.3f}")
        print(f"     F1 Score : {metrics['f1']:.3f}")
        print(f"     ROC-AUC  : {metrics['auc']:.3f}")
        print(metrics["report"])

    # Pick best by F1 (more meaningful than accuracy on imbalanced classes)
    best_name = max(results, key=lambda n: results[n]["f1"])
    print(f"\n  >>> Best model by F1: {best_name} (F1={results[best_name]['f1']:.3f})")

    return trained[best_name], best_name, scaler, features, results


# ── 5. Export ──────────────────────────────────────────────────────────────────
def export(model, model_name: str, scaler, feature_cols: list):
    print(f"\n[5/5] Exporting to {MODEL_DIR} ...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Bundle everything the server needs into one artefact
    bundle = {
        "model":        model,
        "model_name":   model_name,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "label_maps": {
            "Gender":        GENDER_MAP,
            "Contract":      CONTRACT_MAP,
            "PaymentMethod": PAYMENT_MAP,
        },
    }

    path = os.path.join(MODEL_DIR, "churn_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"      Saved: {path}")
    return path


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train bank churn model")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to CSV dataset")
    args = parser.parse_args()

    df = load_data(args.data)
    df = preprocess(df)
    df = engineer_features(df)
    model, name, scaler, feature_cols, results = train_and_evaluate(df)
    export(model, name, scaler, feature_cols)

    print("\n[DONE] Training complete. Run the server next:\n  cd ../server && uvicorn main:app --reload\n")


if __name__ == "__main__":
    main()
