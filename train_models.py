"""
Train five classifiers on the UCI "Predict Students' Dropout and Academic Success"
dataset, evaluate them on a held-out test split, and persist everything the
Streamlit app needs (fitted pipelines, label encoder, feature list, metrics).

Run:  python train_models.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from metrics_utils import compute_metrics

ROOT = Path(__file__).parent
RAW_CSV = ROOT / "data" / "students_dropout_full.csv"
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"
TARGET_COL = "Target"
SEED = 26


def load_dataset() -> pd.DataFrame:
    # The UCI file is ';'-separated and one header carries a stray tab character
    df = pd.read_csv(RAW_CSV, sep=";")
    df.columns = [c.strip().replace("\t", "") for c in df.columns]
    return df


def build_models() -> dict:
    """Distance/margin-based models get a scaler; trees and NB work on raw values."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=8, min_samples_leaf=10,
                                           random_state=SEED)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance")),
        ]),
        "Naive Bayes": Pipeline([
            ("clf", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                        random_state=SEED)),
        ]),
    }


def main():
    df = load_dataset()
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    print(f"Dataset shape: {df.shape}  |  features: {len(feature_cols)}")
    print(df[TARGET_COL].value_counts().to_string(), "\n")

    encoder = LabelEncoder().fit(df[TARGET_COL])
    X = df[feature_cols]
    y = encoder.transform(df[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # Hold-out split is exported so the app can be evaluated on unseen data
    test_df = X_test.copy()
    test_df[TARGET_COL] = encoder.inverse_transform(y_test)
    test_df.to_csv(TEST_CSV, index=False)
    print(f"Saved test split ({len(test_df)} rows) -> {TEST_CSV.name}\n")

    MODEL_DIR.mkdir(exist_ok=True)
    results = {}
    for name, pipe in build_models().items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)
        results[name] = compute_metrics(y_test, y_pred, y_proba)

        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipe, MODEL_DIR / fname)
        print(f"{name:<24} " + "  ".join(f"{k}={v:.4f}" for k, v in results[name].items()))

    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    meta = {
        "target": TARGET_COL,
        "features": feature_cols,
        "classes": encoder.classes_.tolist(),
        "model_files": {n: n.lower().replace(" ", "_") + ".joblib" for n in results},
        "test_metrics": results,
        "seed": SEED,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))

    table = pd.DataFrame(results).T.round(4)
    table.index.name = "ML Model Name"
    table.to_csv(MODEL_DIR / "metrics_comparison.csv")
    print("\n" + table.to_string())


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
