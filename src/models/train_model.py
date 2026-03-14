import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from src.features.feature_engineering import load_data, split_data, create_feature_pipeline


def train():

    print("Loading dataset...")

    df = load_data("data/processed/credit_risk_cleaned.csv")

    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = create_feature_pipeline()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print("Training model...")

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print("Accuracy:", acc)

    joblib.dump(pipeline, "model.pkl")

    print("Model saved as model.pkl")


if __name__ == "__main__":
    train()