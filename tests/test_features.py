import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.feature_engineering import load_data, split_data, create_feature_pipeline


df = load_data("data/processed/credit_risk_cleaned.csv")

X_train, X_test, y_train, y_test = split_data(df)

pipeline = create_feature_pipeline()

pipeline.fit(X_train)

X_train_transformed = pipeline.transform(X_train)

print("Feature pipeline working")
print(X_train_transformed.shape)