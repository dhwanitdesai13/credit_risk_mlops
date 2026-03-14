import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(path):

    df = pd.read_csv(path)

    return df


def split_data(df):

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def create_feature_pipeline():

    numeric_features = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ]

    categorical_features = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor