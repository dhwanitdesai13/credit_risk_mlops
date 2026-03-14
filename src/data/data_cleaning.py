import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):

    # Handle missing values
    df["person_emp_length"].fillna(df["person_emp_length"].median(), inplace=True)

    df["loan_int_rate"].fillna(df["loan_int_rate"].median(), inplace=True)

    # Remove unrealistic ages
    df = df[df["person_age"] < 100]

    return df


def save_data(df, path):
    df.to_csv(path, index=False)


if __name__ == "__main__":

    raw_path = "data/raw/credit_risk_dataset.csv"
    processed_path = "data/processed/credit_risk_cleaned.csv"

    df = load_data(raw_path)

    df_clean = clean_data(df)

    save_data(df_clean, processed_path)

    print("Data cleaning completed")