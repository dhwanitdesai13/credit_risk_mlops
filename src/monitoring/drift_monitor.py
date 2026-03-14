import pandas as pd
from scipy.stats import ks_2samp


def detect_drift(reference_data, current_data):

    drift_results = {}

    for column in reference_data.columns:

        if reference_data[column].dtype != "object":

            stat, p_value = ks_2samp(reference_data[column], current_data[column])

            drift_results[column] = {
                "p_value": p_value,
                "drift_detected": p_value < 0.05
            }

    return drift_results


if __name__ == "__main__":

    reference = pd.read_csv("data/processed/credit_risk_cleaned.csv")

    # simulate new production data
    current = reference.sample(500)

    drift_report = detect_drift(reference, current)

    print("\nDRIFT REPORT\n")

    for feature, result in drift_report.items():
        print(feature, "→", result)