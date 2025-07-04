# preprocess.py

import pandas as pd


def preprocess_data(fact_txn, dim_patient, dim_physician):
    # Merge datasets
    df = fact_txn.merge(dim_patient, on="patient_id", how="left")
    df = df.merge(dim_physician, on="physician_id", how="left")

    # Compute derived features
    df["days_since_symptom"] = (
        pd.to_datetime(df["visit_date"]) - pd.to_datetime(df["symptom_start_date"])
    ).dt.days

    # High-risk condition logic
    high_risk_conditions = [
        "chronic_lung_disease", "cardiovascular_disease", "cancer",
        "immune_suppression", "obesity", "diabetes", "smoking"
    ]
    df["high_risk"] = ((df["age"] >= 65) | (df[high_risk_conditions].sum(axis=1) > 0)).astype(int)

    # Example contraindication medications
    contraindications = ["contra_drug_a", "contra_drug_b"]
    df["multiple_meds"] = (df[contraindications].sum(axis=1) > 1).astype(int)

    # Frequent visits
    df["frequent_visits"] = (df["visit_count_30_days"] >= 3).astype(int)

    # Select features
    features = [
        "patient_id", "age", "gender", "high_risk", "days_since_symptom",
        "multiple_meds", "frequent_visits", "treated"
    ]
    df_model = df[features].dropna()

    return df_model


# Sample usage (if running as script)
if __name__ == "__main__":
    path = "data/DSI LT Interview Exercise - May 2025 (candidate).xlsx"
    fact = pd.read_excel(path, sheet_name="fact_txn")
    pat = pd.read_excel(path, sheet_name="dim_patient")
    doc = pd.read_excel(path, sheet_name="dim_physician")

    df_final = preprocess_data(fact, pat, doc)
    df_final.to_csv("data/final_model_data.csv", index=False)
    print("Saved processed data to data/final_model_data.csv")
