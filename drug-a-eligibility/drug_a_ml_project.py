# Drug A Analysis Project: EMR Alert for Treatment Eligibility

## Phase 1: Data Exploration and Cleaning

import pandas as pd
import numpy as np

# Load data from Excel file
file_path = "/home/ubuntu/DrugAnalysisMLProj/drug-a-eligibility/data/DSI.xlsx"
fact_txn = pd.read_excel(file_path, sheet_name="fact_txn")
dim_patient = pd.read_excel(file_path, sheet_name="dim_patient")
dim_physician = pd.read_excel(file_path, sheet_name="dim_physician")
model_table = pd.read_excel(file_path, sheet_name="model_table")
data_dictionary = pd.read_excel(file_path, sheet_name="Data Dictionary")

# Initial inspection
print(fact_txn.info())
print(dim_patient.info())
print(dim_physician.info())

## Phase 2: Data Processing and Feature Engineering

# Merge data into single DataFrame based on patient_id and physician_id
data = fact_txn.merge(dim_patient, on="patient_id", how="left")
data = data.merge(dim_physician, on="physician_id", how="left")

# Convert date columns and calculate days since symptom onset
#data["days_since_symptom"] = (pd.to_datetime(data["visit_date"]) - pd.to_datetime(data["symptom_start_date"]))
#data["days_since_symptom"] = data["days_since_symptom"].dt.days

# Identify high-risk patients
high_risk_conditions = ["hypertension", "heart_disease", "muscle_ache", "difficulty_breathing", "obesity", "diabetes", "cough"]
data["high_risk"] = ((data["age"] >= 65) | (data[high_risk_conditions].sum(axis=1) > 0)).astype(int)

# Feature: multiple_medications
contraindications = ["high_contraindication", "low_contraindication"]  # placeholder, update with real contraindications

data["multiple_meds"] = data[contraindications].sum(axis=1) > 1

# Feature: recent_physician_visits
# Assuming count_of_visits_last_30_days is available
df["txn_dt"] = to_datetime(data["txn_dt"])
df = df.sort_values(by=['patient_id', 'txn_dt'])
df['date_diff'] = df.groupby('patient_id')['txn_dt'].diff().dt.days
data["frequent_visits"] = data["date_diff"] >= 30

# Final model dataset
model_features = ["patient_id", "age", "gender", "high_risk", "multiple_meds", "frequent_visits", "treated"]
model_df = data[model_features].dropna()

## Phase 3: Machine Learning Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X = model_df.drop(["patient_id", "treated"], axis=1)
y = model_df["treated"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(classification_report(y_test, preds))

## Phase 4: Software Architecture (Python + FastAPI)
# Structure:
# /app
#   /api
#     main.py         - FastAPI REST API
#   /model
#     model.pkl       - Trained model
#     preprocess.py   - Feature generation
# Dockerfile          - Containerization
# requirements.txt    - Dependencies

## FastAPI Server (main.py)
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model/model.pkl")

class PatientInput(BaseModel):
    age: int
    gender: str
    high_risk: int
    multiple_meds: bool
    frequent_visits: bool

@app.post("/predict")
def predict(input: PatientInput):
    features = [[
        input.age, 1 if input.gender == 'M' else 0,
        input.high_risk, int(input.multiple_meds), int(input.frequent_visits)
    ]]
    prediction = model.predict_proba(features)[0][1]
    return {"likelihood_of_treatment": prediction}

## Phase 5: Dockerfile
# Dockerfile
# FROM python:3.10-slim
# WORKDIR /app
# COPY . /app
# RUN pip install -r requirements.txt
# CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

## Phase 6: Version Control
# Use Git best practices:
# - Feature branches for each module (data cleaning, ML, API)
# - Pull requests and code review
# - Git tags for model versions (v1.0-model, v2.0-api)
# - Store model metadata (accuracy, precision) alongside model artifacts
# - Conflict resolution: rebase or merge, ensure clean history

# README.md should include:
# - Setup instructions
# - How to run the API
# - Example input and output
# - Description of model and features used
