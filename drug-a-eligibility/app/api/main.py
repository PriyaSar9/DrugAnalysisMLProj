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
    days_since_symptom: int
    multiple_meds: bool
    frequent_visits: bool

@app.post("/predict")
def predict(input: PatientInput):
    features = [[
        input.age, 1 if input.gender == 'M' else 0,
        input.high_risk, input.days_since_symptom,
        int(input.multiple_meds), int(input.frequent_visits)
    ]]
    prediction = model.predict_proba(features)[0][1]
    return {"likelihood_of_treatment": prediction}