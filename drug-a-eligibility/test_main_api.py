# test_main_api.py

from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_predict_endpoint():
    sample_input = {
        "age": 72,
        "gender": "M",
        "high_risk": 1,
        "days_since_symptom": 2,
        "multiple_meds": True,
        "frequent_visits": False
    }
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "likelihood_of_treatment" in response.json()
    assert 0.0 <= response.json()["likelihood_of_treatment"] <= 1.0
