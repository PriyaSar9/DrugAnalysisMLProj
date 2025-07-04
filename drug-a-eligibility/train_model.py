# train_model.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load preprocessed and merged dataset (adjust path if needed)
data = pd.read_csv("data/final_model_data.csv")

# Define feature columns and target
feature_cols = [
    "age", "gender", "high_risk", "days_since_symptom",
    "multiple_meds", "frequent_visits"
]
X = data[feature_cols].copy()
X["gender"] = X["gender"].map({"M": 1, "F": 0})  # Convert gender to numeric
y = data["treated"]

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model/model.pkl")
print("Model saved to model/model.pkl")
