from fastapi import FastAPI
from fastapi.responses import Response
import pandas as pd
import joblib
import os
from prometheus_client import Counter, Histogram, generate_latest

app = FastAPI()

# ==============================
# Load Model
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(MODEL_PATH)

# ==============================
# Monitoring Metrics
# ==============================

prediction_counter = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction latency"
)

# ==============================
# Prediction Logging
# ==============================

LOG_FILE = os.path.join(BASE_DIR, "prediction_logs.csv")


def log_prediction(input_data, prediction):

    row = input_data.copy()
    row["prediction"] = prediction

    df = pd.DataFrame([row])

    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)


# ==============================
# Health Endpoint
# ==============================

@app.get("/")
def home():
    return {"message": "Credit Risk Prediction API running"}


# ==============================
# Prediction Endpoint
# ==============================

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    with prediction_latency.time():

        prediction = model.predict(df)[0]

    prediction_counter.inc()

    log_prediction(data, prediction)

    return {"prediction": int(prediction)}


# ==============================
# Metrics Endpoint
# ==============================

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type="text/plain"
    )