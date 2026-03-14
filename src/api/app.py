import sys
import os
import time
import pandas as pd
import joblib

from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

app = FastAPI()

model = joblib.load("model.pkl")

REQUEST_COUNT = Counter("api_request_count", "Total prediction requests")
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Prediction latency")


@app.get("/")
def home():
    return {"message": "Credit Risk Prediction API"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.post("/predict")
def predict(data: dict):

    start_time = time.time()

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(time.time() - start_time)

    # -------- Prediction Logging --------

    log_data = df.copy()
    log_data["prediction"] = prediction
    log_data["timestamp"] = pd.Timestamp.now()

    log_file = "data/predictions/prediction_logs.csv"

    if os.path.exists(log_file):
        log_data.to_csv(log_file, mode="a", header=False, index=False)
    else:
        log_data.to_csv(log_file, index=False)

    return {"prediction": int(prediction)}