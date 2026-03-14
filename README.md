# Credit Risk Prediction – End-to-End MLOps Pipeline

This project demonstrates a **production-ready machine learning pipeline** for predicting loan default risk using a supervised learning model.
The focus of the project is not just model training, but **operationalizing the ML system** with monitoring, deployment, and CI/CD.

The system exposes a **real-time API** that predicts whether a loan applicant is likely to default.

---

# Project Objective

Financial institutions need automated systems to evaluate **loan default risk**.
This project builds a system that:

• trains a credit risk model
• exposes it through a REST API
• deploys it to the cloud
• monitors predictions and usage
• tracks API metrics using observability tools

---

# Key Features

### Machine Learning Pipeline

* Data preprocessing and feature engineering
* Model training using scikit-learn
* Model evaluation
* Model persistence using joblib

### API Deployment

* REST API built with FastAPI
* Real-time predictions
* Interactive API documentation (Swagger)

### Monitoring & Observability

* Prometheus metrics collection
* Request tracking
* Latency monitoring
* Grafana visualization dashboards

### Cloud Deployment

* Application deployed to the cloud using Render
* Public endpoint accessible through browser

### CI/CD Automation

* GitHub Actions pipeline
* Automated build and testing

---

# System Architecture

```
Users / Applications
        │
        ▼
   FastAPI REST API
        │
        ▼
   Credit Risk Model
        │
        ▼
 Prediction Response
        │
        ▼
  Prometheus Metrics
        │
        ▼
  Grafana Dashboard
        │
        ▼
 Monitoring & Observability
```

---

# Architecture Flow

1️⃣ Users send loan details to the API
2️⃣ FastAPI receives the request
3️⃣ The trained ML model predicts loan default probability
4️⃣ Prediction result is returned to the user
5️⃣ API metrics are collected by Prometheus
6️⃣ Grafana visualizes system performance and request volume

---

# Technology Stack

| Category            | Tools          |
| ------------------- | -------------- |
| Programming         | Python         |
| Machine Learning    | scikit-learn   |
| Data Processing     | pandas, numpy  |
| Model Serialization | joblib         |
| API Framework       | FastAPI        |
| API Server          | Uvicorn        |
| Monitoring          | Prometheus     |
| Visualization       | Grafana        |
| CI/CD               | GitHub Actions |
| Cloud Deployment    | Render         |

---

# Project Structure

```
credit_risk_mlops
│
├── data
│
├── notebooks
│
├── src
│   │
│   ├── api
│   │   └── app.py
│   │
│   ├── features
│   │   └── feature_engineering.py
│   │
│   ├── models
│   │   └── train_model.py
│   │
│   └── monitoring
│       └── metrics.py
│
├── tests
│
├── model.pkl
├── requirements.txt
└── README.md
```

---

# Model Features

The model predicts credit risk using attributes such as:

• Person age
• Income
• Employment length
• Home ownership status
• Loan intent
• Loan grade
• Loan amount
• Interest rate
• Credit history length

---

# Running the Project Locally

### 1️⃣ Clone the repository

```
git clone https://github.com/dhwanitdesai13/credit_risk_mlops.git
cd credit_risk_mlops
```

### 2️⃣ Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Train the model

```
python src/models/train_model.py
```

### 5️⃣ Run the API

```
uvicorn src.api.app:app --reload
```

API will run at:

```
http://127.0.0.1:8000
```

Swagger docs:

```
http://127.0.0.1:8000/docs
```

---

# Example Prediction Request

Endpoint

```
POST /predict
```

Example request body

```
{
 "person_age": 35,
 "person_income": 60000,
 "person_home_ownership": "RENT",
 "person_emp_length": 5,
 "loan_intent": "PERSONAL",
 "loan_grade": "B",
 "loan_amnt": 10000,
 "loan_int_rate": 11.5,
 "loan_percent_income": 0.2,
 "cb_person_default_on_file": "N",
 "cb_person_cred_hist_length": 7
}
```

Example response

```
{
 "prediction": 0
}
```

---

# Monitoring Metrics

Prometheus collects metrics from:

```
/metrics
```

Example metrics:

• prediction_requests_total
• request_latency_seconds

These metrics are visualized through Grafana dashboards.

---

# CI/CD Pipeline

GitHub Actions performs:

• dependency installation
• code validation
• automated testing
• pipeline verification

Pipeline runs automatically on each push.

---

# Deployment

The API is deployed using **Render Cloud Platform**.

Deployment process:

1️⃣ Push code to GitHub
2️⃣ Render pulls the repository
3️⃣ Dependencies are installed
4️⃣ FastAPI application starts automatically

---

# Future Improvements

• Model versioning
• Alerting system for anomalies
• Feature store integration
• Data pipeline automation

---

# Author

Dhwanit Desai
AI • Machine Learning • MLOps
