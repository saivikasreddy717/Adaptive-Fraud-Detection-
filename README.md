# Adaptive Fraud Detection

Adaptive Fraud Detection is an end-to-end fraud detection pipeline that combines deep anomaly detection (via an Anomaly Transformer) and gradient boosting (XGBoost). It ingests real-time transaction data using Apache Kafka, stores data in Delta Lake for both batch and streaming analytics, and is containerized with Docker and orchestrated on Kubernetes for scalable deployment.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Folder Structure](#folder-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Generate Synthetic Dataset](#1-generate-synthetic-dataset)
  - [2. Kafka Setup and Data Ingestion](#2-kafka-setup-and-data-ingestion)
  - [3. Delta Lake Ingestion](#3-delta-lake-ingestion)
  - [4. Model Training and Prediction](#4-model-training-and-prediction)
  - [5. Containerization with Docker](#5-containerization-with-docker)
  - [6. Kubernetes Deployment](#6-kubernetes-deployment)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

This project implements a real-time fraud detection workflow that:

- **Streams and Stores Data:** Apache Kafka streams transaction data in real time, while Delta Lake manages batch and streaming ingestion.
- **Anomaly Detection:** An **Anomaly Transformer** analyzes time-series data to produce a reconstruction error signal.
- **Fraud Classification:** Combines the computed anomaly signal with tabular features and trains an **XGBoost** model for fraud detection.
- **Containerized and Deployable:** Docker is used to containerize the application, and Kubernetes ensures scalability and high availability in production environments.

---

## Features

- **End-to-End Pipeline** – Data ingestion, anomaly detection, classification, and deployment.
- **Time-Series Analysis** – Employs an Anomaly Transformer to detect anomalies in transaction time-series data.
- **XGBoost Classifier** – Uses gradient boosting for robust binary classification of fraud vs. non-fraud transactions.
- **Real-Time Ingestion** – Kafka provides continuous streaming of new transaction data.
- **Reliable Storage** – Delta Lake supports both batch and streaming writes for analytics.
- **Containerization & Orchestration** – Docker images facilitate portability, and Kubernetes ensures scalable deployment.

---

## Folder Structure

```plaintext
adaptive-fraud-detection/
├── data/
│   ├── transactions.csv          # Synthetic dataset (generated)
│   └── generate_dataset.py       # Script to generate a synthetic dataset
├── ingestion/
│   ├── kafka_producer.py         # Sends data from CSV to Kafka
│   ├── kafka_consumer.py         # (Optional) Consumes and prints Kafka messages
│   └── delta_ingest.py           # PySpark script to ingest Kafka messages into Delta Lake
├── models/
│   ├── anomaly_transformer.py    # Implementation of the Anomaly Transformer model
│   ├── pipeline.py               # Pipeline combining anomaly transformer + XGBoost
│   └── tabnet_model.py           # (Optional) TabNet model (not used if sticking with XGBoost)
├── evaluation/
│   └── evaluate.py               # Script for evaluating model performance (e.g., ROC AUC)
├── Dockerfile                    # Docker build configuration
├── docker-compose.yml            # Docker Compose file for Kafka and Zookeeper (if needed)
├── k8s-deployment.yaml           # Kubernetes deployment and service definitions
├── requirements.txt              # Python dependencies
└── README.md                     # This file
Requirements
Python 3.9+

PyTorch

XGBoost

Pandas, NumPy, Scikit-learn

Apache Kafka (or equivalent Docker containers)

PySpark & delta-spark (for Delta Lake integration)

Docker

Kubernetes CLI (kubectl, kubeconfig)

(Optional) Docker Compose

Installation
Clone the Repository:

bash
Copy
git clone https://github.com/your-username/adaptive-fraud-detection.git
cd adaptive-fraud-detection
Install Python Dependencies:

bash
Copy
pip install -r requirements.txt
Configure Environment Variables:

If required, set environment variables (e.g., SPARK_HOME) in your shell configuration.

Usage
1. Generate Synthetic Dataset
If you don’t have real transaction data available, generate a synthetic dataset:

bash
Copy
python data/generate_dataset.py
This creates a CSV file at data/transactions.csv with ~10,000 synthetic records.

2. Kafka Setup and Data Ingestion
Start Kafka and Zookeeper:

If you are using Docker Compose, run:

bash
Copy
docker-compose up -d
Verify that Kafka and Zookeeper containers are running with:

bash
Copy
docker ps
Run the Kafka Producer:

Push the synthetic data to Kafka:

bash
Copy
python ingestion/kafka_producer.py
This script reads transactions.csv and sends each record as a JSON message to the Kafka topic (e.g., fraud_topic).

3. Delta Lake Ingestion
Stream data from Kafka into Delta Lake using PySpark:

bash
Copy
python ingestion/delta_ingest.py
This script reads messages from the Kafka topic and writes them to a Delta Lake table at /tmp/delta/transactions (adjust the path if needed).

4. Model Training and Prediction
Train the anomaly transformer and XGBoost model, then generate predictions:

bash
Copy
python models/pipeline.py
The pipeline script:

Trains the Anomaly Transformer on time-series data.

Computes reconstruction errors (anomaly signals).

Combines the anomaly errors with additional tabular data.

Splits the data into training and validation sets.

Trains an XGBoost classifier for fraud prediction.

Evaluates performance by printing the ROC AUC score.

5. Containerization with Docker
Build the Docker Image:

bash
Copy
docker build -t your-dockerhub-username/fraud-detection:latest .
Run the Docker Container:

bash
Copy
docker run -p 5000:5000 your-dockerhub-username/fraud-detection:latest
Adjust port mappings and container configurations as necessary.

6. Kubernetes Deployment
Deploy on Kubernetes:

Apply the deployment configuration:

bash
Copy
kubectl apply -f k8s-deployment.yaml
Access the Service:

If the LoadBalancer external IP is pending, use port forwarding:

bash
Copy
kubectl port-forward service/fraud-detection-service 5000:80
Then open http://localhost:5000 in your browser.

Troubleshooting
External IP Pending:
For local clusters, use port-forwarding or run minikube tunnel (if using minikube).

Module Import Errors:
Verify all required modules (e.g., xgboost, pyspark, delta-spark) are listed in requirements.txt and installed.

Slow Processing:
Adjust the sleep interval in your Kafka producer for testing or monitor system resources.

Connectivity Issues:
Ensure that Kafka, Zookeeper, and your Delta Lake ingestion processes are running properly by checking logs.

Future Enhancements
Additional Models:
Experiment with other ensemble methods like CatBoost or LightGBM.

Advanced Feature Engineering:
Add domain-specific features such as rolling statistics, lag features, and time-of-day indicators.

Monitoring and Alerting:
Integrate monitoring tools like Prometheus and Grafana for production deployments.

CI/CD Integration:
Set up automated testing, building, and deployment using GitHub Actions or another CI/CD platform.

Data Drift and Retraining:
Implement mechanisms to monitor data drift and trigger retraining when necessary.