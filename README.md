# Adaptive Fraud Detection

Adaptive Fraud Detection is an end-to-end fraud detection pipeline that combines deep anomaly detection (via an Anomaly Transformer) with gradient boosting (XGBoost). The system streams transactions in real time using Apache Kafka, stores data in Delta Lake for both batch and streaming analytics, and is containerized with Docker and orchestrated on Kubernetes for scalable deployment.

## Table of Contents
- [Overview](#overview)
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

## Overview

This project implements a real-time fraud detection workflow that:

- **Streams and Stores Data:** Leverages Apache Kafka for real-time data streaming and Delta Lake for batch/streaming ingestion.
- **Anomaly Detection:** Utilizes an **Anomaly Transformer** to compute reconstruction errors from transaction time-series data.
- **Fraud Classification:** Integrates anomaly signals with tabular features to train an **XGBoost** classifier.
- **Containerization & Scalability:** Employs Docker for containerization and Kubernetes for scalable deployment.

## Features

- **End-to-End Pipeline** – Covers data ingestion, anomaly detection, fraud classification, and deployment.
- **Real-Time Analytics** – Continuously ingests and processes transaction data.
- **Robust Classification** – Combines deep anomaly signals with gradient boosting for fraud prediction.
- **Containerization & Orchestration** – Simplifies deployment using Docker and Kubernetes.

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
```

## Requirements

- **Python:** 3.9+
- **Libraries:** PyTorch, XGBoost, Pandas, NumPy, Scikit-learn
- **Big Data Tools:** PySpark & delta-spark for Delta Lake integration
- **Messaging:** Apache Kafka (or equivalent Docker containers)
- **Containerization:** Docker (optionally, Docker Compose)
- **Orchestration:** Kubernetes CLI (kubectl, kubeconfig)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/adaptive-fraud-detection.git
   cd adaptive-fraud-detection
   ```

2. **Install Python Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**

   If required, set environment variables (e.g., SPARK_HOME) in your shell configuration.

## Usage

### 1. Generate Synthetic Dataset

Generate a synthetic dataset if real transaction data is unavailable:

```bash
python data/generate_dataset.py
```

This creates `data/transactions.csv` with approximately 10,000 records.

### 2. Kafka Setup and Data Ingestion

1. **Start Kafka and Zookeeper:**

   If using Docker Compose, run:

   ```bash
   docker-compose up -d
   ```

2. **Verify Running Containers:**

   ```bash
   docker ps
   ```

3. **Run the Kafka Producer:**

   Push synthetic data to Kafka (e.g., topic `fraud_topic`):

   ```bash
   python ingestion/kafka_producer.py
   ```

### 3. Delta Lake Ingestion

Stream data from Kafka into Delta Lake using PySpark:

```bash
python ingestion/delta_ingest.py
```

This script reads messages from Kafka and writes them to a Delta Lake table located at `/tmp/delta/transactions` (adjust the path if needed).

### 4. Model Training and Prediction

Train models and generate predictions by running:

```bash
python models/pipeline.py
```

The pipeline:

- Trains the Anomaly Transformer on time-series data.
- Computes reconstruction errors (anomaly signals).
- Combines these errors with additional features.
- Splits the data into training and validation sets.
- Trains an XGBoost classifier.
- Evaluates performance via ROC AUC score.

### 5. Containerization with Docker

1. **Build the Docker Image:**

   ```bash
   docker build -t your-dockerhub-username/fraud-detection:latest .
   ```

2. **Run the Docker Container:**

   ```bash
   docker run -p 5000:5000 your-dockerhub-username/fraud-detection:latest
   ```

Adjust port mappings and configurations as necessary.

### 6. Kubernetes Deployment

1. **Deploy Using Kubernetes:**

   ```bash
   kubectl apply -f k8s-deployment.yaml
   ```

2. **Access the Service:**

   If the LoadBalancer external IP is pending, use port forwarding:

   ```bash
   kubectl port-forward service/fraud-detection-service 5000:80
   ```

   Then open [http://localhost:5000](http://localhost:5000) in your browser.

## Troubleshooting

- **External IP Pending:**  
  For local clusters, use port forwarding or run `minikube tunnel` if using Minikube.

- **Module Import Errors:**  
  Ensure all required modules (e.g., xgboost, pyspark, delta-spark) are installed.

- **Slow Processing:**  
  Adjust the sleep interval in the Kafka producer or monitor system resources.

- **Connectivity Issues:**  
  Verify that Kafka, Zookeeper, and Delta Lake ingestion processes are running properly by checking logs.

## Future Enhancements

- **Additional Models:**  
  Experiment with other ensemble methods like CatBoost or LightGBM.

- **Advanced Feature Engineering:**  
  Add domain-specific features such as rolling statistics, lag features, and time-of-day indicators.

- **Monitoring and Alerting:**  
  Integrate monitoring tools like Prometheus and Grafana for production deployments.

- **CI/CD Integration:**  
  Set up automated testing, building, and deployment using GitHub Actions or another CI/CD platform.

- **Data Drift and Retraining:**  
  Implement mechanisms to monitor data drift and trigger retraining when necessary.

## License

[Insert your license information here.]

## Contact

For questions or further details, please contact [your-email@example.com].