# Battery State of Health (SoH) Prediction API

**Version: 2.0.0**

This project provides a production-ready API for predicting the State of Health (SoH) of a battery based on its discharge cycle data. The API uses a trained LSTM model and is containerized with Docker for easy deployment and scaling.

## Features

* **SoH Prediction:** Accepts a CSV file containing a single discharge cycle's time-series data (`Time`, `Voltage_measured`, `Current_measured`, `Temperature_measured`) and returns the predicted SoH in Ampere-hours (Ah).
* **Enhanced Health Check:** Provides a `/v3/health` endpoint that reports API status, uptime, and resource utilization (CPU, Memory, and GPU if available).
* **GPU Accelerated:** Built on an NVIDIA CUDA base image to leverage GPU for faster model inference. Falls back gracefully to CPU if no GPU is detected.
* **Asynchronous by Design:** Built with FastAPI to handle multiple concurrent requests efficiently.
* **Containerized:** Includes `Dockerfile` and `docker-compose.yml` for easy and reproducible deployment.

## Prerequisites

* **Docker:** [Install Docker](https://docs.docker.com/get-docker/)
* **Docker Compose:** (Usually included with Docker Desktop)
* **NVIDIA Drivers:** (For GPU support) The host machine must have the appropriate NVIDIA drivers installed.
* **NVIDIA Container Toolkit:** (For GPU support) [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## How to Run

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/1bytess/demolab-api
    cd demolab-api/battery-soh
    ```

2.  **Place Model Files**
    Ensure your trained model and scaler files are placed in a `models` directory in the project root:
    ```
    .
    ├── requirements.txt
    ├── test_cycle.csv
    ├── models/
    │   ├── lstm_model.keras
    │   └── lstm_model_scalers.pkl
    ├── main.py
    ├── Dockerfile
    └── docker-compose.yml
    ```

3.  **Build and Run with Docker Compose**
    From the project root directory, run the following command. This will build the Docker image and start the API service.
    ```bash
    docker-compose up --build
    ```
    The API will be available at `http://localhost:5003`.

## API Endpoints

### 1. Health Check

* **Endpoint:** `GET /v3/health`
* **Description:** Checks the operational status of the API, including uptime and resource usage.
* **Example Request:**
    ```bash
    curl http://localhost:5003/v3/health
    ```
* **Example Response (with GPU):**
    ```json
    {
      "status": "UP",
      "uptime": "0:01:45",
      "resources": {
        "cpu_usage_percent": 15.2,
        "memory_usage_percent": 55.1,
        "gpu_usage_percent": 5,
        "gpu_memory_usage_percent": 20.12,
        "compute_device": "GPU"
      }
    }
    ```

### 2. Predict State of Health

* **Endpoint:** `POST /v3/predict`
* **Description:** Predicts the SoH from an uploaded CSV file containing a single discharge cycle.
* **Request Body:** `multipart/form-data` with a single field `file` containing the CSV data.
* **Example Request:**
    ```bash
    # Ensure you have a test_cycle.csv file in your directory
    curl -X POST -F "file=@test_cycle.csv" http://localhost:5003/v3/predict
    ```
* **Example Response:**
    ```json
    {
      "battery_id": "test_cycle",
      "predicted_soh_Ah": 1.1814
    }
    ```
* **Note on CSV Format:** The uploaded CSV file must contain the following columns: `Time`, `Voltage_measured`, `Current_measured`, `Temperature_measured`.