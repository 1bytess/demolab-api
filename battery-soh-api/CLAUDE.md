# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Battery State of Health (SoH) Prediction API built with FastAPI. The API predicts battery capacity degradation using LSTM deep learning models and is designed for car-to-charger station integration.

## Development Commands

### Running the API

#### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the development server
python app.py
# The API will start on http://0.0.0.0:5003

# Alternative: Run with uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 5003 --reload
```

#### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Run in background
docker-compose up -d

# Stop the service
docker-compose down

# View logs
docker-compose logs -f battery-soh-api
```

#### Manual Docker Commands
```bash
# Build the Docker image
docker build -t battery-soh-api .

# Run the container
docker run -p 5003:5003 --name battery-soh-api battery-soh-api
```

### Production Deployment
```bash
# Run in production mode
uvicorn app:app --host 0.0.0.0 --port 5003 --log-level info
```

## Architecture

### Core Components

- **app.py**: Main FastAPI application with ML inference pipeline
- **models/**: Pre-trained LSTM model and data scalers
  - `lstm.keras`: TensorFlow/Keras LSTM model for capacity prediction
  - `data_scalers.pkl`: Pickled scikit-learn scalers for data normalization

### Key Features

1. **Data Processing Pipeline** (app.py:122-286):
   - Feature extraction from battery discharge cycles
   - Time-series interpolation to fixed 100-point sequences
   - Advanced battery health features (dQ/dV, coulombic efficiency)

2. **ML Inference** (app.py:308-437):
   - Dual-input LSTM model (sequence + scalar features)
   - Real-time capacity prediction with confidence scoring
   - Production-ready error handling and validation

3. **API Endpoints**:
   - `/v3/health`: Health check with model status
   - `/v3/predict`: Main prediction endpoint (POST CSV file)
   - `/`: Root endpoint with API documentation

### Expected Data Format

CSV files must contain these columns:
- `Time`: Time in seconds (float/int)
- `Voltage_measured`: Battery voltage in Volts (float)
- `Current_measured`: Battery current in Amperes (float, negative for discharge)
- `Temperature_measured`: Battery temperature in Celsius (float)

Minimum 10 data points required for reliable prediction.

### Model Loading

The ML model and scalers are loaded on application startup (app.py:288-292). The model expects:
- Sequence input: [batch_size, 100, 3] (voltage, current, temperature)
- Scalar input: [batch_size, 15] (engineered features)

### Error Handling

Comprehensive validation for:
- File format (CSV only)
- Required columns presence
- Data quality (minimum points, no missing values)
- Voltage range validation (0-10V)
- Processing pipeline errors

## File Structure

```
battery-soh-api-2/
├── app.py              # Main FastAPI application
├── requirements.txt    # Python dependencies
├── sample_cycle.csv    # Example input data format
├── Dockerfile          # Docker container configuration
├── docker-compose.yml  # Docker Compose orchestration
├── .dockerignore       # Docker build exclusions
└── models/
    ├── lstm.keras      # Pre-trained LSTM model
    └── data_scalers.pkl # Data normalization scalers
```

## Development Notes

- The API uses TensorFlow 2.15.0 for ML inference
- FastAPI with Pydantic for request/response validation
- Production-ready logging and error handling
- CORS enabled for cross-origin requests
- No test files present in current codebase
- API runs on port 5003 (configurable in app.py:474)

## Docker Configuration

### Dockerfile Features
- Uses `python:3.11-slim` for minimal image size
- Multi-stage optimization with dependency caching
- Non-root user for security
- Built-in health checks
- Only essential system dependencies (gcc, g++)

### Docker Compose Features
- Isolated network for the service
- Health check monitoring
- Automatic restart policy
- Optional volume mount for model updates
- Environment variables for Python optimization

### Docker Commands
```bash
# Check API health
curl http://localhost:5003/v3/health

# Test prediction endpoint
curl -X POST -F "file=@sample_cycle.csv" http://localhost:5003/v3/predict
```