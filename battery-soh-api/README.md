# Battery State of Health (SoH) Prediction API

A professional FastAPI service for predicting battery capacity degradation using LSTM deep learning models. Designed for car-to-charger station integration scenarios.

## Overview

This API processes battery discharge cycle data and predicts remaining capacity using advanced machine learning techniques. It accepts CSV files containing time-series battery data and returns detailed capacity predictions with confidence metrics.

### Key Features

- **Real-time prediction**: Process battery data and get instant capacity predictions
- **LSTM-based models**: Uses deep learning for accurate capacity forecasting
- **Professional API**: Production-ready with comprehensive error handling
- **Car-to-charger integration**: Designed for electric vehicle charging station workflows
- **Data validation**: Robust input validation and quality checks

## Quick Start

### Clone the Repository

```bash
git clone https://github.com/1bytess/demolab-api.git
cd demolab-api/battery-soh-api
```

### Installation

#### Option 1: Local Development

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the API server
python app.py
```

The API will be available at `http://localhost:5003`

#### Option 2: Docker (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run in background
docker-compose up -d
```

#### Option 3: Manual Docker

```bash
# Build the image
docker build -t battery-soh-api .

# Run the container
docker run -p 5003:5003 battery-soh-api
```

## API Endpoints

### Base URL: `http://localhost:5003`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and documentation |
| `/v3/health` | GET | Health check with model status |
| `/v3/predict` | POST | Upload CSV for battery capacity prediction |
| `/docs` | GET | Interactive Swagger documentation |

## Usage Examples with curl

### 1. Health Check

```bash
curl -X GET "http://localhost:5003/v3/health"
```

**Expected Response:**
```json
{
  "status": "healthy",
  "message": "Battery SoH Prediction API is operational",
  "model_loaded": true,
  "model_loaded_at": "2024-01-15T10:30:00.123456",
  "version": "2.0.0"
}
```

### 2. Battery Capacity Prediction

```bash
curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_cycle.csv" \
  "http://localhost:5003/v3/predict"
```

**Expected Response:**
```json
{
  "request_id": "123e4567-e89b-12d3-a456-426614174000",
  "predicted_capacity": 2.1847,
  "normalized_predicted_capacity": 0.8234,
  "confidence_score": 0.892,
  "processing_time_ms": 156.78,
  "data_points_processed": 245,
  "status": "success",
  "timestamp": "2024-01-15T10:31:00.123456"
}
```

### 3. Get API Information

```bash
curl -X GET "http://localhost:5003/"
```

### 4. Interactive Documentation

Open in browser: `http://localhost:5003/docs`

## Input Data Format

### Required CSV Structure

Your CSV file must contain exactly these 4 columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Time` | float/int | Time in seconds | 0, 300, 600, ... |
| `Voltage_measured` | float | Battery voltage in Volts | 4.183, 4.158, 4.134, ... |
| `Current_measured` | float | Battery current in Amperes (negative for discharge) | -4.0, -3.8, -4.2, ... |
| `Temperature_measured` | float | Battery temperature in Celsius | 24.61, 25.16, 25.43, ... |

### Example CSV Content

```csv
Time,Voltage_measured,Current_measured,Temperature_measured
0,4.183,-4.0,24.61
300,4.158,-4.0,25.16
600,4.134,-4.0,25.43
900,4.110,-4.0,25.62
1200,4.087,-4.0,25.76
```

### Data Requirements

- **Minimum data points**: 10 entries required for reliable prediction
- **No missing values**: All columns must contain valid numeric data
- **Voltage range**: 0-10V (validation enforced)
- **File format**: CSV only
- **Discharge data**: Current should be negative for discharge cycles

## Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | string | Unique identifier for tracking |
| `predicted_capacity` | float | Predicted battery capacity in Ah |
| `normalized_predicted_capacity` | float | Prediction on 0-1 scale |
| `confidence_score` | float | Model confidence (0-1, higher is better) |
| `processing_time_ms` | float | Processing time in milliseconds |
| `data_points_processed` | int | Number of input data points |
| `status` | string | Processing status ("success" or error) |
| `timestamp` | string | Response timestamp (ISO format) |

## Advanced Usage

### Testing with Sample Data

```bash
# Use the provided sample file
curl -X POST \
  -F "file=@sample_cycle.csv" \
  "http://localhost:5003/v3/predict"

# Test with your own data
curl -X POST \
  -F "file=@your_battery_data.csv" \
  "http://localhost:5003/v3/predict"
```

### Error Handling

The API provides detailed error messages for common issues:

```bash
# Invalid file format
curl -X POST -F "file=@data.txt" "http://localhost:5003/v3/predict"
# Returns: 400 Bad Request - "Invalid file format. Only CSV files are accepted."

# Missing columns
curl -X POST -F "file=@incomplete.csv" "http://localhost:5003/v3/predict"
# Returns: 400 Bad Request - "Missing required columns: ['Current_measured']"

# Insufficient data
curl -X POST -F "file=@short.csv" "http://localhost:5003/v3/predict"
# Returns: 400 Bad Request - "Insufficient data points. Minimum 10 data points required"
```

### Production Deployment

For production use:

```bash
# Run with production settings
uvicorn app:app --host 0.0.0.0 --port 5003 --log-level info

# Or use Docker in production mode
docker-compose -f docker-compose.yml up -d
```

## Integration Workflow

### Car-to-Charger Station Integration

1. **Data Collection**: Car connects to charging station and extracts battery data
2. **Data Formatting**: Convert battery data to required CSV format
3. **API Request**: POST CSV file to `/v3/predict` endpoint
4. **Processing**: API processes data through ML pipeline
5. **Response**: Receive capacity prediction with confidence metrics

### Example Integration Code

```python
import requests

# Prepare your CSV data
files = {'file': open('battery_data.csv', 'rb')}

# Make prediction request
response = requests.post(
    'http://localhost:5003/v3/predict',
    files=files
)

# Process response
if response.status_code == 200:
    result = response.json()
    print(f"Predicted Capacity: {result['predicted_capacity']} Ah")
    print(f"Confidence: {result['confidence_score']:.1%}")
else:
    print(f"Error: {response.json()['detail']}")
```

## Model Information

- **Architecture**: Dual-input LSTM (time-series + engineered features)
- **Framework**: TensorFlow 2.15.0
- **Input Features**: 15 engineered battery health indicators
- **Sequence Length**: 100 interpolated time points
- **Output**: Normalized capacity prediction (0-1 scale)

## Troubleshooting

### Common Issues

1. **Model not loaded**: Restart the service to reload the ML model
2. **CSV parsing errors**: Verify your CSV format matches the requirements
3. **Port conflicts**: Change the port in `app.py` or Docker configuration
4. **Memory issues**: Ensure sufficient RAM for TensorFlow operations

### Health Check

Always verify the service is healthy before making predictions:

```bash
curl http://localhost:5003/v3/health
```

## Support

- **Documentation**: `/docs` endpoint for interactive API documentation
- **Version**: 2.0.0
- **Contact**: escl@bezrahernowo.com

## License

MIT License