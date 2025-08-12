from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from io import StringIO
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import logging
from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import uuid
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with professional configuration
app = FastAPI(
    title="Battery State of Health (SoH) Prediction API",
    description="""
    Professional API for predicting battery capacity degradation using LSTM deep learning models.
    
    **Features:**
    - Real-time battery capacity prediction from discharge cycle data
    - Support for 4-feature CSV input: Time, Voltage, Current, Temperature
    - Advanced feature engineering and interpolation
    - LSTM-based deep learning inference
    - Production-ready health monitoring
    
    **Use Case:** Car-to-Charger Station Integration
    - Cars connect to charger stations
    - Battery data is extracted and converted to CSV format
    - CSV is sent to this API for SoH analysis
    - Server processes data and returns capacity predictions
    """,
    version="2.0.0",
    contact={
        "name": "Battery SoH API Support",
        "email": "escl@bezrahernowo.com"
    },
    license_info={
        "name": "MIT",
    },
)

# Add CORS middleware for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and scalers
model = None
scalers = None
model_loaded_at = None

# Base paths
BASE_PATH = Path(__file__).parent
MODEL_PATH = BASE_PATH / "models" / "lstm.keras"
SCALERS_PATH = BASE_PATH / "models" / "data_scalers.pkl"

# Response Models
class HealthStatus(BaseModel):
    status: str = Field(..., description="API health status")
    message: str = Field(..., description="Health check message")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    model_loaded_at: Optional[str] = Field(None, description="Timestamp when model was loaded")
    version: str = Field(..., description="API version")

class PredictionRequest(BaseModel):
    """Model for prediction request metadata (not used directly but for documentation)"""
    pass

class PredictionResponse(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    predicted_capacity: float = Field(..., description="Predicted battery capacity in Ah")
    normalized_predicted_capacity: float = Field(..., description="Normalized prediction (0-1 scale)")
    confidence_score: float = Field(..., description="Model confidence score (0-1)")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    data_points_processed: int = Field(..., description="Number of data points in input")
    status: str = Field(..., description="Processing status")
    timestamp: str = Field(..., description="Response timestamp")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    request_id: str = Field(..., description="Request identifier")
    timestamp: str = Field(..., description="Error timestamp")

def load_model_and_scalers():
    """Load the trained model and scalers on startup"""
    global model, scalers, model_loaded_at
    
    try:
        # Load the trained LSTM model
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        
        # Load the data scalers
        with open(SCALERS_PATH, 'rb') as f:
            scalers = pickle.load(f)
        logger.info(f"Scalers loaded successfully from {SCALERS_PATH}")
        
        model_loaded_at = datetime.utcnow().isoformat()
        
    except Exception as e:
        logger.error(f"Error loading model or scalers: {e}")
        raise

def extract_cycle_features(cycle_df: pd.DataFrame) -> Dict[str, float]:
    """Extract key features from a single cycle (without coulomb counting)"""
    if cycle_df is None or len(cycle_df) == 0:
        return {}

    features = {}

    # Basic voltage statistics
    features['voltage_start'] = cycle_df['Voltage_measured'].iloc[0]
    features['voltage_end'] = cycle_df['Voltage_measured'].iloc[-1]
    features['voltage_mean'] = cycle_df['Voltage_measured'].mean()
    features['voltage_min'] = cycle_df['Voltage_measured'].min()
    features['voltage_drop'] = features['voltage_start'] - features['voltage_end']

    # Current features
    features['current_mean'] = cycle_df['Current_measured'].mean()
    features['current_std'] = cycle_df['Current_measured'].std()

    # Temperature features
    features['temperature_mean'] = cycle_df['Temperature_measured'].mean()
    features['temperature_max'] = cycle_df['Temperature_measured'].max()
    features['temperature_range'] = (cycle_df['Temperature_measured'].max() -
                                     cycle_df['Temperature_measured'].min())

    # Time features
    features['duration_seconds'] = cycle_df['Time'].max()
    features['duration_hours'] = features['duration_seconds'] / 3600

    # Energy delivered (Wh) - approximate without coulomb counting
    if len(cycle_df) > 1:
        power = np.abs(cycle_df['Voltage_measured'] * cycle_df['Current_measured'])
        time_hours = cycle_df['Time'] / 3600
        features['energy_wh'] = np.trapz(power, time_hours)
    else:
        features['energy_wh'] = 0

    # Advanced features for battery health assessment
    current = cycle_df['Current_measured'].values
    time_seconds = cycle_df['Time'].values
    voltage = cycle_df['Voltage_measured'].values

    # dQ/dV calculation (simplified)
    if len(cycle_df) > 2:
        time_diff = np.diff(time_seconds, prepend=0)
        capacity_cumulative = np.cumsum(-current * time_diff) / 3600
        dq_dv = np.gradient(capacity_cumulative, voltage)
        features['dq_dv_max'] = np.max(np.abs(dq_dv))
    else:
        features['dq_dv_max'] = np.nan

    # Coulombic efficiency for discharge cycles
    discharge_mask = current < 0
    if discharge_mask.any():
        discharge_capacity = features['duration_hours'] * np.mean(-current[discharge_mask])
        total_capacity = np.trapz(-current, time_seconds) / 3600
        features['coulombic_efficiency'] = discharge_capacity / total_capacity if total_capacity > 0 else np.nan
    else:
        features['coulombic_efficiency'] = np.nan

    # Time to voltage threshold (2.7V)
    voltage_threshold = 2.7
    features['time_to_2_7v'] = (cycle_df[cycle_df['Voltage_measured'] < voltage_threshold]['Time'].min()
                               if any(cycle_df['Voltage_measured'] < voltage_threshold) else cycle_df['Time'].max())

    return features

def interpolate_discharge_profile(cycle_df: pd.DataFrame,
                                 n_points: int = 100,
                                 voltage_threshold: float = 2.7) -> Dict[str, np.ndarray]:
    """Interpolate discharge profile to fixed number of points using normalized time"""
    if cycle_df is None or len(cycle_df) < 10:
        return None

    # Sort by time
    cycle_df = cycle_df.sort_values('Time').reset_index(drop=True)

    # Apply voltage threshold
    below_threshold = cycle_df['Voltage_measured'] < voltage_threshold
    if below_threshold.any():
        cutoff_idx = below_threshold.idxmax()
        cycle_df = cycle_df.iloc[:cutoff_idx + 1]

    if len(cycle_df) < 10:
        return None

    # Normalize time to [0, 1]
    time_normalized = (cycle_df['Time'] - cycle_df['Time'].min()) / (cycle_df['Time'].max() - cycle_df['Time'].min())

    # Create interpolation points
    time_interp = np.linspace(0, 1, n_points)

    # Interpolate each variable
    interpolated_data = {}

    try:
        # Voltage interpolation
        f_voltage = interp1d(time_normalized, cycle_df['Voltage_measured'], kind='linear', fill_value='extrapolate')
        interpolated_data['voltage'] = f_voltage(time_interp)

        # Current interpolation
        f_current = interp1d(time_normalized, cycle_df['Current_measured'], kind='linear', fill_value='extrapolate')
        interpolated_data['current'] = f_current(time_interp)

        # Temperature interpolation
        f_temp = interp1d(time_normalized, cycle_df['Temperature_measured'], kind='linear', fill_value='extrapolate')
        interpolated_data['temperature'] = f_temp(time_interp)
        
    except Exception as e:
        logger.error(f"Interpolation failed: {e}")
        return None

    return interpolated_data

def process_cycle_data(cycle_df: pd.DataFrame) -> tuple:
    """Process cycle data through the complete pipeline"""
    # Extract scalar features
    scalar_features = extract_cycle_features(cycle_df)
    
    # Get interpolated sequences
    interpolated = interpolate_discharge_profile(cycle_df)
    
    if interpolated is None:
        raise ValueError("Failed to interpolate discharge profile - insufficient data points")
    
    # Expected feature names (from training)
    expected_scalar_features = [
        'voltage_start', 'voltage_end', 'voltage_mean', 'voltage_min', 'voltage_drop',
        'current_mean', 'current_std', 'temperature_mean', 'temperature_max', 
        'temperature_range', 'duration_seconds', 'duration_hours', 'dq_dv_max', 
        'coulombic_efficiency', 'time_to_2_7v'
    ]
    
    # Create feature vector in the correct order, filling missing values with 0
    feature_vector = []
    for feat_name in expected_scalar_features:
        if feat_name in scalar_features and not pd.isna(scalar_features[feat_name]):
            feature_vector.append(scalar_features[feat_name])
        else:
            feature_vector.append(0.0)
    
    # Prepare sequence data (voltage, current, temperature)
    sequence_data = np.stack([
        interpolated['voltage'],
        interpolated['current'],
        interpolated['temperature']
    ], axis=1)
    
    # Add batch dimension
    sequence_input = sequence_data.reshape(1, 100, 3)
    scalar_input = np.array(feature_vector).reshape(1, -1)
    
    # Normalize using training scalers
    # Normalize sequences
    for i, seq_type in enumerate(['voltage', 'current', 'temperature']):
        scaler_key = f'seq_{seq_type}'
        if scaler_key in scalers:
            sequence_input[0, :, i] = scalers[scaler_key].transform(
                sequence_input[0, :, i].reshape(-1, 1)
            ).flatten()
    
    # Normalize scalar features
    if 'additional_features' in scalers:
        scalar_input = scalers['additional_features'].transform(scalar_input)
    
    return sequence_input, scalar_input

@app.on_event("startup")
async def startup_event():
    """Load model and scalers when the API starts"""
    logger.info("Starting Battery SoH Prediction API v2.0.0")
    load_model_and_scalers()

# API v3 Router
router = FastAPI()

@router.get("/health", response_model=HealthStatus)
async def health_check():
    """Professional health check endpoint for production monitoring"""
    return HealthStatus(
        status="healthy" if model is not None and scalers is not None else "unhealthy",
        message="Battery SoH Prediction API is operational" if model is not None else "Model not loaded - service unavailable",
        model_loaded=model is not None and scalers is not None,
        model_loaded_at=model_loaded_at,
        version="2.0.0"
    )

@router.post("/predict", response_model=PredictionResponse)
async def predict_capacity(file: UploadFile = File(...)):
    """
    Professional battery capacity prediction endpoint for car-to-charger integration
    
    **Process Flow:**
    1. Car connects to charger station
    2. Battery discharge data is extracted
    3. Data is formatted as CSV with required columns
    4. CSV is uploaded to this endpoint
    5. API processes data through ML pipeline
    6. Returns capacity prediction with confidence metrics
    
    **Required CSV Format:**
    - `Time`: Time in seconds (float/int)
    - `Voltage_measured`: Battery voltage in Volts (float)
    - `Current_measured`: Battery current in Amperes (float, negative for discharge)
    - `Temperature_measured`: Battery temperature in Celsius (float)
    
    **Minimum Requirements:**
    - At least 10 data points
    - No missing values in required columns
    - Valid CSV format
    """
    
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    try:
        if model is None or scalers is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                detail="ML model not loaded - service temporarily unavailable"
            )
        
        # Validate file format
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Invalid file format. Only CSV files are accepted."
            )
        
        # Read CSV data
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data)
        
        # Validate required columns
        required_columns = ['Time', 'Voltage_measured', 'Current_measured', 'Temperature_measured']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"Missing required columns: {missing_columns}. Required: {required_columns}"
            )
        
        # Validate data quality
        if len(df) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Insufficient data points. Minimum 10 data points required for reliable prediction."
            )
        
        # Check for missing values
        if df[required_columns].isnull().any().any():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="CSV contains missing values in required columns. All values must be present."
            )
        
        # Validate data ranges
        if df['Voltage_measured'].min() < 0 or df['Voltage_measured'].max() > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Voltage values out of expected range (0-10V). Please verify data quality."
            )
        
        # Process the data
        sequence_input, scalar_input = process_cycle_data(df)
        
        # Make prediction
        normalized_prediction = model.predict([sequence_input, scalar_input], verbose=0)[0][0]
        
        # Denormalize prediction
        if 'target' in scalers:
            actual_capacity = scalers['target'].inverse_transform([[normalized_prediction]])[0][0]
        else:
            actual_capacity = normalized_prediction
        
        # Calculate confidence score based on prediction characteristics
        confidence_score = min(0.95, max(0.60, 1.0 - abs(normalized_prediction - 0.5) * 0.8))
        
        # Calculate processing time
        processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return PredictionResponse(
            request_id=request_id,
            predicted_capacity=round(float(actual_capacity), 4),
            normalized_predicted_capacity=round(float(normalized_prediction), 4),
            confidence_score=round(confidence_score, 3),
            processing_time_ms=round(processing_time_ms, 2),
            data_points_processed=len(df),
            status="success",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Empty CSV file. Please provide valid battery discharge data."
        )
    except pd.errors.ParserError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"CSV parsing error: {str(e)}. Please verify file format."
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Data processing error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error for request {request_id}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Internal server error during prediction. Request ID: {request_id}"
        )

# Mount v3 router
app.mount("/v3", router)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Battery State of Health (SoH) Prediction API",
        "version": "2.0.0",
        "description": "Professional API for car-to-charger battery analysis",
        "endpoints": {
            "/v3/health": "GET - Professional health check with detailed status",
            "/v3/predict": "POST - Upload CSV for battery capacity prediction",
            "/docs": "GET - Interactive API documentation"
        },
        "integration": {
            "use_case": "Car-to-Charger Station Integration",
            "flow": [
                "1. Car connects to charger station",
                "2. Extract battery data to CSV format",
                "3. POST CSV to /v3/predict endpoint",
                "4. Receive capacity prediction with confidence metrics"
            ]
        },
        "support": {
            "contact": "support@battery-soh.com",
            "documentation": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=5003,
        log_level="info",
        access_log=True
    )