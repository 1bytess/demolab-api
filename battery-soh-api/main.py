from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import uvicorn
import os
import logging
import psutil
import GPUtil
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app with custom docs URL
app = FastAPI(
    title="Battery Capacity Prediction API",
    description="AI-powered battery capacity prediction using BiLSTM, TCN, and LSTM models with impedance support",
    version="3.1.0",
    docs_url="/v3/docs",
    redoc_url="/v3/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class BatteryData(BaseModel):
    voltage: List[float] = Field(..., description="Voltage sequence data (V)", min_length=5, max_length=200)
    current: List[float] = Field(..., description="Current sequence data (A)", min_length=5, max_length=200)    
    temperature: List[float] = Field(..., description="Temperature sequence data (°C)", min_length=5, max_length=200)
    battery_id: str = Field(..., description="Battery identifier from dataset")
    model_type: str = Field(default="lstm", description="Model to use: 'lstm', 'bilstm', or 'tcn'")
    
    # NEW: Optional impedance parameters
    re: Optional[float] = Field(None, description="Electrolyte resistance (Ω) - optional", ge=0.0, le=1.0)
    rct: Optional[float] = Field(None, description="Charge transfer resistance (Ω) - optional", ge=0.0, le=5.0)
    
    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "voltage": [4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5],
                "current": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "temperature": [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
                "battery_id": "B0005",
                "model_type": "lstm",
                "re": 0.056,  # Optional
                "rct": 0.201  # Optional
            }
        }
    }

class PredictionResponse(BaseModel):
    soh_percentage: float = Field(..., description="State of Health as percentage (0-100%)")
    estimated_capacity: float = Field(..., description="Estimated capacity in Ah")
    model_used: str = Field(..., description="Model that generated the prediction")
    confidence_score: Optional[float] = Field(None, description="Prediction confidence (0-1)")
    processing_time: float = Field(..., description="Processing time in seconds")
    battery_info: Dict = Field(..., description="Information about the selected battery")
    impedance_info: Dict = Field(..., description="Information about impedance values used")
    timestamp: str
    
    model_config = {
        "protected_namespaces": ()
    }

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    available_models: List[str]
    available_batteries: Dict[str, List[str]]
    gpu_available: bool
    impedance_support: bool
    system_info: Optional[Dict] = None  # NEW: System information
    timestamp: str

class BatteryListResponse(BaseModel):
    train_batteries: List[str]
    test_batteries: List[str]
    total_batteries: int

# Global variables for models and scalers
models = {}
scalers = {}

# Battery database from your training (NASA dataset)
BATTERY_DATABASE = {
    'train_batteries': ['B0005', 'B0006', 'B0007', 'B0018', 'B0025', 'B0026', 'B0027', 'B0028', 
                       'B0031', 'B0032', 'B0033', 'B0034', 'B0036', 'B0039', 'B0041', 'B0042', 
                       'B0043', 'B0044', 'B0045', 'B0047', 'B0048', 'B0053', 'B0055', 'B0056'],
    'test_batteries': ['B0029', 'B0030', 'B0038', 'B0040', 'B0046', 'B0054']
}

# Battery specifications (NASA dataset nominal capacity)
BATTERY_SPECS = {
    'nominal_capacity': 2.0
}

def load_models_and_scalers():
    """Load trained models and scalers"""
    global models, scalers
    
    try:
        # Debug: List current working directory and available files
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Files in current directory: {os.listdir('.')}")
        
        # Check for models directory
        models_dir = 'models'
        if os.path.exists(models_dir):
            logger.info(f"Models directory found: {models_dir}")
            logger.info(f"Files in models directory: {os.listdir(models_dir)}")
        else:
            logger.warning(f"Models directory not found: {models_dir}")
            models = {}
            scalers = create_default_scalers()
            return
        
        # Load models - try .keras first, then .h5 as fallback
        model_files = {
            'bilstm': ('bilstm_model.keras', 'bilstm_model_best.h5'),
            'tcn': ('tcn_model.keras', 'tcn_model_best.h5'),
            'lstm': ('lstm_model.keras', 'lstm_model_best.h5')
        }
        
        for model_name, (keras_file, h5_file) in model_files.items():
            model_loaded = False
            
            # Try .keras file first
            keras_path = f'models/{keras_file}'
            logger.info(f"Checking for {model_name} model at: {keras_path}")
            if os.path.exists(keras_path):
                try:
                    if model_name == 'tcn':
                        # TCN model needs custom objects
                        try:
                            from tcn import TCN
                            models[model_name] = load_model(keras_path, custom_objects={'TCN': TCN})
                        except ImportError:
                            logger.warning(f"TCN library not available, skipping {model_name} model")
                            continue
                    else:
                        models[model_name] = load_model(keras_path)
                    logger.info(f"Loaded {model_name} model successfully from {keras_path}")
                    model_loaded = True
                except Exception as model_error:
                    logger.warning(f"Failed to load {model_name} from .keras: {str(model_error)}")
            else:
                logger.info(f"File not found: {keras_path}")
            
            # Try .h5 file as fallback
            if not model_loaded:
                h5_path = f'models/{h5_file}'
                logger.info(f"Trying fallback {model_name} model at: {h5_path}")
                if os.path.exists(h5_path):
                    try:
                        if model_name == 'tcn':
                            try:
                                from tcn import TCN
                                models[model_name] = load_model(h5_path, custom_objects={'TCN': TCN})
                            except ImportError:
                                logger.warning(f"TCN library not available, skipping {model_name} model")
                                continue
                        else:
                            models[model_name] = load_model(h5_path)
                        logger.info(f"Loaded {model_name} model successfully from {h5_path}")
                        model_loaded = True
                    except Exception as model_error:
                        logger.error(f"Failed to load {model_name} from .h5: {str(model_error)}")
                else:
                    logger.info(f"File not found: {h5_path}")
            
            if not model_loaded:
                logger.warning(f"Could not load {model_name} model from either .keras or .h5 file")
        
        # Load scalers - try individual model scalers first
        scalers_loaded = False
        
        # Try to load from any individual model scaler file (they should all be the same)
        for model_name in ['lstm', 'bilstm', 'tcn']:
            scaler_path = f'models/{model_name}_model_scalers.pkl'
            logger.info(f"Checking for scalers at: {scaler_path}")
            if os.path.exists(scaler_path):
                try:
                    with open(scaler_path, 'rb') as f:
                        scalers = pickle.load(f)
                    logger.info(f"Loaded scalers from {scaler_path}")
                    logger.info(f"Available scalers: {list(scalers.keys())}")
                    scalers_loaded = True
                    break
                except Exception as scaler_error:
                    logger.warning(f"Failed to load scalers from {scaler_path}: {str(scaler_error)}")
            else:
                logger.info(f"Scaler file not found: {scaler_path}")
        
        # Create default scalers if none were loaded
        if not scalers_loaded:
            logger.warning("Could not load scalers from any source, creating defaults")
            scalers = create_default_scalers()
            
    except Exception as e:
        logger.error(f"Error loading models/scalers: {str(e)}")
        models = {}
        scalers = create_default_scalers()

def get_system_info():
    """Get comprehensive system information including GPU/CPU usage"""
    try:
        system_info = {
            "process_type": "gpu" if len(tf.config.list_physical_devices('GPU')) > 0 else "cpu"
        }
        
        # CPU Information
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            system_info.update({
                "cpu_usage": cpu_percent,
                "cpu_cores": psutil.cpu_count(),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_percent": memory.percent
            })
        except Exception as e:
            logger.warning(f"Failed to get CPU info: {e}")
        
        # GPU Information
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                system_info.update({
                    "gpu_name": gpu.name,
                    "gpu_memory_total": int(gpu.memoryTotal),
                    "gpu_memory_used": int(gpu.memoryUsed),
                    "gpu_memory_free": int(gpu.memoryFree),
                    "gpu_load": round(gpu.load * 100, 1),
                    "gpu_temperature": gpu.temperature
                })
            elif len(tf.config.list_physical_devices('GPU')) > 0:
                # TensorFlow detected GPU but GPUtil failed
                try:
                    # Try to get TensorFlow GPU memory info
                    physical_devices = tf.config.list_physical_devices('GPU')
                    if physical_devices:
                        # Get basic GPU info from TensorFlow
                        system_info.update({
                            "gpu_name": "GPU Detected (TensorFlow)",
                            "gpu_memory_total": "Unknown",
                            "gpu_memory_used": "Unknown", 
                            "gpu_load": "Unknown",
                            "gpu_details": f"{len(physical_devices)} GPU(s) available"
                        })
                except Exception as tf_error:
                    logger.warning(f"Failed to get TensorFlow GPU info: {tf_error}")
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            
        # Process Information
        try:
            current_process = psutil.Process()
            system_info.update({
                "process_memory_mb": round(current_process.memory_info().rss / (1024**2), 2),
                "process_cpu_percent": current_process.cpu_percent()
            })
        except Exception as e:
            logger.warning(f"Failed to get process info: {e}")
            
        return system_info
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return {
            "process_type": "cpu",
            "error": "System info unavailable"
        }
    """Create default scalers for when loading fails"""
    default_scalers = {
        'seq_voltage': MinMaxScaler(),
        'seq_current': MinMaxScaler(),
        'seq_temperature': MinMaxScaler(),
        'additional_features': StandardScaler(),
        'capacity': MinMaxScaler()
    }
    
    # Fit with some dummy data so they work
    dummy_data = np.array([[3.0], [3.5], [4.0], [4.2]])  # Realistic voltage range
    default_scalers['seq_voltage'].fit(dummy_data)
    
    dummy_current = np.array([[0.0], [0.5], [1.0], [2.0]])  # Realistic current range
    default_scalers['seq_current'].fit(dummy_current)
    
    dummy_temp = np.array([[0.0], [25.0], [35.0], [60.0]])  # Realistic temperature range
    default_scalers['seq_temperature'].fit(dummy_temp)
    
    # 7 additional features: duration_hours, voltage_drop, temperature_mean, temperature_range, Re_mapped, Rct_mapped, impedance_time_diff
    dummy_features = np.array([
        [1.0, 0.5, 25.0, 10.0, 0.03, 0.2, 24.0],  # Sample 1
        [2.0, 1.0, 30.0, 15.0, 0.05, 0.3, 48.0],  # Sample 2
        [3.0, 1.5, 35.0, 20.0, 0.08, 0.5, 72.0],  # Sample 3
        [4.0, 2.0, 40.0, 25.0, 0.12, 0.8, 96.0]   # Sample 4
    ])
    default_scalers['additional_features'].fit(dummy_features)
    
    dummy_capacity = np.array([[1.0], [1.5], [2.0], [2.5]])  # Realistic capacity range
    default_scalers['capacity'].fit(dummy_capacity)
    
    logger.info("Created and fitted default scalers for 7 additional features")
    return default_scalers

def get_system_info():
    """Get comprehensive system information including GPU/CPU usage"""
    try:
        system_info = {
            "process_type": "gpu" if len(tf.config.list_physical_devices('GPU')) > 0 else "cpu"
        }
        
        # CPU Information
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            system_info.update({
                "cpu_usage": cpu_percent,
                "cpu_cores": psutil.cpu_count(),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_percent": memory.percent
            })
        except Exception as e:
            logger.warning(f"Failed to get CPU info: {e}")
        
        # GPU Information
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                system_info.update({
                    "gpu_name": gpu.name,
                    "gpu_memory_total": int(gpu.memoryTotal),
                    "gpu_memory_used": int(gpu.memoryUsed),
                    "gpu_memory_free": int(gpu.memoryFree),
                    "gpu_load": round(gpu.load * 100, 1),
                    "gpu_temperature": gpu.temperature
                })
            elif len(tf.config.list_physical_devices('GPU')) > 0:
                # TensorFlow detected GPU but GPUtil failed
                try:
                    # Try to get TensorFlow GPU memory info
                    physical_devices = tf.config.list_physical_devices('GPU')
                    if physical_devices:
                        # Get basic GPU info from TensorFlow
                        system_info.update({
                            "gpu_name": "GPU Detected (TensorFlow)",
                            "gpu_memory_total": "Unknown",
                            "gpu_memory_used": "Unknown", 
                            "gpu_load": "Unknown",
                            "gpu_details": f"{len(physical_devices)} GPU(s) available"
                        })
                except Exception as tf_error:
                    logger.warning(f"Failed to get TensorFlow GPU info: {tf_error}")
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            
        # Process Information
        try:
            current_process = psutil.Process()
            system_info.update({
                "process_memory_mb": round(current_process.memory_info().rss / (1024**2), 2),
                "process_cpu_percent": current_process.cpu_percent()
            })
        except Exception as e:
            logger.warning(f"Failed to get process info: {e}")
            
        return system_info
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return {
            "process_type": "cpu",
            "error": "System info unavailable"
        }

def normalize_sequences(voltage, current, temperature):
    """Normalize input sequences using loaded scalers"""
    try:
        # Check if scalers are properly fitted
        if not hasattr(scalers.get('seq_voltage'), 'scale_'):
            logger.warning("Scalers not fitted, using default normalization")
            
            # Simple min-max normalization with typical battery ranges
            voltage_normalized = np.array([(v - 3.0) / (4.2 - 3.0) for v in voltage])
            current_normalized = np.array([(c - 0.0) / (3.0 - 0.0) for c in current])  
            temperature_normalized = np.array([(t - 0.0) / (60.0 - 0.0) for t in temperature])
            
            return voltage_normalized, current_normalized, temperature_normalized
        
        # Normalize voltage
        voltage_array = np.array(voltage).reshape(-1, 1)
        voltage_normalized = scalers['seq_voltage'].transform(voltage_array).flatten()
        
        # Normalize current
        current_array = np.array(current).reshape(-1, 1)
        current_normalized = scalers['seq_current'].transform(current_array).flatten()
        
        # Normalize temperature
        temperature_array = np.array(temperature).reshape(-1, 1)
        temperature_normalized = scalers['seq_temperature'].transform(temperature_array).flatten()
        
        return voltage_normalized, current_normalized, temperature_normalized
    except Exception as e:
        logger.error(f"Error normalizing sequences: {str(e)}")
        # Fallback to simple normalization
        voltage_normalized = np.array([(v - 3.0) / (4.2 - 3.0) for v in voltage])
        current_normalized = np.array([(c - 0.0) / (3.0 - 0.0) for c in current])  
        temperature_normalized = np.array([(t - 0.0) / (60.0 - 0.0) for t in temperature])
        
        return voltage_normalized, current_normalized, temperature_normalized

def estimate_electrolyte_resistance(voltage_seq, temp_seq, battery_id):
    """Estimate Re based on voltage and temperature characteristics (NASA dataset ranges)"""
    # Based on NASA dataset: Re typical range 0.01-0.15 Ω
    temp_mean = np.mean(temp_seq)
    voltage_std = np.std(voltage_seq)
    
    # Base resistance (typical range from NASA data)
    base_re = 0.056  # Average from dataset
    
    # Temperature effect (resistance increases at lower temps)
    temp_factor = 1.0 + (25.0 - temp_mean) * 0.002
    
    # Voltage variation effect (more variation might indicate aging)
    voltage_factor = 1.0 + voltage_std * 0.1
    
    # Battery aging effect based on battery ID
    battery_factor = 1.0 + (ord(battery_id[-1]) % 5) * 0.01
    
    re_estimated = base_re * temp_factor * voltage_factor * battery_factor
    return max(0.01, min(0.15, re_estimated))  # Clamp to NASA dataset range

def estimate_charge_transfer_resistance(voltage_seq, current_seq, battery_id):
    """Estimate Rct based on voltage and current characteristics (NASA dataset ranges)"""
    # Based on NASA dataset: Rct typical range 0.05-1.0 Ω
    voltage_range = np.max(voltage_seq) - np.min(voltage_seq)
    current_mean = np.mean(current_seq)
    
    # Base resistance (typical range from NASA data)
    base_rct = 0.201  # Average from dataset
    
    # Voltage drop effect (larger drops might indicate higher resistance)
    voltage_effect = 1.0 + (1.0 - voltage_range) * 0.3
    
    # Current effect (higher currents might reveal resistance issues)
    current_effect = 1.0 + current_mean * 0.05
    
    # Battery aging simulation
    battery_factor = 1.0 + (ord(battery_id[-1]) % 7) * 0.03
    
    rct_estimated = base_rct * voltage_effect * current_effect * battery_factor
    return max(0.05, min(1.0, rct_estimated))  # Clamp to NASA dataset range

def estimate_impedance_time_diff(duration_hours, battery_id):
    """Estimate impedance_time_diff based on test duration and battery characteristics"""
    # Simulate time since last impedance measurement (in hours)
    base_time = 24.0  # 24 hours as baseline
    
    # Duration effect (longer tests might indicate more time since last measurement)
    duration_effect = duration_hours * 2.0
    
    # Battery-specific variation
    battery_variation = (ord(battery_id[-1]) % 10) * 5.0
    
    time_diff = base_time + duration_effect + battery_variation
    return max(1.0, min(200.0, time_diff))  # Clamp to reasonable range (1-200 hours)

def prepare_model_inputs(voltage, current, temperature, battery_id, re_input=None, rct_input=None):
    """Prepare inputs for different models - with optional Re/Rct inputs"""
    # Normalize sequences
    voltage_norm, current_norm, temp_norm = normalize_sequences(voltage, current, temperature)
    
    # Ensure all sequences have the same length (pad or truncate if needed)
    target_length = 100  # Based on your actual training data
    
    def pad_or_truncate(seq, target_len):
        if len(seq) > target_len:
            return seq[:target_len]
        elif len(seq) < target_len:
            return np.pad(seq, (0, target_len - len(seq)), mode='constant', constant_values=seq[-1])
        return seq
    
    voltage_norm = pad_or_truncate(voltage_norm, target_length)
    current_norm = pad_or_truncate(current_norm, target_length)
    temp_norm = pad_or_truncate(temp_norm, target_length)
    
    # Prepare sequence input (shape: [1, 100, 3])
    sequence_input = np.array([voltage_norm, current_norm, temp_norm]).T
    sequence_input = sequence_input.reshape(1, target_length, 3)
    
    # Calculate the 7 additional features
    voltage_seq = np.array(voltage)
    current_seq = np.array(current)
    temp_seq = np.array(temperature)
    
    # Original 4 features
    duration_hours = len(voltage_seq) * 0.1  # Estimated duration based on sequence length
    voltage_drop = voltage_seq[0] - voltage_seq[-1] if len(voltage_seq) > 1 else 0.0
    temperature_mean = np.mean(temp_seq)
    temperature_range = np.max(temp_seq) - np.min(temp_seq) if len(temp_seq) > 1 else 0.0
    
    # Use provided Re/Rct or estimate them
    if re_input is not None:
        re_mapped = re_input
        re_source = "user_provided"
    else:
        re_mapped = estimate_electrolyte_resistance(voltage_seq, temp_seq, battery_id)
        re_source = "estimated"
    
    if rct_input is not None:
        rct_mapped = rct_input
        rct_source = "user_provided"
    else:
        rct_mapped = estimate_charge_transfer_resistance(voltage_seq, current_seq, battery_id)
        rct_source = "estimated"
    
    # impedance_time_diff (always estimated)
    impedance_time_diff = estimate_impedance_time_diff(duration_hours, battery_id)
    
    additional_features = np.array([
        duration_hours, voltage_drop, temperature_mean, temperature_range,
        re_mapped, rct_mapped, impedance_time_diff
    ]).reshape(1, -1)
    
    # Normalize additional features (7 features now)
    if 'additional_features' in scalers and hasattr(scalers['additional_features'], 'scale_'):
        try:
            additional_features = scalers['additional_features'].transform(additional_features)
        except Exception as e:
            logger.warning(f"Using fallback normalization for additional features: {str(e)}")
            # Fallback normalization based on typical ranges for all 7 features
            feature_ranges = np.array([5.0, 1.0, 30.0, 20.0, 0.1, 0.5, 100.0]).reshape(1, -1)
            additional_features = additional_features / feature_ranges
    else:
        # Simple normalization for 7 features
        feature_ranges = np.array([5.0, 1.0, 30.0, 20.0, 0.1, 0.5, 100.0]).reshape(1, -1)
        additional_features = additional_features / feature_ranges
    
    # Return impedance info for response
    impedance_info = {
        "re_value": float(re_mapped),
        "re_source": re_source,
        "rct_value": float(rct_mapped),
        "rct_source": rct_source,
        "impedance_time_diff_hours": float(impedance_time_diff)
    }
    
    return sequence_input, additional_features, impedance_info

def calculate_soh(predicted_capacity, battery_id):
    """Calculate State of Health percentage"""
    nominal_capacity = BATTERY_SPECS['nominal_capacity']
    soh_percentage = (predicted_capacity / nominal_capacity) * 100
    # Cap at 100% maximum
    soh_percentage = min(100.0, max(0.0, soh_percentage))
    return soh_percentage

def get_battery_info(battery_id):
    """Get information about the selected battery"""
    if battery_id in BATTERY_DATABASE['train_batteries']:
        dataset_type = "Training"
        index = BATTERY_DATABASE['train_batteries'].index(battery_id) + 1
    elif battery_id in BATTERY_DATABASE['test_batteries']:
        dataset_type = "Test"
        index = BATTERY_DATABASE['test_batteries'].index(battery_id) + 1
    else:
        dataset_type = "Unknown"
        index = 0
    
    return {
        "battery_id": battery_id,
        "dataset_type": dataset_type,
        "index_in_dataset": index,
        "nominal_capacity_ah": BATTERY_SPECS['nominal_capacity']
    }

@app.get("/v3/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint with system monitoring"""
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    available_models = [name for name in ['lstm', 'bilstm', 'tcn'] if name in models]
    
    # Get comprehensive system information
    system_info = get_system_info()
    
    return HealthResponse(
        status="healthy",
        models_loaded={name: name in models for name in ['bilstm', 'tcn', 'lstm']},
        available_models=available_models,
        available_batteries={
            "train_batteries": BATTERY_DATABASE['train_batteries'],
            "test_batteries": BATTERY_DATABASE['test_batteries']
        },
        gpu_available=gpu_available,
        impedance_support=True,
        system_info=system_info,
        timestamp=datetime.now().isoformat()
    )

@app.get("/v3/batteries", response_model=BatteryListResponse)
async def get_batteries():
    """Get list of available batteries"""
    return BatteryListResponse(
        train_batteries=BATTERY_DATABASE['train_batteries'],
        test_batteries=BATTERY_DATABASE['test_batteries'],
        total_batteries=len(BATTERY_DATABASE['train_batteries']) + len(BATTERY_DATABASE['test_batteries'])
    )

@app.post("/v3/predict", response_model=PredictionResponse)
async def predict_soh(data: BatteryData):
    """Predict battery State of Health using selected model with optional impedance inputs"""
    start_time = datetime.now()
    
    try:
        # Validate input data
        if len(data.voltage) != len(data.current) or len(data.voltage) != len(data.temperature):
            raise HTTPException(
                status_code=400, 
                detail="Voltage, current, and temperature sequences must have the same length"
            )
        
        if len(data.voltage) < 5:
            raise HTTPException(
                status_code=400,
                detail="Sequences must have at least 5 data points"
            )
        
        # Validate battery ID
        all_batteries = BATTERY_DATABASE['train_batteries'] + BATTERY_DATABASE['test_batteries']
        if data.battery_id not in all_batteries:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid battery_id. Must be one of: {all_batteries}"
            )
        
        # Validate model type
        if data.model_type not in ['lstm', 'bilstm', 'tcn']:
            raise HTTPException(
                status_code=400,
                detail="model_type must be one of: 'lstm', 'bilstm', 'tcn'"
            )
        
        # Check if requested model is available
        if data.model_type not in models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{data.model_type}' is not available. Available models: {list(models.keys())}"
            )
        
        # Prepare inputs - all models use both sequence and additional features
        # NEW: Pass optional Re/Rct inputs
        sequence_input, additional_features, impedance_info = prepare_model_inputs(
            data.voltage, data.current, data.temperature, data.battery_id,
            re_input=data.re, rct_input=data.rct
        )
        
        # Make prediction with selected model
        try:
            # All models (LSTM, BiLSTM, TCN) have the same input format: [sequence_input, additional_features]
            pred_norm = models[data.model_type].predict([sequence_input, additional_features], verbose=0)
            
            # Denormalize prediction
            predicted_capacity = scalers['capacity'].inverse_transform(pred_norm.reshape(-1, 1))[0, 0]
            
            # Calculate SOH percentage
            soh_percentage = calculate_soh(predicted_capacity, data.battery_id)
            
            # Calculate confidence (improved heuristic considering impedance inputs)
            voltage_std = np.std(data.voltage)
            temp_range = max(data.temperature) - min(data.temperature)
            base_confidence = min(0.95, max(0.5, 1.0 - (voltage_std * 0.1) - (temp_range * 0.01)))
            
            # Boost confidence if user provided actual impedance measurements
            impedance_bonus = 0.0
            if data.re is not None:
                impedance_bonus += 0.05
            if data.rct is not None:
                impedance_bonus += 0.05
            
            confidence = min(0.98, base_confidence + impedance_bonus)
            
        except Exception as model_error:
            logger.error(f"{data.model_type} prediction error: {str(model_error)}")
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(model_error)}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        battery_info = get_battery_info(data.battery_id)
        
        return PredictionResponse(
            soh_percentage=float(soh_percentage),
            estimated_capacity=float(predicted_capacity),
            model_used=data.model_type,
            confidence_score=float(confidence),
            processing_time=processing_time,
            battery_info=battery_info,
            impedance_info=impedance_info,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load models and scalers on startup"""
    logger.info("Starting up Battery Capacity Prediction API v3.1 with impedance support")
    load_models_and_scalers()
    
    # Log system information
    system_info = get_system_info()
    logger.info(f"System Info: {system_info}")
    
    # Log GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPU devices available: {len(gpus)}")
        for gpu in gpus:
            logger.info(f"GPU: {gpu}")
        
        # Try to get GPU memory info
        try:
            import GPUtil
            gpu_list = GPUtil.getGPUs()
            if gpu_list:
                for i, gpu in enumerate(gpu_list):
                    logger.info(f"GPU {i}: {gpu.name} | Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB | Load: {gpu.load*100:.1f}%")
        except Exception as e:
            logger.warning(f"Could not get detailed GPU info: {e}")
    else:
        logger.info("No GPU devices found, using CPU")
    
    # Log CPU information
    try:
        import psutil
        logger.info(f"CPU: {psutil.cpu_count()} cores | Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    except Exception as e:
        logger.warning(f"Could not get CPU info: {e}")

@app.get("/v3/")
async def root():
    """Root endpoint"""
    return {
        "message": "Battery State of Health (SOH) Prediction API v3.1",
        "description": "AI-powered battery health estimation using LSTM, BiLSTM, and TCN models with impedance support",
        "endpoints": {
            "health": "/v3/health",
            "batteries": "/v3/batteries", 
            "predict": "/v3/predict",
            "documentation": "/v3/docs"
        },
        "models": ["lstm", "bilstm", "tcn"],
        "features": {
            "impedance_support": True,
            "optional_re_rct_inputs": True,
            "automatic_estimation": True,
            "system_monitoring": True
        },
        "total_batteries": len(BATTERY_DATABASE['train_batteries']) + len(BATTERY_DATABASE['test_batteries'])
    }

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=5003, 
        reload=True,
        log_level="info"
    )