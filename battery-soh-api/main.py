# main.py
# Title: Battery State of Health (SoH) Prediction API

# --- Core Libraries ---
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d
import pickle
import io
from pathlib import Path
import time
from datetime import timedelta
import psutil
import pynvml

# --- CHANGE: Improved GPU monitoring check ---
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
except Exception:
    GPU_AVAILABLE = False


# --- API Initialization ---
app = FastAPI(
    title="Battery SoH Prediction API",
    description="An API to predict Battery State of Health from discharge cycle data.",
    version="2.0.0"
)

# --- Store application start time for uptime calculation ---
start_time = time.time()

# --- Load Model and Scalers ---
MODEL_PATH = Path("models/lstm_model.keras")
SCALERS_PATH = Path("models/lstm_model_scalers.pkl")

lstm_model = None
seq_voltage_scaler = None
seq_current_scaler = None
seq_temp_scaler = None
capacity_scaler = None
additional_features_scaler = None


@app.on_event("startup")
def load_artifacts():
    """Load the trained model and scalers into memory."""
    global lstm_model, seq_voltage_scaler, seq_current_scaler, seq_temp_scaler, capacity_scaler, additional_features_scaler
    
    if not MODEL_PATH.exists() or not SCALERS_PATH.exists():
        raise RuntimeError(f"Model or scalers not found. Checked for {MODEL_PATH} and {SCALERS_PATH}.")

    try:
        lstm_model = tf.keras.models.load_model(MODEL_PATH)
        
        with open(SCALERS_PATH, 'rb') as f:
            scalers_obj = pickle.load(f)

        if not isinstance(scalers_obj, dict):
            raise TypeError(f"Scaler file must contain a dictionary. Got {type(scalers_obj)} instead.")

        required_keys = ['seq_voltage', 'seq_current', 'seq_temperature', 'capacity', 'additional_features']
        if not all(key in scalers_obj for key in required_keys):
            raise ValueError(f"Scaler dict is missing required keys. Expected {required_keys}, but found: {list(scalers_obj.keys())}")

        seq_voltage_scaler = scalers_obj['seq_voltage']
        seq_current_scaler = scalers_obj['seq_current']
        seq_temp_scaler = scalers_obj['seq_temperature']
        capacity_scaler = scalers_obj['capacity']
        additional_features_scaler = scalers_obj['additional_features']
        
        print("Model and all scalers loaded successfully.")

    except Exception as e:
        print(f"Error loading artifacts: {e}")
        raise RuntimeError(f"Could not load model or scalers: {e}")

# --- Data Processing Function ---
def process_cycle_data(df: pd.DataFrame, v_scaler, c_scaler, t_scaler, num_points=100):
    """
    Processes a single discharge cycle DataFrame by interpolating and scaling the data.
    """
    features = ['Voltage_measured', 'Current_measured', 'Temperature_measured']
    
    if not all(col in df.columns for col in ['Time'] + features):
        raise ValueError(f"CSV must contain the following columns: Time, {', '.join(features)}")

    cycle_data = df[['Time'] + features].copy()

    time_delta = cycle_data['Time'].max() - cycle_data['Time'].min()
    if time_delta == 0:
        time_normalized = np.zeros_like(cycle_data['Time'])
    else:
        time_normalized = (cycle_data['Time'] - cycle_data['Time'].min()) / time_delta
    
    interp_funcs = {
        feat: interp1d(time_normalized, cycle_data[feat], kind='linear', fill_value="extrapolate")
        for feat in features
    }

    new_time = np.linspace(0, 1, num_points)
    interpolated_data = np.array([interp_funcs[feat](new_time) for feat in features]).T
    
    scaled_voltage = v_scaler.transform(interpolated_data[:, 0].reshape(-1, 1))
    scaled_current = c_scaler.transform(interpolated_data[:, 1].reshape(-1, 1))
    scaled_temp = t_scaler.transform(interpolated_data[:, 2].reshape(-1, 1))

    scaled_sequence = np.hstack([scaled_voltage, scaled_current, scaled_temp])
    
    return scaled_sequence.reshape(1, num_points, len(features))

# --- API Endpoints ---

@app.get("/v3/health")
async def health_check():
    """
    Enhanced health check endpoint to provide system status,
    resource utilization, and uptime.
    """
    uptime_seconds = time.time() - start_time
    uptime_formatted = str(timedelta(seconds=int(uptime_seconds)))

    cpu_usage = psutil.cpu_percent(interval=None)
    memory_info = psutil.virtual_memory()

    resources = {
        "cpu_usage_percent": cpu_usage,
        "memory_usage_percent": memory_info.percent
    }

    # --- CHANGE: Add GPU utilization if available ---
    if GPU_AVAILABLE:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        resources["gpu_usage_percent"] = gpu_util.gpu
        resources["gpu_memory_usage_percent"] = round(100 * mem_info.used / mem_info.total, 2)
        resources["compute_device"] = "GPU"
    else:
        resources["compute_device"] = "CPU"

    status = "UP"
    if not all([lstm_model, seq_voltage_scaler, seq_current_scaler, seq_temp_scaler, capacity_scaler, additional_features_scaler]):
        status = "DEGRADED"
    elif cpu_usage > 95 or memory_info.percent > 95:
        status = "DEGRADED"

    return JSONResponse(content={
        "status": status,
        "uptime": uptime_formatted,
        "resources": resources
    })


@app.post("/v3/predict")
async def predict_soh(file: UploadFile = File(...)):
    """
    Accepts a CSV file of a single discharge cycle, processes it,
    and returns the predicted State of Health (SoH).
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
    
    if not all([lstm_model, seq_voltage_scaler, seq_current_scaler, seq_temp_scaler, capacity_scaler, additional_features_scaler]):
         raise HTTPException(status_code=503, detail="Model artifacts not loaded. API is not ready.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        battery_id = Path(file.filename).stem

        processed_data = process_cycle_data(df, seq_voltage_scaler, seq_current_scaler, seq_temp_scaler)

        num_additional_features = additional_features_scaler.n_features_in_
        dummy_additional_features = np.zeros((1, num_additional_features))
        scaled_dummy_features = additional_features_scaler.transform(dummy_additional_features)

        scaled_prediction = lstm_model.predict([processed_data, scaled_dummy_features])
        
        predicted_soh = capacity_scaler.inverse_transform(scaled_prediction)[0][0]

        return JSONResponse(content={
            "battery_id": battery_id,
            "predicted_soh_Ah": round(float(predicted_soh), 4)
        })

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

# --- Main execution block to run the API ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5003)
