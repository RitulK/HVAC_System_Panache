import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader

import sys

# --- Fix PosixPath compatibility issue before any other imports ---
import pathlib
import platform

# Monkey patch to handle PosixPath on Windows
if platform.system() == 'Windows':
    # Store original PosixPath
    original_posix_path = pathlib.PosixPath
    
    # Replace PosixPath with WindowsPath for this session
    pathlib.PosixPath = pathlib.WindowsPath
    
    # Also patch it in the pathlib module's namespace
    import sys
    sys.modules['pathlib'].PosixPath = pathlib.WindowsPath

# --- Patch to allow loading FastAI/TSai model picked in Jupyter/Colab ---

# Set matplotlib to use non-interactive backend before any imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

if "IPython.core.display" not in sys.modules:
    import types
    
    # Create comprehensive dummy IPython module
    ipython_module = types.ModuleType("IPython")
    def dummy_get_ipython():
        return None
    ipython_module.get_ipython = dummy_get_ipython
    ipython_module.version_info = (8, 25, 0)  # Mock version info
    sys.modules["IPython"] = ipython_module
    
    # Create dummy IPython.core module
    sys.modules["IPython.core"] = types.ModuleType("core")
    sys.modules["IPython.core.display"] = types.ModuleType("display")
    
    # Dummy DisplayHandle
    class DummyDisplayHandle:
        def __init__(self, *a, **kw): pass
        def update(self, *a, **kw): pass
        def display(self, *a, **kw): pass
        def __repr__(self): return "<DummyDisplayHandle>"
    sys.modules["IPython.core.display"].DisplayHandle = DummyDisplayHandle
else:
    from IPython.core.display import DisplayHandle


# ----- 1. Load model, encoders, and scaler -----
from tsai.all import *
import torch.nn as nn

# Custom regression head class must be defined for pickle
class PatchTSTRegressionHead(nn.Module):
    def __init__(self, base_model, embedding_size, output_dim):
        super().__init__()
        self.base = base_model
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.head = nn.Linear(embedding_size, output_dim)
    def forward(self, x):
        x = self.base(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.head(x)
        return x

# Load model
learn_inf = load_learner('patchtst_occupancy_cooling_model.pkl')

# Load encoders/scaler
with open("label_encoders_uptd.pkl", "rb") as f:
    encoders = pickle.load(f)
with open("scaler_uptd.pkl", "rb") as f:
    scaler = pickle.load(f)

# ----- 2. Prepare the inference data -----

# Update with your actual path
df = pd.read_csv("hvac_with_weather.csv")

# Parse timestamp and sort
df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
df = df.sort_values("timestamp").reset_index(drop=True)
df.set_index("timestamp", inplace=True)

# List of categorical columns
categorical_cols = [
    "zone_id", "zone_type", "zone_function", "weather_condition",
    "zone_orientation", "solar_presence", "is_weekend",
    "occupancy_level", "zoning_mode", "merge_key_weather"
]

# Apply label encoders (assumes all classes seen during training)
for col in categorical_cols:
    df[col] = df[col].astype(str)
    le = encoders[col]
    # Handle unseen values gracefully
    df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# Use features that match the original model (46 features, excluding actual_cooling_load_kWh)
model_features = [
    'hour', 'day_of_week', 'zone_id', 'zone_type', 'zone_capacity', 'zone_area_sq_m', 
    'zone_orientation', 'zone_function', 'occupancy_count', 'occupancy_pct', 
    'occupancy_t_minus_1h', 'occupancy_t_minus_2h', 'departing_flights_next_2h', 
    'arriving_flights_next_2h', 'baseline_cooling_load', 'load_t_minus_1h', 
    'standard_setpoint', 'adjusted_setpoint', 'zoning_mode', 'occupancy_t-1', 
    'cooling_load_t-1', 'rolling_avg_occupancy_3h', 'rolling_max_occupancy_6h', 
    'rolling_std_cooling_3h', 'occupancy_delta', 'occupancy_rate', 'hour_sin', 
    'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'occupancy_level', 
    'recommended_setpoint', 'is_weekend', 'sin_hour', 'cos_hour', 'sin_day_of_week', 
    'cos_day_of_week', 'solar_presence', 'external_temperature', 'relative_humidity', 
    'solar_radiation', 'wind_speed', 'dew_point', 'weather_condition', 
    'temperature_humidity_index', 'merge_key_weather'
]

# First, scale all features using the updated scaler
scaler_features = [
    'hour', 'day_of_week', 'zone_id', 'zone_type', 'zone_capacity', 'zone_area_sq_m', 
    'zone_orientation', 'zone_function', 'occupancy_count', 'occupancy_pct', 
    'occupancy_t_minus_1h', 'occupancy_t_minus_2h', 'departing_flights_next_2h', 
    'arriving_flights_next_2h', 'baseline_cooling_load', 'actual_cooling_load_kWh', 
    'load_t_minus_1h', 'standard_setpoint', 'adjusted_setpoint', 'zoning_mode', 
    'occupancy_t-1', 'cooling_load_t-1', 'rolling_avg_occupancy_3h', 
    'rolling_max_occupancy_6h', 'rolling_std_cooling_3h', 'occupancy_delta', 
    'occupancy_rate', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 
    'occupancy_level', 'recommended_setpoint', 'is_weekend', 'sin_hour', 'cos_hour', 
    'sin_day_of_week', 'cos_day_of_week', 'solar_presence', 'external_temperature', 
    'relative_humidity', 'solar_radiation', 'wind_speed', 'dew_point', 
    'weather_condition', 'temperature_humidity_index', 'merge_key_weather'
]

print(f"Using {len(scaler_features)} features for scaling, {len(model_features)} for model")

# Standardize all features using scaler
df[scaler_features] = scaler.transform(df[scaler_features])

# ----- 3. Windowing for time series input -----
context_window = 48
# Use only the 46 features that the model expects (excluding actual_cooling_load_kWh)
X = df[model_features].values.astype(np.float32)
X_seq = []
indices = []
for i in range(len(df) - context_window):
    x_window = X[i : i + context_window].T # Shape: (features, context_window)
    X_seq.append(x_window)
    indices.append(df.index[i + context_window]) # The timestamp aligning with the "prediction time"

X_seq = np.stack(X_seq) # Shape (samples, features, context_window)

# ----- 4. Model Inference -----
# Create TensorDataset/DataLoader for efficient batch inference
X_seq_tensor = torch.tensor(X_seq)
infer_dataset = TensorDataset(X_seq_tensor) # Only inputs for inference
infer_dl = DataLoader(infer_dataset, batch_size=64, shuffle=False)

# Run inference with loaded model
model_device = learn_inf.model.parameters().__next__().device
preds_list = []

learn_inf.model.eval()
with torch.no_grad():
    for xb, in infer_dl:
        xb = xb.to(model_device)
        yb = learn_inf.model(xb)
        preds_list.append(yb.cpu().numpy())
preds = np.vstack(preds_list) # Shape: (samples, 3)

# ----- 5. Assemble Results -----
results_df = pd.DataFrame(
    preds,
    columns=['predicted_cooling_load_kWh_pred', 'optimized_cooling_load_kWh_pred', 'actual_cooling_load_kWh_pred']
)
results_df['timestamp'] = [str(idx) for idx in indices]
results_df.set_index('timestamp', inplace=True)

# Optional: Save to CSV
results_df.to_csv("model_predictions_on_inference_data.csv")
print(results_df.head())
