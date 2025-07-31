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

# Check what features the scaler expects
print("Features the scaler was trained on:")
scaler_features = scaler.feature_names_in_.tolist()
print(scaler_features)

print("\nAvailable columns in data:")
print(df.columns.tolist())

# Check which features are missing from the data
missing_features = [f for f in scaler_features if f not in df.columns]
if missing_features:
    print(f"\nMissing features in data: {missing_features}")

# Check which features exist in data but not in scaler
extra_features = [f for f in df.columns if f not in scaler_features]
if extra_features:
    print(f"\nExtra features in data (not in scaler): {extra_features}")

# Use only the features that the scaler was trained on and that exist in the data
available_features = [f for f in scaler_features if f in df.columns]
print(f"\nFeatures we can use: {len(available_features)} out of {len(scaler_features)}")

if len(available_features) != len(scaler_features):
    print("Warning: Not all scaler features are available in the data!")
    print("This may affect model performance.")

# Standardize input features using the exact feature order from the scaler
df[available_features] = scaler.transform(df[available_features])

# ----- 3. Windowing for time series input -----
context_window = 48
X = df[available_features].values.astype(np.float32)
X_seq = []
indices = []
for i in range(len(df) - context_window):
    x_window = X[i : i + context_window].T # Shape: (features, context_window)
    X_seq.append(x_window)
    indices.append(df.index[i + context_window]) # The timestamp aligning with the "prediction time"

X_seq = np.stack(X_seq) # Shape (samples, features, context_window)

print(f"\nInput shape for model: {X_seq.shape}")

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

print(f"\nPredictions shape: {preds.shape}")

# ----- 5. Assemble Results -----
results_df = pd.DataFrame(
    preds,
    columns=['predicted_cooling_load_kWh_pred', 'optimized_cooling_load_kWh_pred', 'actual_cooling_load_kWh_pred']
)
results_df['timestamp'] = [str(idx) for idx in indices]
results_df.set_index('timestamp', inplace=True)

# Optional: Save to CSV
results_df.to_csv("model_predictions_on_inference_data.csv")
print("\nFirst 10 predictions:")
print(results_df.head(10))
print(f"\nTotal predictions: {len(results_df)}")