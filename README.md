# ❄️ PatchTST Time Series Model for HVAC Occupancy & Weather Forecasting

This repository contains code, data, and models for training and deploying a multi-output **PatchTST Transformer** that forecasts cooling loads for HVAC systems using occupancy and weather data.

> **Deployed Model**: [Hugging Face – PatchTST for HVAC Forecasting](https://huggingface.co/ritulk/patchtst-model-based-on-occupancy-weather-data-time-series)

---

## 📁 Repository Structure

<img width="680" height="424" alt="image" src="https://github.com/user-attachments/assets/2613c92b-d8ce-4393-ab01-492569ade514" />


## 🏗️ Project Overview

**Goal:** Predict the following metrics at each time point:

- `predicted_cooling_load_kWh`
- `optimized_cooling_load_kWh`

*(Note: `actual_cooling_load_kWh` was previously a prediction target and is now typically used as input.)*

**Input Features:**

- **Temporal**: `timestamp`, `hour`, `day_of_week`, with sine/cosine encodings  
- **Zone Metadata**: `zone_id`, `type`, `function`, `orientation`, `capacity`, etc.  
- **Occupancy**: counts, percentage, rolling stats, lag values  
- **Weather**: `external_temperature`, `humidity`, `wind_speed`, `solar_radiation`, `weather_condition`  

---

## 📊 Data Preprocessing

### 🧩 Categorical Columns
Encoded with `LabelEncoder`, saved in `label_encoders_uptd.pkl`

```python
["zone_id", "zone_type", "zone_function", "weather_condition", 
 "zone_orientation", "solar_presence", "is_weekend", 
 "occupancy_level", "zoning_mode", "merge_key_weather"]
```
### 🔄 Scaling

All numeric features are standardized using StandardScaler → scaler_uptd.pkl

### 🪟 Sliding Window Framing

Window size: 48 timesteps

Input shape: (samples, 46 features, 48 timesteps)

80% training, 20% validation

### 🤖 Model Details
Model Architecture: PatchTST (via TSai)

Regression Head: Global pooling + Linear output layer

Outputs: Simultaneous predictions for:

predicted_cooling_load_kWh

optimized_cooling_load_kWh

Exported Model: patchtst_occupancy_cooling_model.pkl

### ⚙️ Training & Fine-Tuning Pipeline

Fit & save scalers and label encoders

Convert preprocessed data to windowed tensor format

Train the model using MSE/MAE loss

Evaluate and export model, encoders, and scalers

### 🏃 Inference Pipeline

Load:

patchtst_occupancy_cooling_model.pkl

label_encoders_uptd.pkl

scaler_uptd.pkl

Preprocess input data: encode, scale, window

Pass batch to model

Output:

predicted_cooling_load_kWh

optimized_cooling_load_kWh

### 📈 Model Evaluation

Validation Scores (after 10 epochs):

Metric	R² Score	Pearson r
predicted_cooling_load_kWh	0.371	0.609
optimized_cooling_load_kWh	0.454	0.673

For individual prediction results, see:
model_predictions_on_inference_data.csv

### 📊 Visualization

🔁 Predicted vs Optimized Cooling Loads

Below is a visualization comparing the model's predicted and optimized cooling load outputs.

<img width="521" height="404" alt="image" src="https://github.com/user-attachments/assets/64690c88-7091-4c8d-919d-327413f44816" />


### 🍃 How to Run

# Install requirements

pip install tsai fastai scikit-learn pandas numpy torch ipykernel

Prepare data: hvac_with_weather.csv

Run training: main.py

View results: predictions in CSV files, saved plots

For exploratory analysis: see HVAC_model_PatchTST.ipynb

### 🔒 Notes & Best Practices

Always use saved label_encoders_uptd.pkl and scaler_uptd.pkl — do not refit on new data

Update encoders only offline if unseen categories are present

For reproducibility, match tsai/fastai versions used during export

### 📚 Citation & Credits
Built using TSai and fastai

Please cite this work or these libraries if used in your research or product
